"""Pallas rasterizer of the PushT pygame scene, resampled bilinearly to any size.

The kernel shades tiles of output pixels, skipping the goal, block and agent where a tile cannot see them. Off GPU it
runs in Pallas interpret mode.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from . import physics

CANVAS_SIZE = 512

WHITE = (255, 255, 255)
GOAL = (144, 238, 144)
WALL = (211, 211, 211)
AGENT = (65, 105, 225)
AGENT_INNER = (78, 126, 255)
BLOCK = (119, 136, 153)
BLOCK_FILL = (142, 163, 183)

AGENT_INNER_RADIUS = 11.0
OUTLINE_RADIUS = 2.0
# Wall centerlines and half-width.
WALLS = (
    ((5.0, 506.0), (5.0, 5.0)),
    ((5.0, 5.0), (506.0, 5.0)),
    ((506.0, 5.0), (506.0, 506.0)),
    ((5.0, 506.0), (506.0, 506.0)),
)
WALL_RADIUS = 2.0

TILE = (16, 16)


def _taps(i, size, dtype):
    """Canvas coordinates of the two taps resampling output index ``i``, and the second tap's weight."""
    f = (i.astype(dtype) + 0.5) * (CANVAS_SIZE / size) - 0.5
    i0 = jnp.floor(f)
    frac = jnp.where((f < 0) | (i0 >= CANVAS_SIZE - 1), 0.0, f - i0)
    i0 = jnp.maximum(i0, 0.0)
    return i0, jnp.minimum(i0 + 1.0, CANVAS_SIZE - 1.0), frac


def _tap_points(rows, cols, width, height, dtype):
    """The four taps (00, 01, 10, 11) of output pixels ``rows``, ``cols``, and their bilinear weights."""
    x0, x1, ax = _taps(cols, width, dtype)
    y0, y1, ay = _taps(rows, height, dtype)
    return [(x0, y0), (x1, y0), (x0, y1), (x1, y1)], ax, ay


def _in_convex(px, py, verts):
    inside = True
    for (ax, ay), (bx, by) in zip(verts, verts[1:] + verts[:1]):
        inside = inside & ((bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0)
    return inside


def _near_segment(px, py, a, b, radius):
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = (px - a[0]) * abx + (py - a[1]) * aby
    t = jnp.clip(t / jnp.maximum(abx * abx + aby * aby, jnp.finfo(px.dtype).tiny), 0.0, 1.0)
    dx, dy = px - (a[0] + t * abx), py - (a[1] + t * aby)
    return jnp.sqrt(dx * dx + dy * dy) <= radius


def _wall_class(px, py):
    """0 none, 1 wall."""
    c = jnp.zeros(px.shape, jnp.int32)
    for a, b in WALLS:
        c = jnp.where(_near_segment(px, py, a, b, WALL_RADIUS), 1, c)
    return c


def _goal_class(px, py, goal):
    """0 none, 1 goal."""
    c = jnp.zeros(px.shape, jnp.int32)
    for box in goal:
        c = jnp.where(_in_convex(px, py, box), 1, c)
    return c


def _block_class(px, py, block):
    """0 none, 1 fill, 2 outline."""
    c = jnp.zeros(px.shape, jnp.int32)
    for box in block:
        c = jnp.where(_in_convex(px, py, box), 1, c)
        near = False
        for a, b in zip(box, box[1:] + box[:1]):
            near = near | _near_segment(px, py, a, b, OUTLINE_RADIUS)
        c = jnp.where(near, 2, c)
    return c


def _agent_class(px, py, center):
    """0 none, 1 disc, 2 inner disc."""
    dx, dy = px - center[0], py - center[1]
    d2 = dx * dx + dy * dy
    c = jnp.where(d2 <= physics.AGENT_RADIUS**2, 1, jnp.zeros(px.shape, jnp.int32))
    return jnp.where(d2 <= AGENT_INNER_RADIUS**2, 2, c)


def _color(wall, goal, block, agent, ch):
    c = jnp.where(goal == 1, float(GOAL[ch]), float(WHITE[ch]))
    c = jnp.where(block == 1, float(BLOCK_FILL[ch]), c)
    c = jnp.where(block == 2, float(BLOCK[ch]), c)
    c = jnp.where(agent == 1, float(AGENT[ch]), c)
    c = jnp.where(agent == 2, float(AGENT_INNER[ch]), c)
    return jnp.where(wall == 1, float(WALL[ch]), c)


def _round_half_even(v):
    r = jnp.floor(v)
    d = v - r
    odd = r - 2.0 * jnp.floor(r * 0.5) == 1.0
    return jnp.where((d > 0.5) | ((d == 0.5) & odd), r + 1.0, r)


def _shade(walls, goals, blocks, agents, ax, ay):
    """Channels of the rounded bilinear blend of the four taps' colors; ``walls`` packs 1 bit per tap."""
    channels = []
    for ch in range(3):
        c00, c01, c10, c11 = (_color((walls >> k) & 1, goals[k], blocks[k], agents[k], ch) for k in range(4))
        top = (1 - ax) * c00 + ax * c01
        bottom = (1 - ax) * c10 + ax * c11
        channels.append(_round_half_even((1 - ay) * top + ay * bottom))
    return channels


def _grid(shape):
    return jax.lax.broadcasted_iota(jnp.int32, shape, 0), jax.lax.broadcasted_iota(jnp.int32, shape, 1)


@functools.cache
def _wall_layer(width, height, dtype):
    """Packed wall classes of each output pixel's taps, padded to whole tiles."""
    with jax.ensure_compile_time_eval():
        points, _, _ = _tap_points(*_grid((height, width)), width, height, dtype)
        packed = sum(_wall_class(px, py) << k for k, (px, py) in enumerate(points))
        return np.pad(np.asarray(packed), ((0, -height % TILE[0]), (0, -width % TILE[1])))


def _tee(s):
    return [[(s[8 * b + 2 * k], s[8 * b + 2 * k + 1]) for k in range(4)] for b in range(2)]


def _kernel(params_ref, walls_ref, out_ref, *, width, height):
    s = [params_ref[k] for k in range(42)]
    block, goal = _tee(s[:16]), _tee(s[16:32])
    center = (s[32], s[33])
    (bx_lo, bx_hi, by_lo, by_hi), (gx_lo, gx_hi, gy_lo, gy_hi) = s[34:38], s[38:42]
    rows, cols = _grid(TILE)
    points, ax, ay = _tap_points(pl.program_id(0) * TILE[0] + rows, pl.program_id(1) * TILE[1] + cols,
                                 width, height, params_ref.dtype)
    x_lo, x_hi, y_lo, y_hi = points[0][0].min(), points[3][0].max(), points[0][1].min(), points[3][1].max()

    def visible(lo_x, hi_x, lo_y, hi_y):
        return (x_hi >= lo_x) & (x_lo <= hi_x) & (y_hi >= lo_y) & (y_lo <= hi_y)

    def none():
        return [jnp.zeros(TILE, jnp.int32)] * 4

    r = physics.AGENT_RADIUS
    goals = jax.lax.cond(visible(gx_lo, gx_hi, gy_lo, gy_hi),
                         lambda: [_goal_class(px, py, goal) for px, py in points], none)
    blocks = jax.lax.cond(visible(bx_lo, bx_hi, by_lo, by_hi),
                          lambda: [_block_class(px, py, block) for px, py in points], none)
    agents = jax.lax.cond(visible(center[0] - r, center[0] + r, center[1] - r, center[1] + r),
                          lambda: [_agent_class(px, py, center) for px, py in points], none)
    red, green, blue = (c.astype(jnp.int32) for c in _shade(walls_ref[...], goals, blocks, agents, ax, ay))
    out_ref[...] = red | green << 8 | blue << 16


def _bounds(tee, margin):
    xs, ys = tee[..., 0], tee[..., 1]
    return jnp.stack([xs.min(), xs.max(), ys.min(), ys.max()]) + margin * jnp.array([-1, 1, -1, 1], tee.dtype)


def _render(block, goal, center, *, width, height, interpret):
    walls = _wall_layer(width, height, block.dtype)
    params = jnp.concatenate([block.ravel(), goal.ravel(), center, _bounds(block, OUTLINE_RADIUS), _bounds(goal, 0)])
    params = jnp.pad(params, (0, 64 - params.size))  # Triton blocks have power-of-two sizes.
    packed = pl.pallas_call(
        functools.partial(_kernel, width=width, height=height),
        out_shape=jax.ShapeDtypeStruct(walls.shape, jnp.int32),
        grid=(walls.shape[0] // TILE[0], walls.shape[1] // TILE[1]),
        in_specs=[pl.BlockSpec(params.shape, lambda i, j: (0,)), pl.BlockSpec(TILE, lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(TILE, lambda i, j: (i, j)),
        interpret=interpret,
    )(params, jnp.asarray(walls))[:height, :width]
    return jnp.stack([(packed >> (8 * ch)) & 255 for ch in range(3)], axis=-1).astype(jnp.uint8)


def render(agent_pos, block_boxes, goal_boxes, width, height):
    """RGB image (height, width, 3) uint8 of the scene."""
    block = jnp.round(block_boxes)
    goal = jnp.round(goal_boxes).astype(block.dtype)
    center = jnp.round(agent_pos).astype(block.dtype)
    kw = dict(width=width, height=height)
    return jax.lax.platform_dependent(
        block, goal, center,
        cuda=functools.partial(_render, **kw, interpret=False),
        default=functools.partial(_render, **kw, interpret=True),
    )
