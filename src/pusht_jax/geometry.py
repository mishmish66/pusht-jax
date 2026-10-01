"""Planar convex geometry: rigid motions, intersection area, Chipmunk2D contacts.

Polygons are ``(n, 2)`` counter-clockwise vertex arrays, y up.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def dot(a, b):
    return jnp.sum(a * b, axis=-1)


def perp(v):
    """Rotate by +90 degrees."""
    return jnp.stack([-v[..., 1], v[..., 0]], axis=-1)


def rperp(v):
    """Rotate by -90 degrees."""
    return jnp.stack([v[..., 1], -v[..., 0]], axis=-1)


def rotate(v, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def transform(points, pose):
    """Apply ``pose`` ``[x, y, angle]`` to ``points`` (..., 2)."""
    return pose[:2] + rotate(points, pose[2])


def length(v):
    return jnp.sqrt(dot(v, v))


def normalize(v):
    return v / jnp.maximum(length(v), jnp.finfo(v.dtype).tiny)[..., None]


def closest_point_on_segment(p, a, b):
    ab = b - a
    t = jnp.clip(dot(p - a, ab) / jnp.maximum(dot(ab, ab), jnp.finfo(ab.dtype).tiny), 0.0, 1.0)
    return a + t[..., None] * ab


def edge_normals(verts):
    """Outward unit normal of each edge ``verts[i] -> verts[i + 1]``."""
    return normalize(rperp(jnp.roll(verts, -1, axis=0) - verts))


# --- Area of intersection ---------------------------------------------------------------


def polygon_area(verts):
    return 0.5 * jnp.sum(cross(verts, jnp.roll(verts, -1, axis=0)))


def _inside_part_integral(u, enters, parallel, parallel_ok, verts, edges, crossings):
    """Sum of ``cross(start, end) / 2`` over the inside part of each edge (rows) of a polygon clipped by lines (cols).

    Edge ``i`` crosses line ``j`` at parameter ``u[i, j]`` and point ``crossings[i, j]``, entering if ``enters``. A
    parallel edge lies inside only if ``parallel_ok``.
    """
    lo = jnp.where(~parallel & enters, u, -jnp.inf)
    hi = jnp.where(~parallel & ~enters, u, jnp.inf)
    u_lo, u_hi = jnp.max(lo, axis=1), jnp.min(hi, axis=1)
    rows = jnp.arange(u.shape[0])
    start = jnp.where((u_lo > 0)[:, None], crossings[rows, jnp.argmax(lo, axis=1)], verts)
    end = jnp.where((u_hi < 1)[:, None], crossings[rows, jnp.argmin(hi, axis=1)], verts + edges)
    ok = (jnp.maximum(u_lo, 0.0) < jnp.minimum(u_hi, 1.0)) & jnp.all(~parallel | parallel_ok, axis=1)
    return 0.5 * jnp.sum(jnp.where(ok, cross(start, end), 0.0))


def convex_intersection_area(poly_a, poly_b):
    """Boundary integral over the parts of each polygon's edges inside the other.

    Both polygons' parts share every crossing point and side test, so they close into one loop even where edges nearly
    coincide. An edge shared by both counts once, from ``poly_a``, if the edges run the same way.
    """
    ref = jnp.mean(poly_b, axis=0)
    a, b = poly_a - ref, poly_b - ref
    e = jnp.roll(a, -1, axis=0) - a
    f = jnp.roll(b, -1, axis=0) - b
    # (n, m): side of a's vertex i of b's line j, positive inside, and its rate along a's edge i; computed once.
    side, rate = jax.lax.optimization_barrier((cross(f[None], a[:, None] - b[None]), cross(f[None], e[:, None])))
    parallel = rate == 0
    t = -side / jnp.where(parallel, 1.0, rate)
    x = a[:, None] + t[..., None] * e[:, None]
    s = dot(x - b[None], f[None]) / dot(f, f)[None]
    same_way = dot(e[:, None], f[None]) > 0
    a_ok = (side > 0) | ((side == 0) & same_way)
    b_ok = jnp.where(same_way, side < 0, side > 0)
    return _inside_part_integral(t, rate > 0, parallel, a_ok, a, e, x) + _inside_part_integral(
        s.T, rate.T < 0, parallel.T, b_ok.T, b, f, jnp.swapaxes(x, 0, 1)
    )


# --- Contacts ---------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Contact:
    normal: jax.Array  # (n, 2), from shape a to shape b
    point_a: jax.Array  # (n, 2), on the surface of shape a
    point_b: jax.Array  # (n, 2), on the surface of shape b
    active: jax.Array  # (n,) bool
    feature: jax.Array  # (n,) int32, feature-pair id for warm starting


def circle_polygon_contact(center, radius, verts):
    """Contact between a circle (shape a) and a convex polygon (shape b)."""
    normals = edge_normals(verts)
    seps = dot(center - verts, normals)
    face = jnp.argmax(seps)
    sep = seps[face]
    inside = sep <= 0

    closest = closest_point_on_segment(center, verts, jnp.roll(verts, -1, axis=0))
    dists = length(closest - center)
    nearest = jnp.argmin(dists)

    normal = jnp.where(inside, -normals[face], normalize(closest[nearest] - center))
    point_b = jnp.where(inside, center - sep * normals[face], closest[nearest])
    dist = jnp.where(inside, sep, dists[nearest])
    return Contact(
        normal=normal[None],
        point_a=(center + radius * normal)[None],
        point_b=point_b[None],
        active=(dist <= radius)[None],
        feature=jnp.zeros((1,), jnp.int32),
    )


def half_plane_polygon_contact(origin, normal, verts):
    """Contact between the half-plane behind ``origin`` facing unit ``normal`` (shape a) and a convex polygon (shape b).

    The two deepest vertices are always contacts, speculative while separated; the normal points out of the half-plane.
    """
    depth = dot(verts - origin, normal)
    first = jnp.argmin(depth)
    deepest = jnp.stack([first, jnp.argmin(depth.at[first].set(jnp.inf))])
    point_b, d = verts[deepest], depth[deepest]
    return Contact(
        normal=jnp.broadcast_to(normal, point_b.shape),
        point_a=point_b - d[:, None] * normal,
        point_b=point_b,
        active=jnp.ones(2, bool),
        feature=deepest.astype(jnp.int32),
    )
