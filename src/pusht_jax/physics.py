"""PushT rigid-body dynamics, following Chipmunk2D's space step.

A kinematic disc agent, a dynamic tee of two boxes, and static half-plane walls. Contacts are frictionless and
inelastic, resolved by sequential impulses with warm starting and bias velocities for penetration recovery.
"""

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import geometry as geo

AGENT_RADIUS = 15.0
# A point on each wall's inner surface and its inward unit normal.
WALL_PLANES = (
    ((7.0, 0.0), (1.0, 0.0)),
    ((0.0, 7.0), (0.0, 1.0)),
    ((504.0, 0.0), (-1.0, 0.0)),
    ((0.0, 504.0), (0.0, -1.0)),
)
# Body-frame vertices of the tee's two boxes (bar, then stem).
TEE = (
    ((-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)),
    ((-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)),
)
BLOCK_COM = (0.0, 45.0)  # body frame
BLOCK_MASS = 1.0
BLOCK_MOMENT = 3000.0  # about the center of mass
BLOCK_INV_MASS = (1.0 / BLOCK_MASS, 1.0 / BLOCK_MASS, 1.0 / BLOCK_MOMENT)

# Chipmunk defaults, from float32 literals as in pymunk.
COLLISION_SLOP = 0.10000000149011612  # float32(0.1)
COLLISION_BIAS = 0.0017970074436457143  # float32(0.9) ** 60
COLLISION_PERSISTENCE = 3

N_AGENT_PAIRS = len(TEE)
N_PAIRS = N_AGENT_PAIRS + len(WALL_PLANES) * len(TEE)
CONTACTS_PER_PAIR = 2


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = 0.1
    # Fraction of block velocity kept per second.
    damping: float = 0.0
    iterations: int = 1
    # Solve contacts before integrating positions.
    solve_first: bool = True


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ContactCache:
    """Most recent contacts of each shape pair: agent x boxes, then walls x boxes, in solve order."""

    impulse: jax.Array  # (N_PAIRS, CONTACTS_PER_PAIR)
    feature: jax.Array  # (N_PAIRS, CONTACTS_PER_PAIR) int32
    active: jax.Array  # (N_PAIRS, CONTACTS_PER_PAIR) bool
    age: jax.Array  # (N_PAIRS,) int32, steps since the pair last touched


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    agent_pos: jax.Array  # (2,)
    agent_vel: jax.Array  # (2,)
    block_q: jax.Array  # (3,), center of mass x, y and angle
    block_v: jax.Array  # (3,)
    block_v_bias: jax.Array  # (3,)
    block_com: jax.Array  # (2,), center of mass in the body frame
    cache: ContactCache


def block_vertices(block_q, com):
    """World vertices (2, 4, 2) of the tee at ``block_q`` with center of mass ``com``."""
    return geo.transform(jnp.asarray(TEE, block_q.dtype) - com, block_q)


def pose_to_q(pose, com):
    """``block_q`` of the block with its body origin at ``pose``."""
    return jnp.concatenate([geo.transform(com, pose), pose[2:]])


def q_to_pose(q, com):
    """Body-origin pose of the block at ``q``."""
    return jnp.concatenate([geo.transform(-com, q), q[2:]])


def point_velocity(v, r):
    """Velocity of the point at offset ``r`` from the center of mass."""
    return v[:2] + geo.perp(r) * v[2]


def init_state(agent_pos, block_q, block_com=BLOCK_COM) -> PhysicsState:
    """State with bodies at rest and no contact history."""
    agent_pos = jnp.asarray(agent_pos)
    dtype = agent_pos.dtype
    shape = (N_PAIRS, CONTACTS_PER_PAIR)
    return PhysicsState(
        agent_pos=agent_pos,
        agent_vel=jnp.zeros(2, dtype),
        block_q=jnp.asarray(block_q, dtype),
        block_v=jnp.zeros(3, dtype),
        block_v_bias=jnp.zeros(3, dtype),
        block_com=jnp.asarray(block_com, dtype),
        cache=ContactCache(
            impulse=jnp.zeros(shape, dtype),
            feature=jnp.zeros(shape, jnp.int32),
            active=jnp.zeros(shape, bool),
            age=jnp.full((N_PAIRS,), COLLISION_PERSISTENCE, jnp.int32),
        ),
    )


def detect_contacts(agent_pos, block_q, block_com) -> geo.Contact:
    """Contacts of every shape pair in solve order, stacked along a leading pair axis."""
    boxes = block_vertices(block_q, block_com)
    planes = jnp.asarray(WALL_PLANES, block_q.dtype)
    per_box = jax.vmap(geo.half_plane_polygon_contact, (None, None, 0))
    wall_contacts = jax.vmap(per_box, (0, 0, None))(planes[:, 0], planes[:, 1], boxes)
    wall_contacts = jax.tree.map(lambda x: jax.lax.collapse(x, 0, 2), wall_contacts)

    agent_contacts = jax.vmap(geo.circle_polygon_contact, (None, None, 0))(agent_pos, AGENT_RADIUS, boxes)
    pad = CONTACTS_PER_PAIR - 1
    agent_contacts = jax.tree.map(lambda x: jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2)), agent_contacts)
    return jax.tree.map(lambda a, w: jnp.concatenate([a, w]), agent_contacts, wall_contacts)


def step(state: PhysicsState, config: PhysicsConfig) -> PhysicsState:
    """Advance one timestep."""
    if config.solve_first:
        return integrate_positions(solve_contacts(state, config), config.dt)
    return solve_contacts(integrate_positions(state, config.dt), config)


def integrate_positions(state: PhysicsState, dt: float) -> PhysicsState:
    """Move bodies by their velocities plus bias velocities, then clear the bias velocities."""
    return copy.replace(
        state,
        agent_pos=state.agent_pos + state.agent_vel * dt,
        block_q=state.block_q + (state.block_v + state.block_v_bias) * dt,
        block_v_bias=jnp.zeros_like(state.block_v_bias),
    )


def solve_contacts(state: PhysicsState, config: PhysicsConfig) -> PhysicsState:
    """Resolve contacts at the current positions into new block velocities and bias velocities."""
    dt = config.dt
    q, cache = state.block_q, state.cache
    dtype = q.dtype
    inv_mass = jnp.asarray(BLOCK_INV_MASS, dtype)

    contacts = detect_contacts(state.agent_pos, q, state.block_com)
    active = contacts.active
    touched = active.any(axis=-1)

    # A contact inherits the impulse of the old contact with its feature if its pair touched within the
    # persistence window; the impulse is applied up front only if the pair touched last step.
    matches = (
        active[:, :, None]
        & cache.active[:, None, :]
        & (contacts.feature[:, :, None] == cache.feature[:, None, :])
        & (cache.age < COLLISION_PERSISTENCE)[:, None, None]
    )
    impulse = jnp.sum(jnp.where(matches, cache.impulse[:, None, :], 0.0), axis=-1)
    warm = touched & (cache.age == 0)

    # Pre-step: Jacobians, effective masses, bias velocities.
    n = contacts.normal
    r = contacts.point_b - q[:2]
    rn = geo.cross(r, n)
    jac = jnp.concatenate([n, rn[..., None]], axis=-1)
    dist = geo.dot(contacts.point_b - contacts.point_a, n)
    n_mass = jnp.where(active, 1.0 / (inv_mass[0] + inv_mass[2] * rn**2), 0.0)
    bias_coef = 1.0 - COLLISION_BIAS**dt
    # Separated contacts may close at most their gap this step.
    speculative = jnp.maximum(dist, 0.0) / dt
    bias = -bias_coef * jnp.minimum(0.0, dist + COLLISION_SLOP) / dt - speculative

    # Velocity of shape a: the agent's, or zero for walls.
    is_agent = (jnp.arange(N_PAIRS) < N_AGENT_PAIRS)[:, None, None]
    va = jnp.where(is_agent, jnp.broadcast_to(state.agent_vel, n.shape), 0.0)

    # Damp velocities, then warm start.
    v = state.block_v * jnp.asarray(config.damping, dtype) ** dt
    warm_impulse = jnp.where(warm[:, None], impulse, 0.0)
    v = v + inv_mass * jnp.sum(warm_impulse[..., None] * jac, axis=(0, 1))

    # Sequential impulses: each sweep visits every contact in order.
    def apply_impulse(body, contact):
        v, v_bias = body
        jac, r, va, n_mass, bias, speculative, jn, jb = contact
        n = jac[:2]
        jb_new = jnp.maximum(jb + (bias - geo.dot(point_velocity(v_bias, r), n)) * n_mass, 0.0)
        jn_new = jnp.maximum(jn - (geo.dot(point_velocity(v, r) - va, n) + speculative) * n_mass, 0.0)
        body = (v + inv_mass * (jn_new - jn) * jac, v_bias + inv_mass * (jb_new - jb) * jac)
        return body, (jn_new, jb_new)

    def flat(x):
        return jax.lax.collapse(x, 0, 2)

    fixed = tuple(map(flat, (jac, r, va, n_mass, bias, speculative)))

    def sweep(_, carry):
        body, jn, jb = carry
        body, (jn, jb) = jax.lax.scan(apply_impulse, body, (*fixed, jn, jb), unroll=True)
        return body, jn, jb

    jn = flat(impulse)
    (v, v_bias), jn, _ = jax.lax.fori_loop(
        0, config.iterations, sweep, ((v, jnp.zeros(3, dtype)), jn, jnp.zeros_like(jn))
    )
    jn = jn.reshape(impulse.shape)

    new_cache = ContactCache(
        impulse=jnp.where(touched[:, None], jn, cache.impulse),
        feature=jnp.where(touched[:, None], contacts.feature, cache.feature),
        active=jnp.where(touched[:, None], active, cache.active),
        age=jnp.where(touched, 0, jnp.minimum(cache.age + 1, COLLISION_PERSISTENCE)),
    )
    return copy.replace(state, block_v=v, block_v_bias=v_bias, cache=new_cache)
