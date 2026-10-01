"""PushT as ``jax_pomdps`` POMDPs.

A PD-controlled disc pushes a tee onto a goal pose; actions in ``[-1, 1]^2`` are its targets across the arena.
Dynamics, rewards and observations are deterministic, so keys are unused.
"""

import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax_pomdps import Key, register
from jax_pomdps.spaces import Box, Image

from . import geometry as geo
from . import physics as phys
from . import render

GOAL_POSE = (256.0, 256.0, jnp.pi / 4)
ARENA = 512.0  # side of the square arena, in pixels
# Range of a random goal's origin on each axis, keeping the goal inside the walls.
GOAL_RANGE = (130.0, 382.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State:
    """Physics and the episode's goal pose."""

    physics: phys.PhysicsState
    goal_pose: jax.Array  # (3,)


def goal_boxes(goal_pose):
    """World vertices (2, 4, 2) of the tee at ``goal_pose``."""
    return geo.transform(jnp.asarray(phys.TEE, goal_pose.dtype), goal_pose)


def coverage(state: State) -> jax.Array:
    """Fraction of the goal area covered by the block."""
    boxes = phys.block_vertices(state.physics.block_q, state.physics.block_com)
    goal = goal_boxes(state.goal_pose)
    pairs = [geo.convex_intersection_area(b, g) for b in boxes for g in goal]
    goal_area = sum(geo.polygon_area(g) for g in goal)
    return sum(pairs) / goal_area


@dataclass(frozen=True)
class PushT:
    # "state": agent x, y, block x, y, angle. "keypoints": block vertices (8, 2), then agent x, y. "pixels": image.
    obs_type: Literal["state", "keypoints", "pixels"] = "state"
    # Draw each episode's goal pose; "state" then appends goal x, y, angle and "keypoints" the goal vertices (8, 2).
    randomize_goal: bool = False
    # Each episode shifts the unobserved center of mass uniformly by up to this many pixels per axis.
    com_noise: float = 0.0
    success_threshold: float = 0.95
    control_hz: int = 10
    k_p: float = 100.0
    k_v: float = 20.0
    # PD controller timestep; divides physics.dt.
    pd_dt: float = 0.01
    physics: phys.PhysicsConfig = field(default_factory=phys.PhysicsConfig)
    observation_width: int = 96
    observation_height: int = 96

    @property
    def substeps(self) -> int:
        return int(1 / (self.physics.dt * self.control_hz))

    @property
    def pd_steps(self) -> int:
        return round(self.physics.dt / self.pd_dt)

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, (2,))

    @property
    def observation_space(self) -> Box | Image:
        match self.obs_type:
            case "state":
                poses = 2 if self.randomize_goal else 1
                high = [ARENA] * 2 + [ARENA, ARENA, 2 * jnp.pi] * poses
                return Box([0.0] * len(high), high)
            case "keypoints":
                return Box(0.0, ARENA, (34 if self.randomize_goal else 18,))
            case "pixels":
                return Image(self.observation_height, self.observation_width)
            case _:
                raise ValueError(f"Unknown obs_type {self.obs_type!r}")

    # --- Queries ------------------------------------------------------------------------

    def block_pose(self, state: State) -> jax.Array:
        """``[x, y, angle]`` of the block's body origin, angle unwrapped."""
        return phys.q_to_pose(state.physics.block_q, state.physics.block_com)

    def keypoints(self, state: State) -> jax.Array:
        """(8, 2) vertices, bar box then stem box."""
        return phys.block_vertices(state.physics.block_q, state.physics.block_com).reshape(-1, 2)

    def render(self, state: State, width: int = 680, height: int = 680) -> jax.Array:
        boxes = phys.block_vertices(state.physics.block_q, state.physics.block_com)
        return render.render(state.physics.agent_pos, boxes, goal_boxes(state.goal_pose), width, height)

    # --- POMDP --------------------------------------------------------------------------

    def reset(self, key: Key) -> State:
        """``reset_to_state`` of agent in [50, 450)^2, block in [100, 400)^2 (integers), angle in [-pi, pi).

        A random goal has its origin in ``GOAL_RANGE``^2 and angle in [-pi, pi).
        """
        k_agent, k_block, k_angle, k_goal, k_com = jax.random.split(key, 5)
        agent = jax.random.randint(k_agent, (2,), 50, 450)
        block = jax.random.randint(k_block, (2,), 100, 400)
        angle = jax.random.uniform(k_angle, (1,), minval=-jnp.pi, maxval=jnp.pi)
        dtype = angle.dtype
        goal = jnp.asarray(GOAL_POSE, dtype)
        if self.randomize_goal:
            lo, hi = GOAL_RANGE
            goal = jax.random.uniform(k_goal, (3,), dtype, jnp.array([lo, lo, -jnp.pi]), jnp.array([hi, hi, jnp.pi]))
        com = jnp.asarray(phys.BLOCK_COM, dtype)
        com = com + jax.random.uniform(k_com, (2,), dtype, -self.com_noise, self.com_noise)
        return self.reset_to_state(jnp.concatenate([agent.astype(dtype), block.astype(dtype), angle]), goal, com)

    def step(self, key: Key, state: State, action: ArrayLike) -> State:
        target = (jnp.asarray(action, state.physics.agent_pos.dtype) + 1.0) * (ARENA / 2)

        def pd(carry, _):
            pos, vel = carry
            vel = vel + (self.k_p * (target - pos) - self.k_v * vel) * self.pd_dt
            return (pos + vel * self.pd_dt, vel), vel

        def substep(p, _):
            # The kinematic agent moves at its mean PD velocity over the physics step.
            (_, vel), vels = jax.lax.scan(pd, (p.agent_pos, p.agent_vel), length=self.pd_steps, unroll=True)
            moved = phys.step(copy.replace(p, agent_vel=vels.mean(0)), self.physics)
            return copy.replace(moved, agent_vel=vel), None

        return copy.replace(state, physics=jax.lax.scan(substep, state.physics, length=self.substeps)[0])

    def reward(self, key: Key, state: State, action: ArrayLike, next_state: State) -> jax.Array:
        """Coverage minus one: negative until the goal is covered."""
        return coverage(next_state) - 1.0

    def observe(self, key: Key, next_state: State, action: ArrayLike) -> jax.Array:
        match self.obs_type:
            case "state":
                poses = [self.block_pose(next_state)]
                if self.randomize_goal:
                    poses.append(next_state.goal_pose)
                wrapped = [jnp.concatenate([p[:2], jnp.mod(p[2:], 2 * jnp.pi)]) for p in poses]
                obs = jnp.concatenate([next_state.physics.agent_pos, *wrapped])
            case "keypoints":
                parts = [self.keypoints(next_state).ravel(), next_state.physics.agent_pos]
                if self.randomize_goal:
                    parts.append(goal_boxes(next_state.goal_pose).ravel())
                obs = jnp.concatenate(parts)
            case "pixels":
                return self.render(next_state, self.observation_width, self.observation_height)
            case _:
                raise ValueError(f"Unknown obs_type {self.obs_type!r}")
        return obs.astype(jnp.float32)

    def done(self, state: State) -> jax.Array:
        return coverage(state) > self.success_threshold

    # --- Placement ----------------------------------------------------------------------

    def reset_to_state(self, state5: ArrayLike, goal_pose: ArrayLike = GOAL_POSE,
                       block_com: ArrayLike = phys.BLOCK_COM) -> State:
        """gym-pusht's reset from ``[agent_x, agent_y, block_x, block_y, block_angle]``.

        Origin placed at angle zero, rotated about the center of mass, then one physics step.
        """
        state5 = jnp.asarray(state5)
        com = jnp.asarray(block_com, state5.dtype)
        q = state5[2:] + jnp.concatenate([com, jnp.zeros(1, state5.dtype)])
        physics = phys.step(phys.init_state(state5[:2], q, com), self.physics)
        return State(physics, jnp.asarray(goal_pose, state5.dtype))

    def reset_to_pose(self, agent_pos: ArrayLike, block_pose: ArrayLike, goal_pose: ArrayLike = GOAL_POSE,
                      block_com: ArrayLike = phys.BLOCK_COM) -> State:
        """Agent at ``agent_pos``, block origin exactly at ``block_pose``, at rest."""
        agent_pos = jnp.asarray(agent_pos)
        com = jnp.asarray(block_com, agent_pos.dtype)
        q = phys.pose_to_q(jnp.asarray(block_pose, agent_pos.dtype), com)
        return State(phys.init_state(agent_pos, q, com), jnp.asarray(goal_pose, agent_pos.dtype))


# Registered as "push-t", then a task variant, then an observation variant; "/com" shifts the center of mass up to the
# stem's half-width.
TASK_VARIANTS = {
    "": {},
    "/com": {"com_noise": 15.0},
    "/goal": {"randomize_goal": True},
    "/com-goal": {"com_noise": 15.0, "randomize_goal": True},
}
OBSERVATION_VARIANTS = {"": "state", "/kpt": "keypoints", "/pix": "pixels"}
for _task, _kwargs in TASK_VARIANTS.items():
    for _obs, _obs_type in OBSERVATION_VARIANTS.items():
        register(f"push-t{_task}{_obs}", partial(PushT, obs_type=_obs_type, **_kwargs))
