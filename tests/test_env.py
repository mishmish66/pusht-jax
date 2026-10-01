from pathlib import Path

import jax
import jax.numpy as jnp
import jax_pomdps
import numpy as np
import pytest

from pusht_jax import ARENA, GOAL_POSE, PhysicsConfig, PushT, physics
from pusht_jax.env import GOAL_RANGE, coverage, goal_boxes

KEY = jax.random.key(0)
CHIPMUNK = PhysicsConfig(dt=0.01, iterations=10, solve_first=False)


def rollout(env, state, actions):
    """States after each of ``actions``."""

    def body(s, a):
        s = env.step(KEY, s, a)
        return s, s

    return jax.lax.scan(body, state, jnp.asarray(actions))[1]


@pytest.mark.parametrize("task, kwargs", [
    ("", {}),
    ("/com", {"com_noise": 15.0}),
    ("/goal", {"randomize_goal": True}),
    ("/com-goal", {"com_noise": 15.0, "randomize_goal": True}),
])
@pytest.mark.parametrize("obs, obs_type", [("", "state"), ("/kpt", "keypoints"), ("/pix", "pixels")])
def test_make_returns_registered_renderable_pomdp(task, kwargs, obs, obs_type):
    env = jax_pomdps.make(f"push-t{task}{obs}")
    assert isinstance(env, jax_pomdps.POMDP) and isinstance(env, jax_pomdps.Renderable)
    assert env == PushT(obs_type=obs_type, **kwargs)


def test_reset_to_pose_places_block_origin_exactly():
    env = PushT()
    pose = [200.0, 300.0, 1.3]
    state = env.reset_to_pose([50.0, 50.0], pose)
    np.testing.assert_allclose(env.block_pose(state), pose, atol=1e-12)
    np.testing.assert_allclose(env.observe(KEY, state, None)[2:], pose, atol=1e-4)


@pytest.mark.parametrize("pose, expected", [
    (GOAL_POSE, 1.0),
    ((276.0, 256.0, np.pi / 4), 0.524744039731),
    ((256.0, 256.0, 0.0), 0.323966707567),
    ((300.0, 220.0, 1.2), 0.15608081123),
    ((180.0, 330.0, -2.5), 0.442312367653),
    ((400.0, 400.0, 0.3), 0.0),
])
def test_coverage(pose, expected):
    state = PushT().reset_to_pose((30.0, 30.0), pose)
    assert float(coverage(state)) == pytest.approx(expected, abs=1e-9)


def test_reward_is_coverage_minus_one_and_done_at_goal():
    env = PushT()
    for pose, done in [(GOAL_POSE, True), ((276.0, 256.0, np.pi / 4), False)]:
        state = env.reset_to_pose((30.0, 30.0), pose)
        next_state = env.step(KEY, state, [-0.9, -0.9])
        reward = float(env.reward(KEY, state, [-0.9, -0.9], next_state))
        assert reward == pytest.approx(float(coverage(next_state)) - 1.0)
        assert bool(env.done(next_state)) == done


def test_actions_span_the_arena():
    env = PushT()
    state = env.reset_to_pose((256.0, 256.0), (256.0, 450.0, 0.0))
    for corner, target in [([-1.0, -1.0], [0.0, 0.0]), ([1.0, 1.0], [ARENA, ARENA])]:
        for _ in range(30):
            state = env.step(KEY, state, corner)
        np.testing.assert_allclose(state.physics.agent_pos, target, atol=1.0)


def test_reset_samples_within_ranges():
    env = PushT()
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(1), 256))
    obs = jax.vmap(env.observe, (None, 0, None))(KEY, states, None)
    assert ((obs[:, :2] >= 50) & (obs[:, :2] < 450)).all()
    assert (obs[:, 4] >= 0).all() and (obs[:, 4] < 2 * np.pi).all()
    assert np.unique(obs[:, 0]).size > 50


@pytest.mark.parametrize("randomize_goal", [False, True])
@pytest.mark.parametrize("obs_type", ["state", "keypoints", "pixels"])
def test_vmapped_jitted_observations_lie_in_observation_space(obs_type, randomize_goal):
    env = PushT(obs_type=obs_type, randomize_goal=randomize_goal, com_noise=10.0)
    batch, steps = 4, 3
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(KEY, batch))
    actions = jnp.zeros((batch, steps, 2))
    final = jax.jit(jax.vmap(lambda s, a: jax.tree.map(lambda x: x[-1], rollout(env, s, a))))(states, actions)
    obs = jax.jit(jax.vmap(env.observe, (None, 0, None)))(KEY, final, None)
    space = env.observation_space
    assert obs.shape == (batch, *space.shape) and obs.dtype == space.dtype
    assert all(bool(space.contains(o)) for o in obs)


@pytest.mark.parametrize("solve_first", [False, True])
@pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
def test_agent_path_is_independent_of_physics_timestep(dt, solve_first):
    base = PushT(physics=CHIPMUNK)
    coarse = PushT(physics=PhysicsConfig(dt=dt, iterations=2, solve_first=solve_first))
    s_base = base.reset_to_pose([100.0, 100.0], [400.0, 400.0, 0.0])
    s_coarse = coarse.reset_to_pose([100.0, 100.0], [400.0, 400.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = rng.uniform(-1, 1, 2)
        s_base = base.step(KEY, s_base, action)
        s_coarse = coarse.step(KEY, s_coarse, action)
        np.testing.assert_allclose(s_coarse.physics.agent_pos, s_base.physics.agent_pos, atol=1e-9)
        np.testing.assert_allclose(s_coarse.physics.agent_vel, s_base.physics.agent_vel, atol=1e-9)


def test_solve_first_block_moves_on_the_step_it_is_pushed():
    env = PushT()
    # Agent overlapping the bar's lower edge, moving up into it.
    state = env.reset_to_pose([256.0, 290.0], [256.0, 300.0, 0.0])
    next_state = env.step(KEY, state, [0.0, 0.4])
    assert float(env.block_pose(next_state)[1]) > float(env.block_pose(state)[1])


def test_demos_are_done_exactly_on_their_last_step():
    demos = np.load(Path(__file__).parents[1] / "demos.npz")
    env = PushT()
    for i in range(8):
        n = demos["lengths"][i]
        state = env.reset_to_pose(demos["agent_pos"][i, 0], demos["block_pose"][i, 0])
        dones = jax.vmap(env.done)(jax.jit(rollout, static_argnums=0)(env, state, demos["action"][i, : n - 1]))
        np.testing.assert_array_equal(dones, np.arange(n - 1) == n - 2)


def test_goal_and_center_of_mass_are_fixed_by_default():
    states = jax.vmap(PushT().reset)(jax.random.split(KEY, 16))
    goal = np.asarray(GOAL_POSE, states.goal_pose.dtype)
    np.testing.assert_array_equal(states.goal_pose, np.broadcast_to(goal, (16, 3)))
    np.testing.assert_array_equal(states.physics.block_com, np.broadcast_to(physics.BLOCK_COM, (16, 2)))


def test_random_goals_vary_and_lie_inside_the_walls():
    goals = jax.vmap(PushT(randomize_goal=True).reset)(jax.random.split(KEY, 1024)).goal_pose
    verts = jax.vmap(goal_boxes)(goals)
    assert ((verts > 7.0) & (verts < 504.0)).all()
    assert ((goals[:, :2] >= GOAL_RANGE[0]) & (goals[:, :2] < GOAL_RANGE[1])).all()
    assert np.unique(goals[:, 2]).size > 1000


def test_random_goal_is_observed_and_scored():
    goal = jnp.asarray([150.0, 340.0, 2.0])
    for obs_type, expected in [("state", [150.0, 340.0, 2.0]), ("keypoints", goal_boxes(goal).ravel())]:
        env = PushT(obs_type=obs_type, randomize_goal=True)
        state = env.reset_to_pose([30.0, 30.0], goal, goal_pose=goal)
        obs = env.observe(KEY, state, None)
        np.testing.assert_allclose(obs[-len(expected):], expected, atol=1e-4)
        assert float(coverage(state)) == pytest.approx(1.0)
        assert bool(env.done(state))


def test_random_centers_of_mass_lie_within_the_noise_and_vary():
    coms = jax.vmap(PushT(com_noise=10.0).reset)(jax.random.split(KEY, 256)).physics.block_com
    offset = np.asarray(coms) - np.asarray(physics.BLOCK_COM)
    assert (np.abs(offset) <= 10.0).all()
    assert np.unique(offset[:, 0]).size > 200


def test_center_of_mass_leaves_the_block_geometry_unchanged():
    env = PushT()
    pose = [200.0, 300.0, 1.3]
    shifted = env.reset_to_pose([50.0, 50.0], pose, block_com=(8.0, 60.0))
    np.testing.assert_allclose(env.block_pose(shifted), pose, atol=1e-12)
    np.testing.assert_allclose(env.keypoints(shifted), env.keypoints(env.reset_to_pose([50.0, 50.0], pose)), atol=1e-12)


def test_centers_of_mass_on_either_side_turn_a_centered_push_opposite_ways():
    env = PushT()
    turns = []
    for com_x in (-10.0, 10.0):
        state = env.reset_to_pose([256.0, 290.0], [256.0, 300.0, 0.0], block_com=(com_x, 45.0))
        for _ in range(3):
            state = env.step(KEY, state, [0.0, 0.4])
        turns.append(float(env.block_pose(state)[2]))
    assert turns[0] * turns[1] < 0
