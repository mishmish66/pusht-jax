import jax
import numpy as np
import pytest

from pusht_jax import PhysicsConfig, PushT, physics

KEY = jax.random.key(0)


def test_reset_to_state_places_origin_then_rotates_about_center_of_mass():
    env = PushT(physics=PhysicsConfig(dt=0.01, iterations=10, solve_first=False))
    np.testing.assert_allclose(env.block_pose(env.reset_to_state([0.0, 10.0, 20.0, 30.0, 1.0])),
                               [57.866194, 50.686396, 1.0], atol=1e-5)


def test_keypoints_follow_gym_pusht_vertex_order():
    env = PushT()
    pose = np.array([300.0, 250.0, 0.7])
    local = [(-60, 0), (60, 0), (60, 30), (-60, 30), (-15, 30), (15, 30), (15, 120), (-15, 120)]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    want = [(pose[0] + c * x - s * y, pose[1] + s * x + c * y) for x, y in local]
    np.testing.assert_allclose(env.keypoints(env.reset_to_pose([100.0, 100.0], pose)), want, atol=1e-9)


def test_agent_passes_through_walls():
    env = PushT()
    state = env.reset_to_pose([30.0, 256.0], [300.0, 300.0, 0.0])
    for _ in range(20):
        state = env.step(KEY, state, [-1.0, 0.0])
    left_wall = physics.WALL_PLANES[0][0][0]
    assert float(state.physics.agent_pos[0]) < left_wall


def test_block_is_stationary_without_contact():
    env = PushT()
    state = env.reset_to_pose([60.0, 60.0], [300.0, 300.0, 0.3])
    next_state = env.step(KEY, state, [-0.77, -0.73])
    np.testing.assert_array_equal(env.block_pose(next_state), env.block_pose(state))


@pytest.mark.parametrize("angle", [-np.pi / 2, np.pi / 2])
def test_block_stays_in_the_arena_when_driven_into_a_wall(angle):
    env = PushT()
    state = env.reset_to_pose([300.0, 256.0], [420.0, 256.0, angle])
    for _ in range(40):
        state = env.step(KEY, state, [1.0, 0.0])
    assert 0.0 < float(state.physics.block_q[0]) < 512.0
