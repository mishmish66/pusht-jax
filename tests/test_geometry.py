import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pusht_jax import geometry as geo


BOX_PAIRS = [  # (polygon, polygon, intersection area)
    ([[34.9, 108.71], [-40.77, 88.78], [-31.6, 53.95], [44.07, 73.87]],
     [[84.37, 34.18], [98.5, 143.23], [24.35, 152.84], [10.23, 43.78]], 891.0634385406),
    ([[36.3, 36.04], [-25.44, 113.21], [-29.59, 109.89], [32.15, 32.72]],
     [[31.22, -11.35], [55.57, -4.86], [28.72, 95.89], [4.37, 89.4]], 193.7961650992),
    ([[71.22, 53.3], [75.16, 60.55], [58.22, 69.78], [54.27, 62.53]],
     [[100.13, 121.5], [51.36, 127.4], [36.98, 8.59], [85.75, 2.69]], 159.2636),
    ([[81.3, 100.55], [25.68, 37.38], [63.0, 4.52], [118.62, 67.69]],
     [[128.51, 46.15], [97.25, 72.18], [58.3, 25.41], [89.56, -0.62]], 1106.2846636116),
    ([[51.33, 11.76], [-16.83, 78.67], [-39.62, 55.46], [28.55, -11.45]],
     [[53.46, 2.68], [47.07, 24.02], [-7.33, 7.73], [-0.94, -13.62]], 724.4155303502),
    ([[72.21, 55.49], [28.84, 27.68], [43.86, 4.25], [87.23, 32.06]],
     [[79.93, 8.97], [3.03, 38.24], [-6.91, 12.13], [69.99, -17.14]], 310.3615092885),
    ([[39.04, 12.2], [111.03, -16.1], [151.87, 87.78], [79.89, 116.08]],
     [[123.01, 83.32], [69.13, 84.98], [66.78, 8.69], [120.66, 7.03]], 4114.0133477606),
    ([[38.98, 2.45], [129.56, 19.33], [118.17, 80.48], [27.59, 63.6]],
     [[70.18, 93.88], [-11.26, 130.92], [-47.19, 51.93], [34.25, 14.88]], 732.2891643768),
]


@pytest.mark.parametrize("a, b, area", BOX_PAIRS)
def test_convex_intersection_area(a, b, area):
    assert float(geo.convex_intersection_area(jnp.asarray(a), jnp.asarray(b))) == pytest.approx(area, abs=1e-8)


def test_intersection_of_identical_polygons_is_their_area():
    box = jnp.asarray([(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])
    assert geo.convex_intersection_area(box, box) == pytest.approx(12.0)


def test_polygons_sharing_an_edge_have_zero_intersection():
    left = jnp.asarray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert geo.convex_intersection_area(left, left + jnp.asarray([1.0, 0.0])) == pytest.approx(0.0)


def test_circle_inside_polygon_pushes_out_through_nearest_face():
    box = jnp.asarray([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    c = geo.circle_polygon_contact(jnp.asarray([5.0, 8.0]), 3.0, box)
    np.testing.assert_allclose(c.normal[0], [0.0, -1.0])
    np.testing.assert_allclose(c.point_b[0], [5.0, 10.0])
    assert bool(c.active[0])


def test_circle_beyond_radius_is_inactive():
    box = jnp.asarray([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    c = geo.circle_polygon_contact(jnp.asarray([14.0, 14.0]), 5.0, box)
    assert not bool(c.active[0])


def test_box_resting_flat_on_half_plane_gets_contacts_at_its_corners():
    box = jnp.asarray([(0.0, -1.0), (10.0, -1.0), (10.0, 5.0), (0.0, 5.0)])
    c = geo.half_plane_polygon_contact(jnp.asarray([0.0, 0.0]), jnp.asarray([0.0, 1.0]), box)
    np.testing.assert_allclose(c.normal, [[0.0, 1.0], [0.0, 1.0]])
    assert c.active.all()
    np.testing.assert_allclose(np.sort(c.point_b[:, 0]), [0.0, 10.0])
    np.testing.assert_allclose(geo.dot(c.point_b - c.point_a, c.normal), [-1.0, -1.0])


def test_box_behind_half_plane_is_pushed_out_along_its_normal():
    box = jnp.asarray([(0.0, -9.0), (10.0, -9.0), (10.0, -4.0), (0.0, -4.0)])
    c = geo.half_plane_polygon_contact(jnp.asarray([0.0, 0.0]), jnp.asarray([0.0, 1.0]), box)
    np.testing.assert_allclose(c.normal, [[0.0, 1.0], [0.0, 1.0]])
    assert c.active.all()
    np.testing.assert_allclose(geo.dot(c.point_b - c.point_a, c.normal), [-9.0, -9.0])


@pytest.mark.parametrize("angle", np.linspace(-np.pi, np.pi, 17))
def test_intersection_is_exact_where_edges_coincide_up_to_rounding(angle):
    bar = jnp.asarray([(-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)])
    stem = jnp.asarray([(-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)])
    pose = jnp.asarray([150.0, 340.0, angle])
    # The same boxes placed about another origin, so coincident edges differ by rounding.
    origin = jnp.asarray([0.0, 45.0])
    moved = jnp.concatenate([geo.transform(origin, pose), pose[2:]])
    for box in (bar, stem):
        area = geo.convex_intersection_area(geo.transform(box, pose), geo.transform(box - origin, moved))
        assert float(area) == pytest.approx(float(geo.polygon_area(box)), abs=1e-9)
    area = geo.convex_intersection_area(geo.transform(bar, pose), geo.transform(stem - origin, moved))
    assert float(area) == pytest.approx(0.0, abs=1e-9)
