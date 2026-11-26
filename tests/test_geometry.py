import unittest

import numpy as np

from bodies import ArmorPlate, create_cylinder_body, create_plate_body


class PlateGeometryTests(unittest.TestCase):
    def test_requested_dimensions_survive_sampling_and_rotation(self):
        cases = (
            (0.080, 0.090, 0.023, 0, None),
            (0.160, 0.090, 0.023, 37, None),
            (0.045, 0.075, 0.019, 60, 0.012),
            (0.001, 0.001, 0.001, 0, None),
        )
        for width, height, thickness, angle_deg, spacing in cases:
            with self.subTest(dimensions=(width, height, thickness),
                              angle=angle_deg, spacing=spacing):
                pos = create_plate_body(
                    width, height, thickness, angle_deg, 'steel',
                    spacing=spacing)[0]
                angle = np.radians(angle_deg)
                relative_x = pos[:, 0].astype(np.float64) - 0.22
                local_width = relative_x * np.sin(angle) + pos[:, 2] * np.cos(angle)
                local_depth = relative_x * np.cos(angle) - pos[:, 2] * np.sin(angle)
                actual = (np.ptp(local_width), np.ptp(pos[:, 1]), np.ptp(local_depth))
                np.testing.assert_allclose(
                    actual, (width, height, thickness), rtol=0, atol=1e-7,
                    err_msg='Plate dimensions must be independent of lattice rounding')

    def test_equal_thickness_is_preserved_when_face_dimensions_change(self):
        thickness = 0.031
        for width, height in ((0.070, 0.110), (0.095, 0.110), (0.190, 0.160)):
            with self.subTest(width=width, height=height):
                pos = create_plate_body(width, height, thickness, 0, 'steel')[0]
                self.assertAlmostEqual(
                    float(np.ptp(pos[:, 0])), thickness, delta=1e-7,
                    msg=f'{width} m by {height} m plate must retain its requested thickness')


class CurvedPlateGeometryTests(unittest.TestCase):
    def test_signed_bend_follows_an_arc_and_preserves_normal_thickness(self):
        width, height, thickness = 0.120, 0.080, 0.020
        for axis, span in (('width', width), ('height', height)):
            for bend in (-90, 90):
                with self.subTest(axis=axis, bend=bend):
                    pos = create_plate_body(width, height, thickness, 0, 'steel', spacing=0.010,
                                            bend_axis=axis, bend_deg=bend)[0].reshape(3, 9, 13, 3)
                    layers = pos[:, 4, :, :] if axis == 'width' else pos[:, :, 6, :]
                    transverse = 2 if axis == 'width' else 1
                    radius = span / np.radians(bend)
                    center_x = 0.22 + thickness / 2 + radius
                    mid_layer_radius = np.hypot(layers[1, :, 0] - center_x, layers[1, :, transverse])
                    np.testing.assert_allclose(mid_layer_radius, abs(radius), rtol=0, atol=1e-7,
                                               err_msg='The middle layer must lie on the requested arc')
                    bend_rise = layers[1, 0, 0] - layers[1, layers.shape[1] // 2, 0]
                    self.assertGreater(np.sign(bend) * bend_rise, 0,
                                       'The bend sign must select the direction of the arc')
                    ray_lengths = np.linalg.norm(layers[-1] - layers[0], axis=1)
                    np.testing.assert_allclose(ray_lengths, thickness, rtol=0, atol=1e-7,
                                               err_msg='Thickness must remain constant along each local normal')

    def test_bend_updates_spring_lengths_without_dropping_lattice_bonds_or_anchors(self):
        args = (0.114, 0.080, 0.036, 27, 'steel')
        flat = create_plate_body(*args, spacing=0.012)
        pos, _, _, springs, rest_lengths, _, static = create_plate_body(*args, spacing=0.012, bend_deg=175)
        np.testing.assert_array_equal(springs, flat[3], err_msg='Bending must preserve the flat bond graph')
        np.testing.assert_array_equal(static, flat[6], err_msg='Bending must keep the same anchored edges')
        self.assertFalse(np.array_equal(rest_lengths, flat[4]), 'Bending must update spring rest lengths')
        np.testing.assert_allclose(rest_lengths, np.linalg.norm(pos[springs[:, 1]] - pos[springs[:, 0]], axis=1),
                                   rtol=1e-6, atol=1e-9, err_msg='Bonds must start at their curved geometry lengths')


class PlateOrientationTests(unittest.TestCase):
    def test_euler_axes_preserve_legacy_yaw_and_use_independent_pitch_and_roll(self):
        args = (0.080, 0.060, 0.016)
        pivot = np.array((0.22, 0, 0))
        local = create_plate_body(*args, 0, 'steel')[0] - pivot
        x, y, z = local.T
        cases = (
            ('yaw', 90, {}, (z, y, -x)),
            ('pitch', 0, {'pitch_deg': 90}, (-y, x, z)),
            ('roll', 0, {'roll_deg': 90}, (x, -z, y)),
            ('roll, pitch, and yaw', 90, {'pitch_deg': 90, 'roll_deg': 90}, (y, x, -z)),
        )
        for orientation, yaw, angles, expected in cases:
            with self.subTest(orientation=orientation):
                pos = create_plate_body(*args, yaw, 'steel', **angles)[0]
                np.testing.assert_allclose(pos - pivot, np.column_stack(expected), rtol=0, atol=1e-7,
                                           err_msg=f'{orientation} must use the documented plate axes and order')

    def test_rigid_pitch_and_roll_preserve_plate_spring_parameters(self):
        for bend in (0, 75):
            with self.subTest(bend=bend):
                args = (0.080, 0.060, 0.016, 34, 'steel')
                original = create_plate_body(*args, bend_deg=bend)
                rotated = create_plate_body(*args, bend_deg=bend, pitch_deg=82, roll_deg=-53)
                for array_index, metric in ((3, 'indices'), (4, 'rest lengths'), (5, 'stiffness')):
                    np.testing.assert_array_equal(rotated[array_index], original[array_index],
                                                  err_msg=f'Rigid rotations must preserve spring {metric}')

    def test_rebuilding_an_edited_plate_preserves_its_moved_center(self):
        plate = ArmorPlate()
        plate.pos += (0.09, -0.04, 0.05)
        plate.rest_pos[:] = plate.pos
        center = plate.pos.mean(axis=0)
        plate.angle = 135
        plate.pitch = 25
        plate.roll = -35
        plate.bend = 80
        plate.bend_axis = 'height'
        plate.rebuild()
        np.testing.assert_allclose(plate.pos.mean(axis=0), center, rtol=0, atol=1e-6,
                                   err_msg='Changing shape and orientation must preserve the scene placement')
        self.assertEqual(plate.n_particles, np.prod(plate.lattice_shape))


class CylinderGeometryTests(unittest.TestCase):
    def test_small_diameters_have_three_dimensional_cross_sections(self):
        for diameter in (0.001, 0.006, 0.0127, 0.013):
            with self.subTest(diameter=diameter):
                pos = create_cylinder_body(0.025, diameter, 'steel')[0]
                np.testing.assert_allclose(
                    np.ptp(pos[:, 1:], axis=0), (diameter, diameter), rtol=0, atol=1e-7,
                    err_msg=f'{diameter} m cylinder must span its diameter in both transverse axes')

    def test_requested_length_and_radius_do_not_depend_on_spacing(self):
        length, diameter = 0.055, 0.030
        for spacing in (0.007, 0.012):
            with self.subTest(spacing=spacing):
                pos = create_cylinder_body(length, diameter, 'steel', spacing=spacing)[0]
                self.assertAlmostEqual(
                    float(np.ptp(pos[:, 0])), length, delta=1e-7,
                    msg='Cylinder axial extent must equal its requested length')
                self.assertAlmostEqual(
                    float(np.linalg.norm(pos[:, 1:], axis=1).max()), diameter / 2,
                    delta=1e-7,
                    msg='Cylinder outer radius must not round down to a lattice multiple')
                np.testing.assert_allclose(
                    np.ptp(pos[:, 1:], axis=0), (diameter, diameter), rtol=0, atol=1e-7,
                    err_msg='Cylinder transverse extents must equal its requested diameter')

    def test_springs_match_brute_force_distance_neighbors(self):
        cases = (
            (0.055, 0.030, 0.012),
            (0.031, 0.017, 0.0073),
            (0.025, 0.013, 0.007),
            (0.007, 0.007, 0.00625),
        )
        for length, diameter, spacing in cases:
            with self.subTest(length=length, diameter=diameter, spacing=spacing):
                pos, _, _, springs, rest, _ = create_cylinder_body(
                    length, diameter, 'steel', spacing=spacing)
                expected = {}
                for start in range(len(pos)):
                    for end in range(start + 1, len(pos)):
                        distance = np.linalg.norm(pos[end] - pos[start])
                        if distance < spacing * 1.8:
                            expected[start, end] = distance

                actual_pairs = [tuple(pair) for pair in springs]
                self.assertEqual(
                    set(actual_pairs), set(expected),
                    'Cylinder springs must include exactly the pairs within the distance cutoff')
                np.testing.assert_allclose(
                    rest, [expected[pair] for pair in actual_pairs], rtol=1e-6, atol=1e-9,
                    err_msg='Cylinder spring lengths must match the brute-force reference')


class SmallGeometrySpringTests(unittest.TestCase):
    def test_small_valid_shapes_have_connected_valid_springs(self):
        cases = (
            ('minimum cylinder', create_cylinder_body, (0.001, 0.001, 'steel')),
            ('narrow cylinder', create_cylinder_body, (0.040, 0.001, 'steel')),
            ('minimum plate', create_plate_body, (0.001, 0.001, 0.001, 0, 'steel')),
            ('thin plate', create_plate_body, (0.025, 0.021, 0.001, 37, 'steel')),
        )
        for name, factory, args in cases:
            with self.subTest(shape=name):
                pos, _, _, springs, rest, stiffness = factory(*args)[:6]
                self.assertGreater(len(springs), 0, f'{name} must have connecting springs')
                self.assertEqual(springs.shape, (len(rest), 2))
                self.assertTrue(np.issubdtype(springs.dtype, np.integer))
                self.assertTrue(np.all((springs >= 0) & (springs < len(pos))),
                                f'{name} has out-of-range spring endpoints')
                self.assertTrue(np.all(np.isfinite(rest) & (rest > 0)),
                                f'{name} has a zero-length or nonfinite spring')
                self.assertTrue(np.all(np.isfinite(stiffness) & (stiffness > 0)),
                                f'{name} has an invalid spring stiffness')
                self.assertEqual(
                    len(np.unique(np.sort(springs, axis=1), axis=0)), len(springs),
                    f'{name} has duplicate springs')
                actual_lengths = np.linalg.norm(pos[springs[:, 1]] - pos[springs[:, 0]], axis=1)
                np.testing.assert_allclose(
                    rest, actual_lengths, rtol=1e-6, atol=1e-9,
                    err_msg=f'{name} spring rest lengths must match its geometry')

                neighbors = [set() for _ in range(len(pos))]
                for start, end in springs:
                    neighbors[start].add(end)
                    neighbors[end].add(start)
                visited = {0}
                pending = [0]
                while pending:
                    for neighbor in neighbors[pending.pop()] - visited:
                        visited.add(neighbor)
                        pending.append(neighbor)
                self.assertEqual(
                    len(visited), len(pos),
                    f'{name} has particles disconnected from its spring graph')


class CylinderSamplingTests(unittest.TestCase):
    def test_shaft_cross_sections_avoid_unnecessarily_close_samples(self):
        for diameter, spacing in ((0.030, 0.012), (0.020, 0.008), (0.050, 0.012)):
            with self.subTest(diameter=diameter, spacing=spacing):
                pos = create_cylinder_body(0.055, diameter, 'steel', spacing=spacing)[0]
                cross_section = pos[pos[:, 0] == pos[:, 0].min(), 1:]
                distances = np.linalg.norm(
                    cross_section[:, None] - cross_section[None, :], axis=2)
                np.fill_diagonal(distances, np.inf)
                self.assertGreaterEqual(
                    float(distances.min()), 0.7 * spacing,
                    'Shaft samples must stay near the requested spacing instead of crowding the interior')

    def test_tiny_diameter_changes_do_not_add_an_unnecessary_shaft_ring(self):
        spacing = 0.012
        for diameter in (0.024, 0.048):
            with self.subTest(diameter=diameter):
                counts = []
                for sampled_diameter in (diameter, diameter + 0.000001):
                    pos = create_cylinder_body(
                        0.055, sampled_diameter, 'steel', spacing=spacing)[0]
                    counts.append(np.count_nonzero(pos[:, 0] == pos[:, 0].min()))
                self.assertEqual(
                    counts[0], counts[1],
                    'A 0.001 mm diameter increase across a spacing multiple must not add a shaft ring')

    def test_taper_cross_sections_do_not_crowd_samples_near_the_tip(self):
        for length, diameter, spacing in ((0.400, 0.030, 0.012), (0.600, 0.013, 0.007)):
            with self.subTest(length=length, diameter=diameter, spacing=spacing):
                pos = create_cylinder_body(length, diameter, 'steel', spacing=spacing)[0]
                for x in np.unique(pos[:, 0]):
                    cross_section = pos[pos[:, 0] == x, 1:]
                    if len(cross_section) < 2:
                        continue
                    distances = np.linalg.norm(
                        cross_section[:, None] - cross_section[None, :], axis=2)
                    np.fill_diagonal(distances, np.inf)
                    self.assertGreaterEqual(
                        float(distances.min()), 0.35 * spacing,
                        f'Taper samples at x={float(x):.6f} m must not crowd into an undersized ring')


if __name__ == '__main__':
    unittest.main()
