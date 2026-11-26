import unittest

import numpy as np

from surface_mesh import SurfacePointSampler


class SurfacePointSamplerTests(unittest.TestCase):
    def test_samples_include_corners_and_share_points_on_a_common_edge(self):
        rest = np.array([[0, 0, 0], [0.004, 0, 0], [0.004, 0.004, 0], [0, 0.004, 0]])
        faces = [[0, 1, 2], [0, 2, 3]]
        samples = SurfacePointSampler(rest, faces).sample(rest)

        np.testing.assert_array_equal(samples[:4], rest, err_msg='Each source corner needs one sample')
        diagonal = samples[np.isclose(samples[:, 0], samples[:, 1])]
        np.testing.assert_allclose(np.sort(diagonal[:, 0]), np.linspace(0, 0.004, 4), atol=1e-9,
                                   err_msg='The shared edge needs a single set of 2 mm-target samples')
        self.assertEqual(len(np.unique(samples, axis=0)), len(samples), 'Shared points must not be duplicated')

    def test_cached_samples_follow_deformation_and_interpolate_attributes(self):
        rest = np.array([[0, 0, 0], [0.008, 0, 0], [0, 0.006, 0]], dtype=np.float32)
        sampler = SurfacePointSampler(rest, [[0, 1, 2]])
        samples = sampler.sample(rest)
        transform = np.array([[1, 0, 0], [0, 0.5, 0], [2, 0, 1]], dtype=np.float32)
        translation = np.array([0.003, -0.002, 0.013], dtype=np.float32)
        color = np.array([[0.1], [0.9], [0.4]], dtype=np.float32)
        moved = sampler.sample(np.column_stack((rest @ transform.T + translation, color)))

        np.testing.assert_allclose(moved[:, :3], samples @ transform.T + translation, atol=1e-8,
                                   err_msg='Samples must move with their deformed triangle')
        expected_color = 0.1 + 0.8 * samples[:, 0] / 0.008 + 0.3 * samples[:, 1] / 0.006
        np.testing.assert_allclose(moved[:, 3], expected_color, atol=1e-7,
                                   err_msg='Attributes must use the same weights as position')

    def test_large_triangle_stays_within_the_display_budget(self):
        rest = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
        sampler = SurfacePointSampler(rest, [[0, 1, 2]], max_points=13)
        samples = sampler.sample(rest)

        self.assertEqual(len(samples), 13, 'Oversized surfaces must honor the point budget')
        self.assertTrue((samples >= 0).all() and (samples.sum(axis=1) <= 10).all(),
                        'Every selected sample must stay inside the source triangle')
        self.assertGreater(np.ptp(samples[:, 0]), 1, 'Selected samples must span the large surface')
        self.assertGreater(np.ptp(samples[:, 1]), 1, 'Selected samples must span both surface dimensions')

    def test_empty_surface_preserves_attribute_shape(self):
        rest = np.zeros((3, 3), dtype=np.float32)
        samples = SurfacePointSampler(rest, []).sample(np.zeros((3, 7), dtype=np.float32))

        self.assertEqual(samples.shape, (0, 7), 'An empty surface must have zero points and preserve columns')
