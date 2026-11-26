import unittest

import numpy as np

from bodies import DETAIL_LEVELS, ArmorPlate, Penetrator


class DetailGeometryTests(unittest.TestCase):
    def test_default_body_nodes_increase_at_each_detail(self):
        for body_type in (ArmorPlate, Penetrator):
            with self.subTest(body=body_type.__name__):
                bodies = [body_type(detail=detail) for detail in DETAIL_LEVELS]
                spacings = [body.particle_spacing for body in bodies]
                np.testing.assert_allclose(spacings, [0.014, 0.012, 0.011, 0.010],
                                           err_msg='Default-size bodies must use the expected preset spacings')
                counts = [body.n_particles for body in bodies]
                self.assertTrue(all(a < b for a, b in zip(counts, counts[1:])),
                                f'Default-size body node counts must rise at each preset, got {counts}')

    def test_ultra_retains_a_finer_medium_mesh(self):
        cases = ((ArmorPlate(), 'thickness', 0.010), (Penetrator(), 'diameter', 0.015))
        for body, dimension, value in cases:
            with self.subTest(body=body.body_type):
                setattr(body, dimension, value)
                body.rebuild()
                self.assertLess(body.particle_spacing, 0.010, 'The reference mesh must be finer than 10 mm')
                medium_spacing = body.particle_spacing
                medium_count = body.n_particles
                medium_bonds = body.springs.copy()

                body.detail = 'Ultra'
                body.rebuild()

                self.assertEqual(body.particle_spacing, medium_spacing, 'Ultra must retain the finer spacing')
                self.assertEqual(body.n_particles, medium_count, 'Ultra must retain the finer node count')
                np.testing.assert_array_equal(body.springs, medium_bonds,
                                              err_msg='Ultra must retain the finer bond topology')
