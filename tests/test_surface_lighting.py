import unittest

import numpy as np

from surface_lighting import SurfaceShading, surface_groups


class SurfaceShadingTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.faces = [[0, 1, 2], [1, 0, 3]]

    def test_sharp_face_groups_duplicate_only_the_shared_corners(self):
        surface = SurfaceShading(self.faces, [-2, 7])
        positions = self.positions[surface.vertex_indices]
        normals = surface.normals(positions)

        np.testing.assert_array_equal(positions[surface.faces], self.positions[self.faces],
                                      err_msg='Shading must replace indices without changing face geometry')
        np.testing.assert_array_equal(np.bincount(surface.vertex_indices), [2, 2, 1, 1],
                                      err_msg='Only the two shared corners must be duplicated')
        np.testing.assert_array_equal(normals[surface.faces[0]], [[0, 0, 1]] * 3,
                                      err_msg='The first face should keep its own normal across the crease')
        np.testing.assert_array_equal(normals[surface.faces[1]], [[0, 1, 0]] * 3,
                                      err_msg='The second face should keep its own normal across the crease')

    def test_smooth_neighbors_use_area_weighted_shared_normals(self):
        surface = SurfaceShading(self.faces, [0, 0])
        normals = surface.normals(self.positions[surface.vertex_indices])

        np.testing.assert_array_equal(surface.vertex_indices, [0, 1, 2, 3],
                                      err_msg='One face group needs one render vertex per source vertex')
        expected = np.tile(np.array([0, 1, 2]) / np.sqrt(5), (2, 1))
        np.testing.assert_allclose(normals[:2], expected, atol=1e-7,
                                   err_msg='The twice-as-large face must contribute twice the weight')

    def test_folded_faces_keep_a_normal_when_their_contributions_cancel(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 1, 0]])
        surface = SurfaceShading(self.faces, [0, 0])

        normals = surface.normals(positions[surface.vertex_indices])

        np.testing.assert_array_equal(normals, [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, -1]],
                                      err_msg='A shared corner must fall back to its first nondegenerate face')

    def test_folded_faces_use_the_same_fallback_on_both_sides_of_cancellation(self):
        surface = SurfaceShading(self.faces, [0, 0])
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 1, 0]], dtype=np.float32)
        for toward in (np.float32(0), np.float32(2)):
            height = np.nextafter(np.float32(1), toward)
            with self.subTest(height=height):
                positions[3, 1] = height
                normals = surface.normals(positions[surface.vertex_indices])
                np.testing.assert_array_equal(normals[:2], [[0, 0, 1], [0, 0, 1]],
                                              err_msg='Float32 rounding must not flip the shared normals')

    def test_tiny_well_defined_normals_keep_their_area_weights(self):
        positions = np.concatenate((self.positions * 1e-8, [[2, 0, 0], [3, 0, 0], [2, 1, 0]]))
        surface = SurfaceShading([*self.faces, [4, 5, 6]], [0, 0, 0])

        normals = surface.normals(positions[surface.vertex_indices])

        expected = np.tile(np.array([0, 1, 2]) / np.sqrt(5), (2, 1))
        np.testing.assert_allclose(normals[:2], expected, atol=1e-7,
                                   err_msg='A coarse disconnected face must not change the small local normal')

    def test_degenerate_faces_produce_finite_contiguous_float32_normals(self):
        surface = SurfaceShading([[0, 1, 2]], [0])
        normals = surface.normals(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))

        np.testing.assert_array_equal(normals, np.zeros((3, 3)),
                                      err_msg='A zero-area face must supply no normal contribution')
        self.assertEqual(normals.dtype, np.float32)
        self.assertTrue(normals.flags.c_contiguous, 'The renderer requires a contiguous normal buffer')


class SurfaceGroupTests(unittest.TestCase):
    def test_plate_faces_use_one_group_for_each_lattice_boundary(self):
        faces = [[0, 1, 2], [4, 5, 6], [0, 1, 4], [2, 3, 6], [0, 2, 4], [1, 3, 5], [1, 3, 2]]

        groups = surface_groups(faces, lattice_shape=(2, 2, 2))

        np.testing.assert_array_equal(groups, [0, 1, 2, 3, 4, 5, 0],
                                      err_msg='Every boundary is separate and its alternate split has the same group')

    def test_round_cap_is_separate_from_coarse_side_faces(self):
        faces = [[0, 1, 2], [2, 1, 4], [3, 2, 4]]
        axial_positions = np.array([2, 2, 2, 6, 6])

        groups = surface_groups(faces, axial_positions=axial_positions)

        np.testing.assert_array_equal(groups, [1, 0, 0],
                                      err_msg='The cap must split from both side faces at its rim')
