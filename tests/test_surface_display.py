import unittest

import numpy as np

from surface_display import SurfacePatchMesh


class SurfacePatchMeshTests(unittest.TestCase):
    def test_saddle_midpoint_uses_all_four_corners(self):
        positions = np.array([[0, 0, 0], [1, 0, 2], [1, 1, 0], [0, 1, 2]], dtype=np.float32)
        for subdivisions in (2, 4, 6):
            with self.subTest(subdivisions=subdivisions):
                points = SurfacePatchMesh([[0, 1, 2, 3]], [], subdivisions).sample(positions)
                np.testing.assert_array_equal(points[:4], positions, err_msg='The original corners must be exact')
                center = points[(points[:, :2] == 0.5).all(axis=1)]
                np.testing.assert_array_equal(center, [[0.5, 0.5, 1.0]],
                                              err_msg='The saddle center must include the two raised corners')

    def test_refined_quad_shares_the_original_edge_with_a_linear_triangle(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, -1, 0]])
        mesh = SurfacePatchMesh([[0, 1, 2, 3]], [[1, 0, 4]], 6)
        points = mesh.sample(positions)
        edges = np.concatenate((mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]))

        np.testing.assert_array_equal(points[mesh.faces[-1]], positions[[1, 0, 4]],
                                      err_msg='Standalone triangles must retain their winding and coordinates')
        np.testing.assert_array_equal(np.flatnonzero(points[:, 1] == 0), [0, 1],
                                      err_msg='Refinement must leave the shared boundary edge whole')
        self.assertEqual(np.count_nonzero((edges == [0, 1]).all(axis=1)), 1)
        self.assertEqual(np.count_nonzero((edges == [1, 0]).all(axis=1)), 1)

    def test_refined_faces_cover_the_quad_with_the_original_winding(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        for subdivisions in (1, 2, 3, 6):
            with self.subTest(subdivisions=subdivisions):
                mesh = SurfacePatchMesh([[0, 1, 2, 3]], [], subdivisions)
                faces = mesh.sample(positions)[mesh.faces]
                normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])

                self.assertTrue((normals[:, 2] > 0).all(), 'Every face must keep positive winding and area')
                self.assertAlmostEqual(normals[:, 2].sum() * 0.5, 1.0, places=6,
                                       msg='Refined faces must cover the area of the original quad')

    def test_bilinear_weights_apply_to_other_vertex_attributes(self):
        values = np.array([[100, 100], [0, 0], [1, 0], [1, 2], [0, 1]], dtype=np.float32)
        mesh = SurfacePatchMesh([[1, 2, 3, 4]], [], 3)
        samples = mesh.sample(values)

        np.testing.assert_array_equal(samples[:4], values[1:], err_msg='Unused original points must be skipped')
        expected = np.array([[1 / 3, 4 / 9], [2 / 3, 5 / 9], [1 / 3, 8 / 9], [2 / 3, 10 / 9]])
        np.testing.assert_allclose(samples[4:], expected, atol=1e-7,
                                   err_msg='Interior vertex attributes must share the bilinear weights')

    def test_empty_mesh_preserves_attribute_shape(self):
        mesh = SurfacePatchMesh([], [], 3)

        self.assertEqual(mesh.faces.shape, (0, 3))
        self.assertEqual(mesh.sample(np.zeros((4, 7))).shape, (0, 7))
