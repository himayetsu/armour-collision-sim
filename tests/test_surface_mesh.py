import unittest
from types import SimpleNamespace

import numpy as np

from bodies import ArmorPlate, Penetrator
from surface_mesh import BodySurfaceTopology


def _assert_closed_outward_surface(test, positions, faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    undirected = np.sort(edges, axis=1)
    unique_edges, indices, counts = np.unique(undirected, axis=0, return_inverse=True,
                                             return_counts=True)
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    np.testing.assert_array_equal(counts, np.full(len(unique_edges), 2),
                                  err_msg='Each shell edge must join exactly two faces')
    np.testing.assert_array_equal(np.bincount(indices.ravel(), weights=direction),
                                  np.zeros(len(unique_edges)),
                                  err_msg='Shared shell edges must have opposite directions')
    triangles = positions[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    test.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 0), 'Shell faces must have area')
    volume = np.sum(triangles[:, 0] * normals) / 6
    test.assertGreater(volume, 0, 'The shell must face outward')


class ExteriorSurfaceTests(unittest.TestCase):
    def test_plate_only_renders_boundary_cells(self):
        plate = ArmorPlate()
        surface = BodySurfaceTopology(plate)
        shape = np.asarray(plate.lattice_shape)
        nz, ny, nx = shape - 1
        self.assertEqual(len(surface.faces), 4 * (nx * ny + ny * nz + nx * nz))
        indices = np.array(np.unravel_index(surface.faces, shape)).transpose(1, 2, 0)
        for axis in range(len(shape)):
            on_boundary = (indices[:, :, axis] == 0).all(axis=1)
            on_boundary |= (indices[:, :, axis] == shape[axis] - 1).all(axis=1)
            if axis == 0:
                boundary_faces = on_boundary
            else:
                boundary_faces |= on_boundary
        self.assertTrue(boundary_faces.all(), 'Every triangle must lie on an exterior plane')
        self.assertEqual(len(surface.particles), 0, 'An intact shell must hide interior points')
        _assert_closed_outward_surface(self, plate.pos, surface.faces)

    def test_cylinder_stitches_unequal_outer_rings_and_covers_ends(self):
        cylinder = Penetrator()
        cylinder.diameter = 0.080
        cylinder.length = 0.165
        cylinder.rebuild()
        surface = BodySurfaceTopology(cylinder)
        _assert_closed_outward_surface(self, cylinder.pos, surface.faces)
        front = np.argmax(cylinder.pos[:, 0])
        self.assertIn(front, surface.vertices, 'The taper endpoint must remain in the mesh')
        x = cylinder.pos[:, 0]
        for section in np.unique(x)[1:-1]:
            layer = np.flatnonzero(x == section)
            offsets = cylinder.pos[layer, 1:] - cylinder.pos[layer[0], 1:]
            radius = np.linalg.norm(offsets, axis=1)
            interior = layer[~np.isclose(radius, radius.max())]
            self.assertFalse(np.isin(interior, surface.vertices).any(),
                             'Intermediate cylinder rings must exclude their interior points')

    def test_removed_bonds_hide_their_faces_and_keep_detached_particles(self):
        plate = ArmorPlate()
        surface = BodySurfaceTopology(plate)
        faces = surface.faces.copy()
        corner = 0
        breaks = (plate.springs == corner).any(axis=1)
        surface.update(plate.springs[~breaks])
        expected = faces[~(faces == corner).any(axis=1)]
        np.testing.assert_array_equal(surface.faces, expected,
                                      err_msg='A facet must disappear when its corner detaches')
        np.testing.assert_array_equal(surface.particles, [corner],
                                      err_msg='The detached corner must remain visible as a point')

    def test_facet_without_direct_bonds_cannot_join_detached_components(self):
        # The last corner reaches the face through two intermediate particles.
        body = SimpleNamespace(n_particles=5, body_type='penetrator',
                               surface_faces=np.array([[0, 1, 2]]),
                               springs=np.array([[0, 1], [1, 3], [3, 4], [2, 4]]))
        surface = BodySurfaceTopology(body)
        self.assertEqual(len(surface.faces), 1, 'The indirectly supported facet starts intact')

        surface.update(body.springs[[0, 1, 3]])

        self.assertEqual(len(surface.faces), 0,
                         'The facet must disappear when its two bonded groups separate')
