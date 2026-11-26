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
        self.assertFalse((surface.faces == corner).any(),
                         'Rendered faces must not include the detached corner')
        self.assertTrue(set(map(tuple, expected)).issubset(map(tuple, surface.faces)),
                        'Triangles away from the detached corner must remain visible')
        np.testing.assert_array_equal(surface.particles, [corner],
                                      err_msg='The detached corner must remain visible as a point')

    def test_breaking_only_the_shared_diagonal_preserves_both_faces(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        perimeter = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        body = SimpleNamespace(n_particles=4, body_type='mesh', surface_faces=faces,
                               surface_quad_pairs=np.array([[0, 1]]),
                               springs=np.vstack((perimeter, [0, 2])))
        surface = BodySurfaceTopology(body)
        rendered_faces = surface.faces
        mesh_faces = surface.mesh_faces
        quads = surface.quads

        surface.update(perimeter)

        np.testing.assert_array_equal(surface.faces, faces,
                                      err_msg='A broken diagonal must leave both supported triangles filled')
        self.assertIs(surface.faces, rendered_faces, 'Unchanged faces should retain their index buffer')
        self.assertIs(surface.mesh_faces, mesh_faces, 'Unchanged faces should retain their local index buffer')
        self.assertIs(surface.quads, quads, 'A ring or plate patch must retain its full quad')
        np.testing.assert_array_equal(surface.quads, [[0, 1, 2, 3]])
        self.assertEqual(surface.triangles.shape, (0, 3), 'A whole quad should not also render as triangles')

    def test_alternate_diagonal_fills_three_surviving_corners(self):
        for body in (ArmorPlate(), Penetrator()):
            with self.subTest(body=body.body_type):
                surface = BodySurfaceTopology(body)
                corner, *survivors = surface.quads[0]
                keep = (body.springs != corner).all(axis=1)

                surface.update(body.springs[keep], body.rest_pos)

                self.assertIn(tuple(survivors), set(map(tuple, surface.triangles)),
                              'A patch should fill the surviving corners using the alternate diagonal')
                self.assertFalse((surface.faces == corner).any(),
                                 'Changing the diagonal must keep the detached corner out of every face')

    def test_unsupported_face_opens_as_a_remote_tether_stretches(self):
        rest_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 1.5, 0]])
        faces = np.array([[0, 1, 2]])
        tether = np.array([[0, 1], [1, 3], [3, 4], [4, 2]])
        body = SimpleNamespace(n_particles=5, body_type='mesh', surface_faces=faces, rest_pos=rest_pos,
                               springs=np.vstack((tether, [[0, 2], [1, 2]])))
        surface = BodySurfaceTopology(body)
        previous_faces = surface.faces
        positions = rest_pos.copy()
        positions[2, 1] = 1.3
        surface.update(tether, positions)
        self.assertIs(surface.faces, previous_faces,
                      'Indirect support should keep a nearby face filled after moderate deformation')

        positions[2, 1] = 9
        surface.update(tether, positions)

        self.assertEqual(len(surface.faces), 0,
                         'A remote tether must not draw a stretched face across the gap')
        self.assertEqual(len(surface.triangles), 0, 'The gap should also remove the individual triangle')
        self.assertIn(2, surface.particles, 'The far corner must remain visible as a particle')

    def test_intact_deformation_retains_surface_indices(self):
        for body in (ArmorPlate(), Penetrator()):
            with self.subTest(body=body.body_type):
                surface = BodySurfaceTopology(body)
                original = (surface.faces, surface.mesh_faces, surface.quads, surface.triangles)
                positions = body.rest_pos * (1.08, 0.95, 1.1) + (0.4, 0.2, -0.3)

                surface.update(body.springs, positions)

                for name, expected in zip(('faces', 'mesh_faces', 'quads', 'triangles'), original):
                    self.assertIs(getattr(surface, name), expected,
                                  f'Intact deformation should keep the {name} index buffer')

    def test_face_stays_filled_when_the_surviving_paths_are_indirect(self):
        faces = np.array([[0, 1, 2]])
        surrounding = np.array([[0, 3], [1, 3], [2, 3]])
        body = SimpleNamespace(n_particles=4, body_type='mesh', surface_faces=faces,
                               springs=np.vstack((surrounding, [[0, 1], [0, 2], [1, 2]])))
        surface = BodySurfaceTopology(body)

        surface.update(surrounding)

        np.testing.assert_array_equal(surface.faces, faces,
                                      err_msg='A surface triangle should follow its connected corners')

    def test_separated_plate_halves_keep_only_their_own_faces(self):
        plate = ArmorPlate()
        surface = BodySurfaceTopology(plate)
        nx = plate.lattice_shape[-1]
        left = np.arange(plate.n_particles) % nx < nx // 2
        face_side = left[surface.faces]
        expected = surface.faces[(face_side == face_side[:, :1]).all(axis=1)]
        spring_side = left[plate.springs]

        surface.update(plate.springs[spring_side[:, 0] == spring_side[:, 1]])

        np.testing.assert_array_equal(surface.faces, expected,
                                      err_msg='A split must remove faces across the cut and retain both exteriors')
        self.assertEqual(len(surface.particles), 0, 'Both plate halves should remain filled at their exterior nodes')

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
