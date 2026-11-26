import unittest
from unittest.mock import patch

import numpy as np

from bodies import ArmorPlate, Penetrator
from surface_cuts import build_cut_faces
from surface_mesh import BodySurfaceTopology, connected_components


def _split_springs(body, sides):
    return body.springs[sides[body.springs[:, 0]] == sides[body.springs[:, 1]]]


def _shell_edges(faces):
    return np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))


def _concave_prism_fixture():
    ring = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]])
    layers = np.arange(24).reshape(4, 6)
    positions = np.column_stack((np.tile(ring, (4, 1)), np.repeat([-1, 0, 1, 2], 6)))
    labels = np.repeat([0, 1], 12)

    def walls(lower, upper):
        b, c = np.roll(lower, -1), np.roll(upper, -1)
        return np.concatenate((np.column_stack((lower, b, c)),
                               np.column_stack((lower, c, upper))))

    back = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 5], [3, 4, 5]])
    shell = np.concatenate((walls(*layers[:2]), walls(*layers[2:]),
                            layers[0, back[:, ::-1]], layers[3, back]))
    candidates = np.concatenate((shell, walls(*layers[1:3])))
    return positions, labels, shell, candidates


class SurfaceCutTests(unittest.TestCase):
    def test_clean_split_closes_each_component_with_outward_winding(self):
        plate, round_part = ArmorPlate(), Penetrator()
        width = plate.lattice_shape[-1]
        axial = round_part.surface_rest_pos[:, 0]
        sections = np.unique(axial)
        for body, sides in ((plate, np.arange(plate.n_particles) % width < width // 2),
                            (round_part, axial < sections[len(sections) // 2])):
            with self.subTest(body=body.body_type):
                surface = BodySurfaceTopology(body)
                self.assertEqual(surface.cut_surfaces()[0].shape, (0, 3))
                surface.update(_split_springs(body, sides))

                caps, cut_groups = surface.cut_surfaces()
                labels = sides[caps]
                self.assertGreater(len(caps), 0, 'A genuine planar split needs cap triangles')
                self.assertTrue((labels == labels[:, :1]).all(), 'Caps must not mix fragment nodes')
                self.assertTrue(np.isin(caps, surface.vertices).all(), 'Cap indices must reuse the shell vertices')
                groups = np.unique(cut_groups)
                self.assertEqual(len(groups), 2, 'The two cut surfaces need separate lighting groups')
                self.assertTrue((groups > 0).all(), 'Group zero is reserved for the exterior')
                for group in groups:
                    self.assertEqual(len(np.unique(labels[cut_groups == group])), 1,
                                     'Each lighting group must belong to one fragment')

                closed = np.concatenate((surface.faces, caps))
                edges = _shell_edges(closed)
                _, indices, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                               return_inverse=True, return_counts=True)
                np.testing.assert_array_equal(counts, 2, err_msg='Each closed shell edge needs two faces')
                directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
                np.testing.assert_array_equal(np.bincount(indices.ravel(), weights=directions), 0,
                                              err_msg='Adjacent faces must wind oppositely along their seam')
                for side in (False, True):
                    points = body.surface_rest_pos[closed[sides[closed[:, 0]] == side]].astype(np.float64)
                    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                    self.assertGreater(np.sum(points[:, 0] * normals), 0,
                                       'Each closed fragment must wind outward')

    def test_large_rotated_ultra_plate_keeps_every_authored_rim_edge(self):
        body = ArmorPlate(detail='Ultra')
        body.width, body.height, body.thickness = 0.12, 1.28, 0.036
        body.angle, body.pitch, body.roll = 37, 23, -31
        body.rebuild()
        thickness, height, width = body.lattice_shape
        rim_size = 2 * (thickness + height - 2)
        self.assertGreater(rim_size, 256, 'The fixture must need the large-rim path')
        sides = np.arange(body.n_particles) % width < width // 2
        surface = BodySurfaceTopology(body)
        surface.update(_split_springs(body, sides), body.pos)

        caps, groups = surface.cut_surfaces()
        np.testing.assert_array_equal(np.unique(groups, return_counts=True)[1], [rim_size - 2] * 2,
                                      err_msg='Both large rims must triangulate with their original vertices')
        boundaries = []
        for faces in (surface.faces, caps):
            edges = _shell_edges(faces)
            _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                           return_inverse=True, return_counts=True)
            boundaries.append(edges[counts[inverse.ravel()] == 1])
        shell_rim, cap_rim = boundaries
        self.assertSetEqual(set(map(tuple, cap_rim)), set(map(tuple, shell_rim[:, ::-1])),
                            'Each original rim segment must close with the opposite direction')
        positions = body.surface_rest_pos[caps].astype(np.float64)
        areas = np.linalg.norm(np.cross(positions[:, 1] - positions[:, 0],
                                        positions[:, 2] - positions[:, 0]), axis=1)
        self.assertTrue((areas > 0).all(), 'Collinear samples must not create degenerate faces')

    def test_single_layer_split_caps_only_the_thick_fragment(self):
        body = ArmorPlate()
        thickness, height, width = body.lattice_shape
        thin = np.arange(body.n_particles) % width == 0
        surface = BodySurfaceTopology(body)
        thin_faces = surface.faces[thin[surface.faces].all(axis=1)]
        surface.update(_split_springs(body, thin), body.pos)

        caps, groups = surface.cut_surfaces()
        self.assertEqual(len(caps), 2 * (thickness + height - 2) - 2,
                         'The layer must still provide the reciprocal rim for the thick piece')
        self.assertFalse(thin[caps].any(), 'No cap may duplicate the separated layer')
        self.assertEqual(len(np.unique(groups)), 1, 'Only the thick piece gets a cut lighting group')
        np.testing.assert_array_equal(surface.faces[thin[surface.faces[:, 0]]], thin_faces,
                                      err_msg='The one-layer fragment must retain its single original exterior')

        closed = np.concatenate((surface.faces[~thin[surface.faces[:, 0]]], caps))
        edges = _shell_edges(closed)
        _, indices, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        np.testing.assert_array_equal(counts, 2, err_msg='The thick fragment must be watertight')
        directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
        np.testing.assert_array_equal(np.bincount(indices.ravel(), weights=directions), 0,
                                      err_msg='The thick fragment cap must close the exterior winding')

    def test_concave_rims_fill_only_the_polygon(self):
        positions, labels, shell, candidates = _concave_prism_fixture()
        faces, groups = build_cut_faces(shell, candidates, positions, labels)
        np.testing.assert_array_equal(np.unique(groups, return_counts=True)[1], [4, 4])
        for group in np.unique(groups):
            triangles = positions[faces[groups == group]]
            normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            self.assertEqual(np.abs(normal[:, 2]).sum() / 2, 5,
                             'The cap must not fill the missing corner of the L-shaped section')
            self.assertFalse((triangles.mean(axis=1)[:, :2] > 1).all(axis=1).any(),
                             'A cap triangle must not cross into the concave recess')

    def test_large_rim_path_leaves_concave_openings_unfilled(self):
        positions, labels, shell, candidates = _concave_prism_fixture()
        with patch('surface_cuts._EAR_CLIP_LIMIT', 5):
            faces, groups = build_cut_faces(shell, candidates, positions, labels)

        self.assertEqual(faces.shape, (0, 3), 'The large convex path must not cover a recess')
        self.assertEqual(len(groups), 0)

    def test_second_opening_leaves_the_ambiguous_rims_open(self):
        positions, labels, shell, candidates = _concave_prism_fixture()
        shell = np.delete(shell, 24, axis=0)

        faces, groups = build_cut_faces(shell, candidates, positions, labels)

        self.assertEqual(faces.shape, (0, 3), 'Multiple rims on one piece make the split ambiguous')
        self.assertEqual(len(groups), 0)

    def test_close_planar_rims_without_an_authored_join_stay_open(self):
        positions, labels, shell, _ = _concave_prism_fixture()

        faces, groups = build_cut_faces(shell, shell, positions, labels)

        self.assertEqual(faces.shape, (0, 3), 'Nearby rims alone must not establish a cut')
        self.assertEqual(len(groups), 0)

    def test_isolated_corner_has_no_reciprocal_rim(self):
        body = ArmorPlate()
        surface = BodySurfaceTopology(body)
        attached = np.arange(body.n_particles) != 0
        surface.update(_split_springs(body, attached))

        self.assertLess(len(surface.faces), len(BodySurfaceTopology(body).faces))
        self.assertEqual(len(surface.cut_surfaces()[0]), 0, 'A detached node must not close an old corner')

    def test_nonplanar_cut_remains_open(self):
        body = ArmorPlate()
        body.bend = 75
        body.rebuild()
        surface = BodySurfaceTopology(body)
        depth = np.indices(body.lattice_shape)[0].ravel()
        surface.update(_split_springs(body, depth < body.lattice_shape[0] // 2))

        self.assertLess(len(surface.faces), len(BodySurfaceTopology(body).faces))
        self.assertEqual(len(surface.cut_surfaces()[0]), 0, 'A curved thickness split has no planar section')

    def test_same_component_gap_has_no_cut_faces(self):
        body = ArmorPlate()
        surface = BodySurfaceTopology(body)
        width = body.lattice_shape[-1]
        sides = np.arange(body.n_particles) % width < width // 2
        crosses = sides[body.springs[:, 0]] != sides[body.springs[:, 1]]
        interior = ~np.isin(body.springs, surface.vertices).any(axis=1)
        keep = ~crosses
        keep[np.flatnonzero(crosses & interior)[0]] = True
        positions = body.pos.copy()
        positions[sides] += (0, 0, -0.4)
        original_faces = len(surface.faces)

        surface.update(body.springs[keep], positions)

        self.assertLess(len(surface.faces), original_faces, 'The two sides must open a visible gap')
        self.assertEqual(len(surface.cut_surfaces()[0]), 0, 'A gap in a still-tethered component must remain open')

    def test_cut_indices_and_component_labels_are_reused_during_motion(self):
        body = ArmorPlate()
        surface = BodySurfaceTopology(body)
        width = body.lattice_shape[-1]
        sides = np.arange(body.n_particles) % width < width // 2
        springs = _split_springs(body, sides)
        with patch('surface_mesh.build_cut_faces', wraps=build_cut_faces) as build, patch(
                'surface_mesh.connected_components', wraps=connected_components) as components:
            surface.update(springs, body.pos)
            faces, groups = surface.cut_surfaces()
            for distance in (0.01, 0.05, 0.1):
                positions = body.pos.copy()
                positions[sides] += (0, 0, -distance)
                surface.update(springs, positions)
                next_faces, next_groups = surface.cut_surfaces()
                self.assertIs(next_faces, faces, 'Motion must preserve the cap face buffer')
                self.assertIs(next_groups, groups, 'Motion must preserve the cap group buffer')

        self.assertGreater(len(faces), 0)
        build.assert_called_once()
        components.assert_called_once()
        surface.update(body.springs, body.rest_pos)
        self.assertEqual(len(surface.cut_surfaces()[0]), 0, 'Restoring bonds must clear old cut faces')

    def test_internal_bond_loss_retriangulates_only_when_component_membership_changes(self):
        body = ArmorPlate()
        surface = BodySurfaceTopology(body)
        width = body.lattice_shape[-1]
        sides = np.arange(body.n_particles) % width < width // 2
        springs = _split_springs(body, sides)
        interior = ~np.isin(springs, surface.vertices).any(axis=1)
        bond = np.flatnonzero(interior)[0]
        interior_vertex = springs[bond, 0]

        with patch('surface_mesh.build_cut_faces', wraps=build_cut_faces) as build:
            surface.update(springs, body.pos)
            shell, caps = surface.faces, surface.cut_surfaces()[0]
            self.assertGreater(len(caps), 0, 'The authored split needs cap faces to exercise reuse')

            springs = np.delete(springs, bond, axis=0)
            surface.update(springs, body.pos)
            self.assertIs(surface.faces, shell, 'An internal bond loss must leave the shell intact')
            self.assertIs(surface.cut_surfaces()[0], caps, 'The unchanged cut should reuse its cap indices')
            build.assert_called_once()

            springs = springs[(springs != interior_vertex).all(axis=1)]
            surface.update(springs, body.pos)
            self.assertIs(surface.faces, shell, 'Isolating an interior vertex must not change the shell')
            surface.cut_surfaces()
            self.assertEqual(build.call_count, 2, 'Changed component membership must revalidate the cut')
