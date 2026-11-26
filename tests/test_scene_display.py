import unittest

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication, QLabel

from mainwindow import MainWindow


class SceneDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['scene-display-tests', '-platform', 'offscreen'])

    def setUp(self):
        self.window = MainWindow()
        self.addCleanup(self.close_window)
        self.armor = next(body for body in self.window.physics.bodies if body.body_type == 'armor')

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.sendPostedEvents(self.window, QEvent.DeferredDelete)

    def scene_row(self, body):
        rows = self.window.tree.findItems(body.name, Qt.MatchExactly | Qt.MatchRecursive, 0)
        self.assertEqual(len(rows), 1, f'{body.name} must have exactly one Scene row')
        return rows[0]

    def test_render_style_updates_the_display_count_without_changing_simulation_counts(self):
        self.window.detail_combo.setCurrentText('Ultra')
        row = self.scene_row(self.armor)
        nodes = self.armor.n_particles
        bonds = len(self.armor.springs)
        faces = len(self.armor.surface.opts['meshdata'].faces())

        self.assertEqual(row.text(1), f'{faces:,} faces', 'Solid must count the refined triangles')
        self.assertIn(f'{nodes:,} simulation nodes', row.toolTip(1))
        self.assertIn(f'{bonds:,} simulation bonds', row.toolTip(1))

        self.window.render_style_combo.setCurrentText('Surface Nodes')
        exterior = np.unique(self.armor.surface.topology.faces)
        self.assertLess(len(exterior), nodes, 'The intact plate fixture must have interior nodes')
        self.assertEqual(row.text(1), f'{len(exterior):,} nodes')
        self.assertIn(f'{nodes:,} simulation nodes', row.toolTip(1))

        self.window.render_style_combo.setCurrentText('Wireframe')
        self.assertEqual(row.text(1), f'{nodes:,} nodes', 'Wireframe must count every solver node')
        np.testing.assert_array_equal(self.armor.scatter.pos, self.armor.pos,
                                      err_msg='Wireframe must draw every solver position')
        self.assertIn(f'{bonds:,} simulation bonds', row.toolTip(1))

    def test_hiding_a_body_clears_its_display_counts_and_showing_restores_them(self):
        row = self.scene_row(self.armor)
        initial = row.text(1)
        self.assertTrue(initial.endswith(' faces'))

        self.window.armor_vis_cb.setChecked(False)
        self.assertEqual(row.text(1), 'Hidden')
        self.assertIn('0 display triangles\n0 display nodes', row.toolTip(1))

        self.window.armor_vis_cb.setChecked(True)
        self.assertEqual(row.text(1), initial, 'Showing the body must restore its display count')

    def test_adding_and_deleting_a_body_rebuilds_rows_for_the_active_style(self):
        self.window.render_style_combo.setCurrentText('Surface Nodes')
        initial = self.scene_row(self.armor).text(1)

        self.window.add_armor()
        added = self.window.selected
        self.assertIsNot(added, self.armor)
        self.assertEqual(self.scene_row(added).text(1), f'{len(added.scatter.pos):,} nodes')
        self.assertEqual(self.scene_row(self.armor).text(1), initial)

        self.window.delete_selected()
        self.assertFalse(self.window.tree.findItems(added.name, Qt.MatchExactly | Qt.MatchRecursive, 0),
                         'Deleting the body must remove its Scene row')
        self.window.render_style_combo.setCurrentText('Wireframe')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{self.armor.n_particles:,} nodes',
                         'The surviving row must still refresh after the tree is rebuilt')

    def test_solid_reports_faces_and_loose_nodes_together(self):
        corner = 0
        self.armor.springs = self.armor.springs[(self.armor.springs != corner).all(axis=1)]
        self.window.update_visuals(self.armor)

        faces = len(self.armor.surface.opts['meshdata'].faces())
        self.assertGreater(faces, 0, 'The plate must retain Solid geometry after the corner detaches')
        np.testing.assert_array_equal(self.armor.scatter.pos, self.armor.pos[[corner]],
                                      err_msg='Solid must draw the detached corner at its solver position')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{faces:,} faces\n1 loose node')

    def test_surface_nodes_draws_all_loose_nodes_when_no_faces_remain(self):
        self.window.render_style_combo.setCurrentText('Surface Nodes')
        self.armor.springs = np.empty((0, 2), dtype=np.int32)
        self.window.update_visuals(self.armor)

        row = self.scene_row(self.armor)
        np.testing.assert_array_equal(self.armor.scatter.pos, self.armor.pos,
                                      err_msg='The loose-only fallback must draw every solver position once')
        self.assertEqual(row.text(1), f'{self.armor.n_particles:,} nodes')
        self.assertIn('0 display triangles', row.toolTip(1))

    def test_surface_nodes_follows_the_original_exterior_through_inert_motion(self):
        self.window.detail_combo.setCurrentText('Ultra')
        self.window.render_style_combo.setCurrentText('Surface Nodes')
        exterior = np.unique(self.armor.surface.topology.faces)
        initial = self.armor.scatter.pos.copy()
        np.testing.assert_array_equal(initial, self.armor.pos[exterior],
                                      err_msg='Surface Nodes must draw the original exterior positions')

        self.armor.pos[exterior[0]] += (0.005, -0.003, 0.002)
        self.window.update_visuals(self.armor)

        self.assertFalse(np.array_equal(initial, self.armor.scatter.pos), 'The display must follow node motion')
        np.testing.assert_array_equal(self.armor.scatter.pos, self.armor.pos[exterior],
                                      err_msg='Surface Nodes must draw the live solver positions after motion')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{len(exterior):,} nodes')

    def test_surface_nodes_adds_a_detached_interior_node(self):
        self.window.render_style_combo.setCurrentText('Surface Nodes')
        exterior = np.unique(self.armor.surface.topology.faces)
        interior = np.setdiff1d(np.arange(self.armor.n_particles), exterior)
        self.assertGreater(len(interior), 1, 'The plate needs an interior node that can stay hidden')
        detached = interior[0]
        self.armor.springs = self.armor.springs[(self.armor.springs != detached).all(axis=1)]
        self.armor.pos[detached] += (0.1, 0.02, 0.03)
        self.window.update_visuals(self.armor)

        displayed = np.union1d(exterior, [detached])
        np.testing.assert_array_equal(self.armor.scatter.pos, self.armor.pos[displayed],
                                      err_msg='Surface Nodes must add only the detached interior position')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{len(displayed):,} nodes')

    def test_dimension_rebuild_refreshes_selected_solver_info_in_place(self):
        self.window.select_body(self.armor)
        info = self.window.props_box.itemAt(0).widget()
        labels = info.findChildren(QLabel)
        self.assertIn('12 mm nominal node spacing', [label.text() for label in labels])

        self.window.change_dim(self.armor, 'thickness', 0.010)
        self.window._do_armor_rebuild()

        self.assertIs(self.window.props_box.itemAt(0).widget(), info,
                      'Rebuilding the selected body should preserve its editing controls')
        updated = [label.text() for label in labels]
        self.assertIn('7 mm nominal node spacing', updated)
        self.assertIn(f'{self.armor.n_particles:,} simulation nodes', updated)
        self.assertIn(f'{len(self.armor.springs):,} simulation bonds', updated)

    def test_equal_solver_spacing_keeps_the_same_solid_mesh_across_detail_levels(self):
        self.window.change_dim(self.armor, 'thickness', 0.010)
        self.window._do_armor_rebuild()
        self.window.select_body(self.armor)
        spacing = self.armor.particle_spacing
        nodes = self.armor.n_particles
        medium_mesh = self.armor.surface.opts['meshdata']
        self.assertLess(spacing, 0.010, 'The fixture must use finer spacing than the Ultra target')

        self.window.detail_combo.setCurrentText('Ultra')

        self.assertEqual(self.armor.particle_spacing, spacing)
        self.assertEqual(self.armor.n_particles, nodes)
        ultra_mesh = self.armor.surface.opts['meshdata']
        np.testing.assert_array_equal(ultra_mesh.faces(), medium_mesh.faces(),
                                      err_msg='Solid must use the same tessellation when solver spacing matches')
        np.testing.assert_allclose(ultra_mesh.vertexes(), medium_mesh.vertexes(), rtol=0, atol=1e-6,
                                   err_msg='Matching solver geometry must produce the same rendered surface')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{len(medium_mesh.faces()):,} faces')
        info = self.window.props_box.itemAt(0).widget()
        self.assertIn(f'{spacing * 1000:g} mm nominal node spacing', [label.text() for label in info.findChildren(QLabel)])
