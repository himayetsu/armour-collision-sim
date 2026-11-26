import unittest

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QApplication

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

        self.window.render_style_combo.setCurrentText('Surface Points')
        points = len(self.armor.scatter.pos)
        self.assertGreater(points, nodes, 'Surface Points must include generated display samples')
        self.assertEqual(row.text(1), f'{points:,} points')
        self.assertIn(f'{nodes:,} simulation nodes', row.toolTip(1))

        self.window.render_style_combo.setCurrentText('Wireframe')
        self.assertEqual(row.text(1), f'{nodes:,} points', 'Wireframe must count the original particles')
        self.assertIn(f'{bonds:,} simulation bonds', row.toolTip(1))

    def test_hiding_a_body_clears_its_display_counts_and_showing_restores_them(self):
        row = self.scene_row(self.armor)
        initial = row.text(1)
        self.assertTrue(initial.endswith(' faces'))

        self.window.armor_vis_cb.setChecked(False)
        self.assertEqual(row.text(1), 'Hidden')
        self.assertIn('0 display triangles\n0 display points', row.toolTip(1))

        self.window.armor_vis_cb.setChecked(True)
        self.assertEqual(row.text(1), initial, 'Showing the body must restore its display count')

    def test_adding_and_deleting_a_body_rebuilds_rows_for_the_active_style(self):
        self.window.render_style_combo.setCurrentText('Surface Points')
        initial = self.scene_row(self.armor).text(1)

        self.window.add_armor()
        added = self.window.selected
        self.assertIsNot(added, self.armor)
        self.assertEqual(self.scene_row(added).text(1), f'{len(added.scatter.pos):,} points')
        self.assertEqual(self.scene_row(self.armor).text(1), initial)

        self.window.delete_selected()
        self.assertFalse(self.window.tree.findItems(added.name, Qt.MatchExactly | Qt.MatchRecursive, 0),
                         'Deleting the body must remove its Scene row')
        self.window.render_style_combo.setCurrentText('Wireframe')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{self.armor.n_particles:,} points',
                         'The surviving row must still refresh after the tree is rebuilt')

    def test_solid_reports_faces_and_loose_points_together(self):
        corner = 0
        self.armor.springs = self.armor.springs[(self.armor.springs != corner).all(axis=1)]
        self.window.update_visuals(self.armor)

        faces = len(self.armor.surface.opts['meshdata'].faces())
        self.assertGreater(faces, 0, 'The plate must retain Solid geometry after the corner detaches')
        self.assertEqual(len(self.armor.scatter.pos), 1, 'The detached corner must be a loose point')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{faces:,} faces\n1 point')

    def test_surface_points_labels_loose_particles_as_points_when_no_faces_remain(self):
        self.window.render_style_combo.setCurrentText('Surface Points')
        self.armor.springs = np.empty((0, 2), dtype=np.int32)
        self.window.update_visuals(self.armor)

        row = self.scene_row(self.armor)
        self.assertEqual(row.text(1), f'{self.armor.n_particles:,} points')
        self.assertIn('0 display triangles', row.toolTip(1))

    def test_rebuilding_a_moved_body_preserves_its_surface_point_count(self):
        self.window.detail_combo.setCurrentText('Ultra')
        self.window.render_style_combo.setCurrentText('Surface Points')
        points = len(self.armor.scatter.pos)

        offset = np.array([1000, -500, 300], dtype=np.float32)
        self.armor.rest_pos += offset
        self.armor.pos += offset
        self.window.detail_combo.setCurrentText('Medium')
        self.window.detail_combo.setCurrentText('Ultra')

        self.assertEqual(len(self.armor.scatter.pos), points,
                         'A rebuilt plate must use the same sampling geometry regardless of its placement')
        self.assertEqual(self.scene_row(self.armor).text(1), f'{points:,} points')
