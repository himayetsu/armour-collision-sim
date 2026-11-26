# -*- coding: utf-8 -*-
"""3D view (SimViewWidget) and 2D ortho canvas (OrthoCanvas)."""

import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtCore import Qt, QVariantAnimation
from PyQt5.QtGui import QColor, QMatrix4x4, QVector3D, QPainter, QPainterPath, QPen, QBrush
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL import GL

from gizmo import TransformGizmo
from materials import MATERIALS
from surface_display import SurfacePatchMesh
from surface_mesh import BodySurfaceTopology, SurfacePointSampler


# Velocity → colour ramp (same stops as the 3D scatter in mainwindow.py).
# Defined once at module level to avoid re-allocation on every paintEvent.
_VEL_MAX = 1800.0
_VEL_T   = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
_VEL_RGB = np.array([[0.10, 0.20, 1.00], [0.05, 0.80, 1.00],
                      [0.10, 1.00, 0.40], [1.00, 1.00, 0.10],
                      [1.00, 0.45, 0.05], [1.00, 0.05, 0.05]])

PROJ_AXES = {
    'front': (2, 1, 'Z', 'Y'),
    'right': (0, 1, 'X', 'Y'),
    'top':   (0, 2, 'X', 'Z'),
}

# Internal camera keys stay front/right; labels match armor looking-direction.
VIEW_LABELS = {
    'perspective': 'Perspective',
    'front': 'Side',
    'right': 'Front',
    'top': 'Top',
}


def _surface_lighting_shader():
    # Reuse PyQtGraph's vertex stage for both its fixed-function (0.13) and
    # buffer-backed (0.14) mesh renderers.
    vertex = gl.shaders.getShaderProgram('shaded').shaders[0]
    legacy = 'gl_Normal' in vertex.code
    normal = 'normal' if legacy else 'v_normal'
    color = 'gl_Color' if legacy else 'v_color'
    fragment = gl.shaders.FragmentShader(f"""
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec3 {normal};
        {'' if legacy else 'varying vec4 v_color;'}
        void main() {{
            vec3 light = normalize(vec3(-0.45, 0.65, 1.0));
            vec3 facing_normal = normalize({normal});
            facing_normal = gl_FrontFacing ? facing_normal : -facing_normal;
            float brightness = 0.90 + 0.17 * dot(facing_normal, light);
            gl_FragColor = vec4({color}.rgb * brightness, {color}.a);
        }}
    """)
    return gl.shaders.ShaderProgram('armorsim_surface', [vertex, fragment])


_SURFACE_SHADER = _surface_lighting_shader()
_SURFACE_POINT_SPACING = {'Low': 0.016, 'Medium': 0.008, 'High': 0.004, 'Ultra': 0.002}
_SURFACE_POINT_SIZE = {'Low': 5, 'Medium': 4, 'High': 3, 'Ultra': 2}
_SURFACE_SUBDIVISIONS = {'High': 4, 'Ultra': 6}
_MAX_REFINED_TRIANGLES = 100_000


class BodySurface(gl.GLMeshItem):
    def __init__(self, body):
        topology = BodySurfaceTopology(body)
        subdivisions = _SURFACE_SUBDIVISIONS.get(body.detail, 1)
        if len(topology.quads):
            triangle_budget = max(0, _MAX_REFINED_TRIANGLES - len(topology.triangles))
            subdivisions = min(subdivisions, int(np.sqrt(triangle_budget / (2 * len(topology.quads)))))
        super().__init__(smooth=False, shader=_SURFACE_SHADER.name, glOptions='opaque',
                         drawEdges=body.detail == 'Low', edgeColor=(0.16, 0.20, 0.25, 1.0))
        self.updateGLOptions({GL.GL_CULL_FACE: False})
        self.topology = topology
        self.point_size = _SURFACE_POINT_SIZE[body.detail]
        self._point_spacing = _SURFACE_POINT_SPACING[body.detail]
        self._subdivisions = subdivisions
        self._refined = None
        self._refined_quads = None
        self._refined_triangles = None
        self._positions = None
        self._colors = None
        self._faces = None
        self._sample_faces = None
        self._sample_input = None
        self._sample_loose = None
        self._sample_buffers = None

    def paint(self):
        if not self.opts['drawEdges']:
            super().paint()
            return
        # Bias the fill away from the outlines so they stay visible at a grazing angle.
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        try:
            super().paint()
        finally:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

    def _refined_mesh(self):
        if self._subdivisions < 2:
            return None
        quads, triangles = self.topology.quads, self.topology.triangles
        if self._refined_quads is not quads or self._refined_triangles is not triangles:
            self._refined = SurfacePatchMesh(quads, triangles, self._subdivisions)
            self._refined_quads, self._refined_triangles = quads, triangles
        return self._refined

    def update_body(self, body, colors):
        """Upload the visible shell and return particle indices for loose geometry."""
        self.topology.update(body.springs, body.pos)
        refined = self._refined_mesh()
        faces = refined.faces if refined is not None else self.topology.mesh_faces
        self.setVisible(bool(len(faces)))
        if len(faces):
            positions = body.pos[self.topology.vertices]
            surface_colors = colors[self.topology.vertices]
            if (self._faces is not faces or not np.array_equal(positions, self._positions)
                    or not np.array_equal(surface_colors, self._colors)):
                if refined is None:
                    self.setMeshData(vertexes=positions, faces=faces, vertexColors=surface_colors)
                else:
                    values = refined.sample(np.column_stack((body.pos, colors)))
                    self.setMeshData(vertexes=np.ascontiguousarray(values[:, :3]), faces=faces,
                                     vertexColors=np.ascontiguousarray(values[:, 3:]))
                self._positions = positions
                self._colors = surface_colors
                self._faces = faces
        return self.topology.particles

    def surface_points(self, body, colors):
        self.topology.update(body.springs, body.pos)
        refined = self._refined_mesh()
        faces = refined.faces if refined is not None else self.topology.faces
        loose = self.topology.particles
        if not len(faces):
            return body.pos[loose], colors[loose]

        if self._sample_faces is not faces:
            rest = refined.sample(body.rest_pos) if refined is not None else body.rest_pos
            self._point_sampler = SurfacePointSampler(rest, faces, spacing=self._point_spacing)
            self._sample_faces = faces
            self._sample_input = None

        values = np.column_stack((body.pos, colors))
        if (not np.array_equal(self._sample_input, values)
                or not np.array_equal(self._sample_loose, loose)):
            source = refined.sample(values) if refined is not None else values
            samples = self._point_sampler.sample(source)
            if len(loose):
                samples = np.concatenate((samples, values[loose]))
            self._sample_input = values
            self._sample_loose = loose
            self._sample_buffers = (np.ascontiguousarray(samples[:, :3], dtype=np.float32),
                                    np.ascontiguousarray(samples[:, 3:], dtype=np.float32))
        return self._sample_buffers


class SimViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None, mode='perspective'):
        super().__init__(parent)
        self.ortho = mode != 'perspective'
        self.view_mode = mode
        self.gizmo = None
        self.setFocusPolicy(Qt.StrongFocus)

        self._drag_axis = None
        self._drag_start = None
        self._drag_body = None
        self._drag_pos0 = None
        self._drag_rest0 = None

        rng = np.random.default_rng(0)
        offsets = rng.normal(size=(64, 3))
        self._burst_offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
        self._burst_offsets *= rng.uniform(0.45, 1.0, size=(len(offsets), 1))
        self._burst = gl.GLScatterPlotItem(pxMode=True, glOptions='translucent')
        self._burst.setDepthValue(1)
        self._burst.hide()
        self.addItem(self._burst)
        self._burst_animation = QVariantAnimation(self)
        self._burst_animation.setDuration(650)
        self._burst_animation.setStartValue(0.0)
        self._burst_animation.setEndValue(1.0)
        self._burst_animation.valueChanged.connect(self._draw_burst)
        self._burst_animation.finished.connect(self._burst.hide)

        self._view_label = QLabel(VIEW_LABELS.get(mode, mode.capitalize()), self)
        self._view_label.setStyleSheet(
            "color: #6af; background: rgba(0,0,0,120); font-size: 11px; padding: 3px 6px;")
        self._view_label.move(6, 6)

        if mode == 'front':
            self.setCameraPosition(distance=0.5, elevation=0, azimuth=90)
        elif mode == 'right':
            self.setCameraPosition(distance=0.5, elevation=0, azimuth=0)
        elif mode == 'top':
            self.setCameraPosition(distance=0.5, elevation=90, azimuth=90)
        else:
            self.setCameraPosition(distance=0.5, elevation=15, azimuth=90)

    def preview_burst(self, center=None):
        self._burst_animation.stop()
        if center is None:
            target = self.opts['center']
            center = (target.x(), target.y(), target.z())
        self._burst_center = np.asarray(center, dtype=np.float32)
        self._burst_extent = self.opts['distance'] * 0.16
        self._draw_burst(0.0)
        self._burst.show()
        self._burst_animation.start()

    def _draw_burst(self, progress):
        spread = progress * (2 - progress) * self._burst_extent
        opacity = (1 - progress) ** 1.5
        self._burst.setData(pos=self._burst_center + self._burst_offsets * spread,
                            size=6 - 3 * progress, color=(1.0, 0.58, 0.25, opacity))

    def _default_viewport(self):
        dpr = self.devicePixelRatio()
        return (0, 0, int(self.width() * dpr), int(self.height() * dpr))

    def projectionMatrix(self, region=None, viewport=None):
        if viewport is None:
            viewport = self._default_viewport()
        if region is None:
            region = viewport

        if not self.ortho:
            return super().projectionMatrix(region, viewport)

        x0, y0, w, h = viewport
        w = max(w, 1)
        h = max(h, 1)
        dist = self.opts['distance']
        half_h = dist * 0.5
        half_w = half_h * w / h

        left   = half_w * ((region[0] - x0) * (2.0 / w) - 1)
        right  = half_w * ((region[0] + region[2] - x0) * (2.0 / w) - 1)
        bottom = half_h * ((region[1] - y0) * (2.0 / h) - 1)
        top    = half_h * ((region[1] + region[3] - y0) * (2.0 / h) - 1)

        tr = QMatrix4x4()
        tr.ortho(left, right, bottom, top, dist * 0.001, dist * 1000.)
        return tr

    def orbit(self, azim, elev):
        if self.ortho:
            return
        super().orbit(azim, elev)

    def mousePressEvent(self, ev):
        self.setFocus()
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

        if ev.button() == Qt.LeftButton and self.gizmo and self.gizmo.body:
            axis = self.gizmo.pick_axis(ev.x(), ev.y(), self)
            if axis:
                self._drag_axis = axis
                self._drag_start = (ev.x(), ev.y())
                self._drag_body = self.gizmo.body
                self._drag_pos0 = self.gizmo.body.pos.copy()
                self._drag_rest0 = self.gizmo.body.rest_pos.copy()
                ev.accept()
                return

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if self._drag_axis:
            dx = ev.x() - self._drag_start[0]
            dy = ev.y() - self._drag_start[1]
            world_delta = self._compute_drag_delta(dx, dy, self._drag_axis)
            offset = np.zeros(3, dtype=np.float32)
            idx = {'x': 0, 'y': 1, 'z': 2}[self._drag_axis]
            offset[idx] = world_delta
            self._drag_body.pos = self._drag_pos0 + offset
            self._drag_body.rest_pos = self._drag_rest0 + offset
            if self.gizmo:
                self.gizmo.update_position()
            main_window = getattr(self, 'main_window', None)
            if main_window:
                main_window.update_visuals(self._drag_body)
            self.update()
            ev.accept()
            return

        if ev.buttons() == Qt.RightButton:
            if ev.modifiers() & Qt.ControlModifier:
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == Qt.MiddleButton:
            if ev.modifiers() & Qt.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')

    def mouseReleaseEvent(self, ev):
        if self._drag_axis:
            body = self._drag_body
            self._drag_axis = None
            if body and getattr(self, 'main_window', None):
                self.main_window.update_visuals(body)
                if self.main_window.quad_mode:
                    self.main_window._sync_ortho_views()
            ev.accept()
            return

    def _compute_drag_delta(self, mouse_dx, mouse_dy, axis):
        center = np.mean(self._drag_pos0, axis=0)
        axis_dir = {'x': np.array([1, 0, 0], dtype=np.float64),
                     'y': np.array([0, 1, 0], dtype=np.float64),
                     'z': np.array([0, 0, 1], dtype=np.float64)}[axis]
        ref = 0.01
        view_mat = self.viewMatrix()
        vp = self._default_viewport()
        proj_mat = self.projectionMatrix(vp, vp)
        mvp = proj_mat * view_mat
        w, h = self.width(), self.height()

        def to_screen(pos):
            v = QVector3D(float(pos[0]), float(pos[1]), float(pos[2]))
            c = mvp.map(v)
            return np.array([(c.x() * 0.5 + 0.5) * w, (0.5 - c.y() * 0.5) * h])

        s0 = to_screen(center)
        s1 = to_screen(center + axis_dir * ref)
        screen_dir = s1 - s0
        screen_len = np.linalg.norm(screen_dir)
        if screen_len < 0.1:
            return 0.0
        screen_dir /= screen_len
        ppu = screen_len / ref
        projected_px = np.dot(np.array([mouse_dx, mouse_dy], dtype=np.float64), screen_dir)
        return float(projected_px / ppu)

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_1:
            self.ortho = True
            self.view_mode = 'front'
            self.setCameraPosition(elevation=0, azimuth=90)
        elif key == Qt.Key_3:
            self.ortho = True
            self.view_mode = 'right'
            self.setCameraPosition(elevation=0, azimuth=0)
        elif key == Qt.Key_7:
            self.ortho = True
            self.view_mode = 'top'
            self.setCameraPosition(elevation=90, azimuth=90)
        elif key == Qt.Key_5:
            self.ortho = not self.ortho
        elif key == Qt.Key_0:
            self.ortho = False
            self.view_mode = 'perspective'
            self.setCameraPosition(distance=0.5, elevation=15, azimuth=90)
        else:
            super().keyPressEvent(ev)
            return
        lbl = VIEW_LABELS.get(self.view_mode, self.view_mode.capitalize())
        if self.ortho and self.view_mode == 'perspective':
            lbl = 'Ortho'
        self._view_label.setText(lbl)
        self.update()


class OrthoCanvas(QWidget):
    def __init__(self, mode='front', parent=None):
        super().__init__(parent)
        self.mode = mode
        self.bodies = []
        self.show_springs = True
        self.show_penetrator = True
        self.show_armor = True
        self.setMinimumSize(100, 100)
        self.setStyleSheet("background: #16181c;")

        self._label = QLabel(VIEW_LABELS.get(mode, mode.capitalize()), self)
        self._label.setStyleSheet(
            "color: #6af; background: rgba(0,0,0,120); font-size: 11px; padding: 3px 6px;")
        self._label.move(6, 6)

        self._pan = np.array([0.0, 0.0])
        self._zoom = 1.0
        self._last_mouse = None

    def set_bodies(self, bodies):
        self.bodies = bodies
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(22, 24, 28))

        w, h = self.width(), self.height()
        cx, cy = w / 2 + self._pan[0], h / 2 + self._pan[1]
        ax_h, ax_v, lbl_h, lbl_v = PROJ_AXES[self.mode]
        scale = min(w, h) * 2.5 * self._zoom

        grid_pen = QPen(QColor(35, 35, 35, 120))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        grid_step = 0.04 * scale
        if grid_step > 4:
            for i in range(-30, 31):
                gx = cx + i * grid_step
                gy = cy + i * grid_step
                if 0 <= gx <= w:
                    p.drawLine(int(gx), 0, int(gx), h)
                if 0 <= gy <= h:
                    p.drawLine(0, int(gy), w, int(gy))

        for body in self.bodies:
            visible = self.show_penetrator if body.body_type == 'penetrator' else self.show_armor
            if body.pos is None or not visible:
                continue
            mat = MATERIALS[body.material]
            base_color = mat['color']

            pos_h = body.pos[:, ax_h]
            pos_v = body.pos[:, ax_v]
            sx = cx + pos_h * scale
            sy = cy - pos_v * scale

            if self.show_springs and body.springs is not None and len(body.springs) > 0:
                lc = QColor(int(base_color[0] * 153), int(base_color[1] * 153),
                            int(base_color[2] * 153), 60)
                spring_pen = QPen(lc)
                spring_pen.setWidth(1)
                p.setPen(spring_pen)
                springs = body.springs
                path = QPainterPath()
                si = springs[:, 0]
                sj = springs[:, 1]
                for i, j in zip(si, sj):
                    path.moveTo(float(sx[i]), float(sy[i]))
                    path.lineTo(float(sx[j]), float(sy[j]))
                p.drawPath(path)

            vel_mag = np.linalg.norm(body.vel, axis=1)
            vel_norm = np.clip(vel_mag / _VEL_MAX, 0.0, 1.0)
            cr = np.interp(vel_norm, _VEL_T, _VEL_RGB[:, 0])
            cg = np.interp(vel_norm, _VEL_T, _VEL_RGB[:, 1])
            cb = np.interp(vel_norm, _VEL_T, _VEL_RGB[:, 2])

            radius = 3
            for k in range(len(sx)):
                color = QColor(int(cr[k] * 255), int(cg[k] * 255), int(cb[k] * 255))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(color))
                p.drawEllipse(int(sx[k] - radius), int(sy[k] - radius),
                              radius * 2, radius * 2)

        axis_pen = QPen(QColor(100, 100, 100, 180))
        axis_pen.setWidth(1)
        p.setPen(axis_pen)
        p.drawLine(int(cx), 0, int(cx), h)
        p.drawLine(0, int(cy), w, int(cy))

        label_font = p.font()
        label_font.setPointSize(9)
        p.setFont(label_font)
        p.setPen(QColor(80, 80, 80))
        p.drawText(w - 18, int(cy) - 4, lbl_h)
        p.drawText(int(cx) + 4, 14, lbl_v)

        p.end()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        factor = 1.001 ** delta
        self._zoom *= factor
        self._zoom = max(0.1, min(50.0, self._zoom))
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton or ev.button() == Qt.MiddleButton:
            self._last_mouse = (ev.x(), ev.y())

    def mouseMoveEvent(self, ev):
        if self._last_mouse is not None:
            dx = ev.x() - self._last_mouse[0]
            dy = ev.y() - self._last_mouse[1]
            self._pan[0] += dx
            self._pan[1] += dy
            self._last_mouse = (ev.x(), ev.y())
            self.update()

    def mouseReleaseEvent(self, ev):
        self._last_mouse = None
