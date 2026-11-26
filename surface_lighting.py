"""Indexed display geometry with separate normals at structural creases."""

import numpy as np


_CANCELLATION_TOLERANCE = 8 * np.finfo(np.float32).eps


def surface_groups(faces, lattice_shape=None, axial_positions=None):
    """Label plate boundary planes or the rear cap of a round shell."""
    faces = np.asarray(faces, dtype=np.intp).reshape(-1, 3)
    if lattice_shape is not None:
        coordinates = np.unravel_index(faces, lattice_shape)
        groups = np.full(len(faces), 6, dtype=np.uint8)
        for axis, (indices, length) in enumerate(zip(coordinates, lattice_shape)):
            groups[(indices == 0).all(axis=1)] = 2 * axis
            groups[(indices == length - 1).all(axis=1)] = 2 * axis + 1
        return groups
    if axial_positions is not None and len(faces):
        axial_positions = np.asarray(axial_positions)
        return (axial_positions[faces] == axial_positions.min()).all(axis=1).astype(np.uint8)
    return np.zeros(len(faces), dtype=np.uint8)


class SurfaceShading:
    """Share normals within each group and duplicate vertices at creases."""

    def __init__(self, faces, groups):
        faces = np.asarray(faces, dtype=np.intp).reshape(-1, 3)
        groups = np.asarray(groups)
        if groups.shape != (len(faces),):
            raise ValueError('Each display face needs one shading group')
        stride = int(faces.max(initial=0)) + 1
        _, group_indices = np.unique(groups, return_inverse=True)
        keys = faces + group_indices[:, None] * stride
        vertices, remapped = np.unique(keys, return_inverse=True)
        self.vertex_indices = vertices % stride
        self.faces = remapped.reshape(-1, 3)
        self._max_faces_per_vertex = np.bincount(self.faces.ravel()).max(initial=0)

    def normals(self, positions):
        """Calculate area-weighted normals for the remapped vertex positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (len(self.vertex_indices), 3):
            raise ValueError('Surface normals need positions in remapped vertex order')
        a, b, c = positions[self.faces].transpose(1, 0, 2)
        with np.errstate(invalid='ignore', over='ignore'):
            face_normals = np.cross(b - a, c - a)
            face_normals[~np.isfinite(face_normals).all(axis=1)] = 0
            corner_indices = self.faces.ravel()
            summed = np.column_stack([
                np.bincount(corner_indices, weights=component.repeat(3), minlength=len(positions))
                for component in face_normals.T
            ])
            lengths = np.linalg.norm(summed, axis=1, keepdims=True)
            peak = max(face_normals.max(initial=0), -face_normals.min(initial=0))
            # At most this many components contribute to any local scale.
            bound = 3 * self._max_faces_per_vertex * peak * _CANCELLATION_TOLERANCE
            if peak > 0 and lengths.min(initial=np.inf) <= bound:
                # Scale the roundoff allowance by incident areas, so tiny valid
                # faces keep their weighted normal even beside a coarse mesh.
                weights = np.abs(face_normals).sum(axis=1)
                local_scale = np.bincount(corner_indices, weights=weights.repeat(3), minlength=len(positions))
                cancelled = (local_scale > 0) & (lengths[:, 0] <= _CANCELLATION_TOLERANCE * local_scale)
                if cancelled.any():
                    # Folded neighbors can cancel while their faces still have area.
                    face_ids, corners = np.nonzero((weights > 0)[:, None] & cancelled[self.faces])
                    vertices, first = np.unique(self.faces[face_ids, corners], return_index=True)
                    summed[vertices] = face_normals[face_ids[first]]
                    lengths[vertices] = np.linalg.norm(summed[vertices], axis=1, keepdims=True)
        normals = np.zeros((len(positions), 3), dtype=np.float32)
        np.divide(summed, lengths, out=normals, where=np.isfinite(lengths) & (lengths > 0))
        return normals
