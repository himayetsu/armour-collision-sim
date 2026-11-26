"""Graphics-only refinement for visible quadrilateral surface patches."""

from operator import index

import numpy as np


def _quad_refinement(subdivisions):
    if subdivisions == 1:
        return np.array([[0, 1, 2], [0, 2, 3]]), np.empty((0, 4), dtype=np.float32)

    parameters = np.arange(1, subdivisions, dtype=np.float32) / subdivisions
    u, v = np.meshgrid(parameters, parameters, indexing='xy')
    weights = np.stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v), axis=-1)
    grid = np.arange(4, 4 + u.size, dtype=np.intp).reshape(u.shape)
    a = grid[:-1, :-1].ravel()
    b = grid[:-1, 1:].ravel()
    c = grid[1:, 1:].ravel()
    d = grid[1:, :-1].ravel()
    faces = [np.column_stack((a, b, c)), np.column_stack((a, c, d))]

    # Fill the four strips between the inner grid and the original perimeter.
    # Leaving each outer edge whole lets an ordinary triangle share it exactly.
    sides = (grid[0], grid[:, -1], grid[-1, ::-1], grid[::-1, 0])
    middle = (len(grid) - 1) // 2
    first_half = np.arange(len(grid) - 1) < middle
    for first, side in enumerate(sides):
        next_corner = (first + 1) % 4
        faces.append(np.array([[first, next_corner, side[middle]]]))
        anchor = np.where(first_half, first, next_corner)
        faces.append(np.column_stack((anchor, side[1:], side[:-1])))
    return np.concatenate(faces), weights.reshape(-1, 4)


class SurfacePatchMesh:
    """Refine quad interiors and retain standalone triangles and shared edges."""

    def __init__(self, quads, triangles, subdivisions):
        subdivisions = index(subdivisions)
        if subdivisions < 1:
            raise ValueError('Surface subdivisions must be positive')
        self._quads = np.asarray(quads, dtype=np.intp).reshape(-1, 4)
        triangles = np.asarray(triangles, dtype=np.intp).reshape(-1, 3)
        self._vertices, local_indices = np.unique(
            np.concatenate((self._quads.ravel(), triangles.ravel())), return_inverse=True)
        local_quads = local_indices[:self._quads.size].reshape(self._quads.shape)
        local_triangles = local_indices[self._quads.size:].reshape(triangles.shape)

        pattern, self._weights = _quad_refinement(subdivisions)
        new_vertices = np.arange(len(self._quads) * len(self._weights), dtype=np.intp)
        new_vertices = new_vertices.reshape(len(self._quads), len(self._weights)) + len(self._vertices)
        patch_vertices = np.column_stack((local_quads, new_vertices))
        patch_faces = patch_vertices[:, pattern].reshape(-1, 3)
        self.faces = np.concatenate((patch_faces, local_triangles))

    def sample(self, values):
        """Evaluate vertex positions or other attributes on the cached mesh."""
        values = np.asarray(values)
        inner = np.einsum('sc,qcd->qsd', self._weights, values[self._quads])
        inner = inner.reshape(len(self._quads) * len(self._weights), values.shape[1])
        return np.concatenate((values[self._vertices], inner))
