"""Build and maintain display meshes for particle bodies."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def plate_faces(shape):
    """Triangulate the six boundary planes of a thickness/height/width lattice."""
    nz, ny, nx = shape
    layer = ny * nx
    # The lattice's width, height and thickness axes form a left-handed basis.
    sides = (
        # Rows, columns, row stride, column stride, and first particle.
        (ny, nx, nx, 1, 0), (nx, ny, 1, nx, layer * (nz - 1)),
        (nx, nz, 1, layer, 0), (nz, nx, layer, 1, nx * (ny - 1)),
        (nz, ny, layer, nx, 0), (ny, nz, nx, layer, nx - 1),
    )
    faces = []
    for rows, columns, row_step, col_step, first in sides:
        row_ids = np.arange(rows - 1, dtype=np.int32) * row_step
        col_ids = np.arange(columns - 1, dtype=np.int32) * col_step
        a = (first + row_ids[:, None] + col_ids).ravel()
        b, c, d = a + col_step, a + row_step + col_step, a + row_step
        faces.extend((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    return np.concatenate(faces)


def _join_rings(inner, outer):
    """Stitch angularly ordered rings, advancing the next closest angular sample."""
    m, n = len(inner), len(outer)
    if m == 1:
        return [(inner[0], outer[(j + 1) % n], outer[j]) for j in range(n)]
    if n == 1:
        return [(inner[i], inner[(i + 1) % m], outer[0]) for i in range(m)]

    faces = []
    i = j = 0
    while i < m or j < n:
        a, b = inner[i % m], outer[j % n]
        if (i + 1) * n <= (j + 1) * m:
            faces.append((a, inner[(i + 1) % m], b))
            i += 1
        else:
            faces.append((a, outer[(j + 1) % n], b))
            j += 1
    return faces


def cylinder_faces(rest_pos):
    """Cover the rear cap, the outer ring of each section, and the taper tip."""
    # Every axial section starts at the center; the generated rings are grouped
    # in increasing radius and all start at the same angle.
    change = np.flatnonzero(rest_pos[1:, 0] != rest_pos[:-1, 0]) + 1
    sections = np.split(np.arange(len(rest_pos), dtype=np.int32), change)
    faces = []
    outer_rings = []
    for section in sections:
        if len(section) <= 1:
            continue
        offsets = rest_pos[section, 1:] - rest_pos[section[0], 1:]
        r2 = np.sum(offsets * offsets, axis=1)
        boundaries = np.flatnonzero(~np.isclose(r2[1:], r2[:-1], rtol=1e-4, atol=0)) + 1
        last_ring = section[boundaries[-1]:]
        if not outer_rings:
            rings = np.split(section, boundaries)
            for inner, outer in zip(rings, rings[1:]):
                faces.extend(_join_rings(inner, outer))
        outer_rings.append(last_ring)

    if outer_rings:
        outer_rings.append(sections[-1][-1:])
        for inner, outer in zip(outer_rings, outer_rings[1:]):
            faces.extend(_join_rings(inner, outer))
    return np.asarray(faces, dtype=np.int32).reshape(-1, 3)


def _edge_keys(a, b, particle_count):
    return np.minimum(a, b).astype(np.int64) * particle_count + np.maximum(a, b)


class BodySurfaceTopology:
    """Cache the rest shell and discard facets whose supporting bonds break."""

    def __init__(self, body):
        self.particle_count = body.n_particles
        if body.body_type == 'armor':
            self._rest_faces = plate_faces(body.lattice_shape)
        else:
            self._rest_faces = body.surface_faces

        face_keys = _edge_keys(self._rest_faces, self._rest_faces[:, [1, 2, 0]],
                               self.particle_count)
        rest_keys = self._spring_keys(body.springs)
        self._bonded_edges = np.isin(face_keys, rest_keys)
        self._bond_keys = face_keys[self._bonded_edges]
        self._needs_connectivity = self._bonded_edges.sum(axis=1) < 2
        self.vertices, local_indices = np.unique(self._rest_faces, return_inverse=True)
        self._rest_mesh_faces = local_indices.reshape(-1, 3)
        self._surface_particles = np.zeros(self.particle_count, dtype=bool)
        self._surface_particles[self.vertices] = True
        self._last_springs = object()
        self.update(body.springs)

    def _spring_keys(self, springs):
        if springs is None:
            return np.empty(0, dtype=np.int64)
        return _edge_keys(springs[:, 0], springs[:, 1], self.particle_count)

    def update(self, springs):
        # The solver replaces the spring array when bonds are removed.
        if springs is self._last_springs:
            return
        self._last_springs = springs
        present = np.ones(self._bonded_edges.shape, dtype=bool)
        present[self._bonded_edges] = np.isin(self._bond_keys, self._spring_keys(springs))
        bonded = np.zeros(self.particle_count, dtype=bool)
        if springs is not None:
            bonded[springs.ravel()] = True
        keep = present.all(axis=1) & bonded[self._rest_faces].all(axis=1)
        weak_faces = keep & self._needs_connectivity
        if weak_faces.any():
            # The remaining direct springs may not connect all three corners.
            i, j = springs.T
            graph = csr_matrix((np.ones(len(i), dtype=bool), (i, j)),
                               shape=(self.particle_count, self.particle_count))
            _, components = connected_components(graph, directed=False)
            labels = components[self._rest_faces[weak_faces]]
            keep[weak_faces] = (labels == labels[:, :1]).all(axis=1)
        self.faces = self._rest_faces[keep]
        self.mesh_faces = self._rest_mesh_faces[keep]

        if not len(self.faces):
            self.particles = np.arange(self.particle_count)
            return
        on_face = np.zeros(self.particle_count, dtype=bool)
        on_face[self.faces.ravel()] = True
        self.particles = np.flatnonzero(~on_face & (self._surface_particles | ~bonded))
