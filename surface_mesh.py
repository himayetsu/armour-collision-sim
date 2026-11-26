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


class SurfacePointSampler:
    """Cache display samples on the exterior faces of a deformable mesh."""

    def __init__(self, rest_positions, faces, spacing=0.002, max_points=100_000):
        if not np.isfinite(spacing) or spacing <= 0 or max_points < 0:
            raise ValueError('Surface sampling needs positive spacing and a nonnegative point limit')
        faces = np.asarray(faces, dtype=np.intp).reshape(-1, 3)
        rest_positions = np.asarray(rest_positions)
        corners = np.unique(faces)
        face_edges = np.sort(np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1), axis=2)
        edges, face_edge_indices = np.unique(face_edges.reshape(-1, 2), axis=0, return_inverse=True)
        edge_lengths = np.linalg.norm(rest_positions[edges[:, 1]] - rest_positions[edges[:, 0]], axis=1)
        edge_intervals = np.maximum(1, np.ceil(edge_lengths / spacing).astype(np.int64))
        face_intervals = edge_intervals[face_edge_indices.reshape(-1, 3)].max(axis=1)

        # Share edge samples across adjacent faces. A face with n intervals
        # contributes (n - 1)(n - 2)/2 samples strictly inside its edges.
        interior_counts = (face_intervals - 1) * (face_intervals - 2) // 2
        counts = np.concatenate((np.ones(len(corners), dtype=np.int64), edge_intervals - 1, interior_counts))
        ends = np.cumsum(counts)
        total = int(ends[-1]) if len(ends) else 0
        if total > max_points:
            # Select across the full sample range before allocating anything
            # proportional to the potentially oversized lattice.
            chosen = np.linspace(0, total - 1, max_points, dtype=np.int64)
        else:
            chosen = np.arange(total)
        groups = np.searchsorted(ends, chosen, side='right')
        offsets = chosen - (ends - counts)[groups]
        self._indices = np.empty((len(chosen), 3), dtype=np.intp)
        self._weights = np.zeros((len(chosen), 3), dtype=np.float32)

        edge_start = len(corners)
        face_start = edge_start + len(edges)
        on_corner = groups < edge_start
        self._indices[on_corner] = corners[groups[on_corner], None]
        self._weights[on_corner, 0] = 1

        on_edge = (groups >= edge_start) & (groups < face_start)
        edge_ids = groups[on_edge] - edge_start
        edge_fraction = (offsets[on_edge] + 1) / edge_intervals[edge_ids]
        self._indices[on_edge] = edges[edge_ids][:, [0, 1, 0]]
        self._weights[on_edge, 0] = 1 - edge_fraction
        self._weights[on_edge, 1] = edge_fraction

        in_face = groups >= face_start
        face_ids = groups[in_face] - face_start
        # Count rows backward from the final row, which has a single point.
        # The triangular-number inverse finds the row for each selected index.
        reverse_offset = interior_counts[face_ids] - offsets[in_face] - 1
        row_width = ((np.sqrt(8 * reverse_offset + 1) + 1) / 2).astype(np.int64)
        intervals = face_intervals[face_ids]
        row = (intervals - row_width - 1) / intervals
        column = (row_width * (row_width + 1) // 2 - reverse_offset) / intervals
        self._indices[in_face] = faces[face_ids]
        self._weights[in_face] = np.column_stack((1 - row - column, row, column))

    def sample(self, values):
        """Interpolate an (n_vertices, n_attributes) array at the display points."""
        return np.einsum('ij,ijk->ik', self._weights, np.asarray(values)[self._indices])


class BodySurfaceTopology:
    """Map the rest shell to the body's connected fragments."""

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
        self.vertices, local_indices = np.unique(self._rest_faces, return_inverse=True)
        self._rest_mesh_faces = local_indices.reshape(-1, 3)
        self._surface_particles = np.zeros(self.particle_count, dtype=bool)
        self._surface_particles[self.vertices] = True
        self._face_mask = None
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
        present = np.zeros(self._bonded_edges.shape, dtype=bool)
        present[self._bonded_edges] = np.isin(self._bond_keys, self._spring_keys(springs))
        bonded = np.zeros(self.particle_count, dtype=bool)
        if springs is not None:
            bonded[springs.ravel()] = True
        # Two surviving edges already connect the triangle. When fewer edges
        # survive, the surrounding mesh can still connect its three corners.
        keep = present.sum(axis=1) >= 2
        needs_connectivity = ~keep & bonded[self._rest_faces].all(axis=1)
        if needs_connectivity.any():
            i, j = springs.T
            graph = csr_matrix((np.ones(len(i), dtype=bool), (i, j)),
                               shape=(self.particle_count, self.particle_count))
            _, components = connected_components(graph, directed=False)
            labels = components[self._rest_faces[needs_connectivity]]
            keep[needs_connectivity] = (labels == labels[:, :1]).all(axis=1)
        if not np.array_equal(self._face_mask, keep):
            self._face_mask = keep
            self.faces = self._rest_faces[keep]
            self.mesh_faces = self._rest_mesh_faces[keep]

        if not len(self.faces):
            self.particles = np.arange(self.particle_count)
            return
        on_face = np.zeros(self.particle_count, dtype=bool)
        on_face[self.faces.ravel()] = True
        self.particles = np.flatnonzero(~on_face & (self._surface_particles | ~bonded))
