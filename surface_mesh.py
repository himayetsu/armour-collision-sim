"""Build and maintain display meshes for particle bodies."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from surface_cuts import build_cut_faces


def plate_faces(shape):
    """Triangulate the six boundary planes of a thickness/height/width lattice."""
    return _plate_surface(shape)[0]


def _plate_surface(shape):
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
    quad_pairs = []
    offset = 0
    for rows, columns, row_step, col_step, first in sides:
        row_ids = np.arange(rows - 1, dtype=np.int32) * row_step
        col_ids = np.arange(columns - 1, dtype=np.int32) * col_step
        a = (first + row_ids[:, None] + col_ids).ravel()
        b, c, d = a + col_step, a + row_step + col_step, a + row_step
        faces.extend((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
        first_faces = np.arange(offset, offset + len(a), dtype=np.int32)
        quad_pairs.append(np.column_stack((first_faces, first_faces + len(a))))
        offset += 2 * len(a)
    return np.concatenate(faces), np.concatenate(quad_pairs)


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
    return cylinder_surface(rest_pos)[0]


def cylinder_surface(rest_pos):
    """Return the cylinder shell and pairs of faces in its regular ring strips."""
    # Every axial section starts at the center; the generated rings are grouped
    # in increasing radius and all start at the same angle.
    change = np.flatnonzero(rest_pos[1:, 0] != rest_pos[:-1, 0]) + 1
    sections = np.split(np.arange(len(rest_pos), dtype=np.int32), change)
    joins = []
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
            joins.extend(zip(rings, rings[1:]))
        outer_rings.append(last_ring)

    if outer_rings:
        outer_rings.append(sections[-1][-1:])
        joins.extend(zip(outer_rings, outer_rings[1:]))

    faces = []
    quad_starts = []
    for inner, outer in joins:
        strip = _join_rings(inner, outer)
        if 1 < len(inner) == len(outer):
            quad_starts.extend(range(len(faces), len(faces) + len(strip), 2))
        faces.extend(strip)
    starts = np.asarray(quad_starts, dtype=np.int32)
    return np.asarray(faces, dtype=np.int32).reshape(-1, 3), np.column_stack((starts, starts + 1))


def _quad_corners(faces, pairs):
    first, second = faces[pairs[:, 0]], faces[pairs[:, 1]]
    shared = first[:, :, None] == second[:, None, :]
    first_corner = np.argmin(shared.any(axis=2), axis=1)
    second_corner = np.argmin(shared.any(axis=1), axis=1)
    row = np.arange(len(pairs))
    # Use the first face's winding, placing its shared edge across corners a-c.
    return np.column_stack((first[row, (first_corner - 1) % 3], first[row, first_corner],
                            first[row, (first_corner + 1) % 3], second[row, second_corner]))


def _edge_keys(a, b, particle_count):
    return np.minimum(a, b).astype(np.int64) * particle_count + np.maximum(a, b)


class BodySurfaceTopology:
    """Map the rest shell to the body's connected fragments."""

    # A display tolerance for unsupported edges, independent of spring failure.
    _MAX_UNSUPPORTED_EDGE_STRETCH = 2.0

    def __init__(self, body):
        self.particle_count = body.n_particles
        if body.body_type == 'armor':
            rest_faces, self._quad_pairs = _plate_surface(body.lattice_shape)
        else:
            rest_faces = body.surface_faces
            self._quad_pairs = getattr(body, 'surface_quad_pairs', np.empty((0, 2), dtype=np.int32))
        self._shell_face_count = len(rest_faces)
        self._rest_quads = _quad_corners(rest_faces, self._quad_pairs)
        alternative = np.stack((self._rest_quads[:, [0, 1, 3]],
                                self._rest_quads[:, [1, 2, 3]]), axis=1)
        self._alternate_pairs = np.arange(len(rest_faces), len(rest_faces) + 2 * len(alternative)).reshape(-1, 2)
        self._candidates = np.concatenate((rest_faces, alternative.reshape(-1, 3)))
        edge_keys = _edge_keys(self._candidates, self._candidates[:, [1, 2, 0]], self.particle_count)
        self._edge_keys, edge_indices = np.unique(edge_keys, return_inverse=True)
        self._face_edges = edge_indices.reshape(-1, 3)
        self.vertices, local_indices = np.unique(self._candidates, return_inverse=True)
        self._local_faces = local_indices.reshape(-1, 3)
        self._surface_particles = np.zeros(self.particle_count, dtype=bool)
        self._surface_particles[self.vertices] = True
        rest_positions = getattr(body, 'surface_rest_pos', None)
        if rest_positions is None:
            rest_positions = getattr(body, 'rest_pos', None)
        self._rest_positions = rest_positions
        self._stretch_limit_sq = None
        if rest_positions is not None:
            self._edge_starts, self._edge_ends = np.divmod(self._edge_keys, max(1, self.particle_count))
            delta = rest_positions[self._edge_ends] - rest_positions[self._edge_starts]
            self._stretch_limit_sq = np.einsum('ij,ij->i', delta, delta) * self._MAX_UNSUPPORTED_EDGE_STRETCH**2
        self._face_mask = None
        self._quad_mask = None
        self._triangle_mask = None
        self.particles = np.empty(0, dtype=np.intp)
        self._cut_faces = np.empty((0, 3), dtype=np.intp)
        self._cut_groups = np.empty(0, dtype=np.intp)
        self._cut_shell = None
        self._cut_components = None
        self._last_springs = object()
        self.update(body.springs)

    def _spring_keys(self, springs):
        if springs is None:
            return np.empty(0, dtype=np.int64)
        return _edge_keys(springs[:, 0], springs[:, 1], self.particle_count)

    def _update_bond_support(self, springs):
        self._components = None
        present = np.isin(self._edge_keys, self._spring_keys(springs))[self._face_edges]
        self._bonded = np.zeros(self.particle_count, dtype=bool)
        if springs is not None:
            self._bonded[springs.ravel()] = True
        # Two surviving edges already connect the triangle. When fewer edges
        # survive, the surrounding mesh can still connect its three corners.
        keep = present.sum(axis=1) >= 2
        weak = np.flatnonzero(~keep & self._bonded[self._candidates].all(axis=1))
        if len(weak):
            _, components = self._fragment_labels(springs)
            labels = components[self._candidates[weak]]
            weak = weak[(labels == labels[:, :1]).all(axis=1)]
            keep[weak] = True
        self._bond_support = keep
        self._stretch_faces = np.empty(0, dtype=np.intp)
        if self._stretch_limit_sq is not None and len(weak):
            rows, columns = np.nonzero(~present[weak])
            self._stretch_faces = weak[rows]
            measured_edges = self._face_edges[self._stretch_faces, columns]
            self._stretch_edges, self._stretch_edge_slots = np.unique(measured_edges, return_inverse=True)

    def _fragment_labels(self, springs):
        if self._components is None:
            i, j = springs.T
            graph = csr_matrix((np.ones(len(i), dtype=bool), (i, j)),
                               shape=(self.particle_count, self.particle_count))
            self._components = connected_components(graph, directed=False)
        return self._components

    def cut_surfaces(self):
        """Build display caps on demand after the exterior has been updated."""
        if self._rest_positions is not None and 0 < len(self.faces) < self._shell_face_count:
            count, labels = self._fragment_labels(self._last_springs)
            if count > 1:
                # A bond loss inside a fragment can leave both the visible
                # boundary and its component membership unchanged.
                if self._cut_shell is not self.faces or (
                        self._cut_components is not labels and not np.array_equal(self._cut_components, labels)):
                    cut_faces, cut_groups = build_cut_faces(
                        self.faces, self._candidates, self._rest_positions, labels)
                    if not np.array_equal(self._cut_faces, cut_faces):
                        self._cut_faces = cut_faces
                        self._cut_groups = cut_groups
                self._cut_shell, self._cut_components = self.faces, labels
                return self._cut_faces, self._cut_groups
        self._cut_shell = self._cut_components = None
        if len(self._cut_faces):
            self._cut_faces = np.empty((0, 3), dtype=np.intp)
            self._cut_groups = np.empty(0, dtype=np.intp)
        return self._cut_faces, self._cut_groups

    def _select_faces(self, keep):
        primary_counts = keep[self._quad_pairs].sum(axis=1)
        alternate_counts = keep[self._alternate_pairs].sum(axis=1)
        use_alternate = alternate_counts > primary_counts
        keep[self._quad_pairs[use_alternate]] = False
        keep[self._alternate_pairs[~use_alternate]] = False
        whole_quads = np.maximum(primary_counts, alternate_counts) == 2
        face_change = not np.array_equal(self._face_mask, keep)
        if face_change:
            self._face_mask = keep
            self.faces = self._candidates[keep]
            self.mesh_faces = self._local_faces[keep]
        if not np.array_equal(self._quad_mask, whole_quads):
            self._quad_mask = whole_quads
            self.quads = self._rest_quads[whole_quads]
        keep_triangles = keep.copy()
        keep_triangles[self._quad_pairs[whole_quads]] = False
        keep_triangles[self._alternate_pairs[whole_quads]] = False
        if not np.array_equal(self._triangle_mask, keep_triangles):
            self._triangle_mask = keep_triangles
            self.triangles = self._candidates[keep_triangles]
        return face_change

    def update(self, springs, positions=None):
        """Refresh the shell, optionally checking moving weak edges for visible gaps."""
        # The solver replaces the spring array when bonds are removed.
        springs_changed = springs is not self._last_springs
        if springs_changed:
            self._last_springs = springs
            self._update_bond_support(springs)
            self._last_stretched_edges = None
        check_stretch = positions is not None and len(self._stretch_faces) > 0
        if not springs_changed and not check_stretch:
            return
        if check_stretch:
            edges = self._stretch_edges
            delta = positions[self._edge_ends[edges]] - positions[self._edge_starts[edges]]
            stretched = np.einsum('ij,ij->i', delta, delta) > self._stretch_limit_sq[edges]
            if not springs_changed and np.array_equal(self._last_stretched_edges, stretched):
                return
            self._last_stretched_edges = stretched

        keep = self._bond_support.copy()
        if check_stretch:
            keep[self._stretch_faces[stretched[self._stretch_edge_slots]]] = False

        faces_changed = self._select_faces(keep)
        if not faces_changed and not springs_changed:
            return
        if not len(self.faces):
            particles = np.arange(self.particle_count)
        else:
            on_face = np.zeros(self.particle_count, dtype=bool)
            on_face[self.faces.ravel()] = True
            particles = np.flatnonzero(~on_face & (self._surface_particles | ~self._bonded))
        if not np.array_equal(self.particles, particles):
            self.particles = particles
