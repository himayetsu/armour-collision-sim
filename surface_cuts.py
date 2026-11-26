"""Close verified planar cuts in a separated display shell."""

from dataclasses import dataclass

import numpy as np


# Bound ear clipping while allowing verified large convex sections.
_EAR_CLIP_LIMIT = 256
_POLYGON_TOLERANCE = 1e-7


@dataclass
class _Rim:
    vertices: np.ndarray
    center: np.ndarray
    normal: np.ndarray
    axes: np.ndarray
    tolerance: float
    has_depth: bool


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _edge_key(a, b):
    return min(a, b), max(a, b)


def _face_edges(faces):
    return np.column_stack((faces.ravel(), faces[:, [1, 2, 0]].ravel()))


def _edge_codes(edges, particle_count):
    return edges.min(axis=1).astype(np.int64) * particle_count + edges.max(axis=1)


def _boundary_loops(boundary, labels):
    owners = labels[boundary[:, 0]]
    order = np.argsort(owners, kind='stable')
    owners, boundary = owners[order], boundary[order]
    changes = np.flatnonzero(owners[1:] != owners[:-1]) + 1
    for edges in np.split(boundary, changes):
        if len(edges) < 3:
            continue
        starts, ends = edges.T
        if len(np.unique(starts)) != len(edges) or not np.array_equal(np.sort(starts), np.sort(ends)):
            continue
        next_vertex = dict(edges)
        first = min(next_vertex)
        loop = [first]
        while next_vertex[loop[-1]] != first:
            loop.append(next_vertex[loop[-1]])
        # A component with several openings may have nested rims or unrelated holes.
        if len(loop) == len(edges):
            yield labels[first], np.asarray(loop, dtype=np.intp)


def _opposing_components(candidates, labels):
    a = candidates
    b = candidates[:, [1, 2, 0]]
    c = candidates[:, [2, 0, 1]]
    across_cut = (labels[a] == labels[b]) & (labels[a] != labels[c])
    neighbors = {}
    for i, j, opposite in zip(a[across_cut], b[across_cut], labels[c[across_cut]]):
        neighbors.setdefault(_edge_key(i, j), set()).add(opposite)
    return neighbors


def _fit_rim(loop, positions, owner_positions):
    # Oppose the exposed shell edge winding, independent of the plane basis.
    vertices = loop[::-1]
    points = positions[vertices].astype(np.float64)
    if not np.isfinite(points).all():
        return None
    center = points.mean(axis=0)
    local = points - center
    edges = np.roll(local, -1, axis=0) - local
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths == 0):
        return None
    _, _, axes = np.linalg.svd(local, full_matrices=False)
    tolerance = 0.01 * np.median(edge_lengths)
    if np.max(np.abs(local @ axes[2])) > tolerance:
        return None
    normal = np.cross(local, edges).sum(axis=0)
    magnitude = np.linalg.norm(normal)
    if magnitude == 0:
        return None
    normal /= magnitude
    depth = (owner_positions - center) @ normal
    if depth.max() > tolerance:
        return None
    return _Rim(vertices, center, normal, axes[:2], tolerance, depth.min() < -tolerance)


def _is_same_cut(rim, opposite, positions):
    offset = opposite.center - rim.center
    if (rim.normal @ opposite.normal >= -0.95 or offset @ rim.normal <= rim.tolerance
            or -offset @ opposite.normal <= opposite.tolerance):
        return False
    current = (positions[rim.vertices] - rim.center) @ rim.axes.T
    other = (positions[opposite.vertices] - rim.center) @ rim.axes.T
    return np.all(np.maximum(current.min(axis=0), other.min(axis=0))
                  < np.minimum(current.max(axis=0), other.max(axis=0)))


def _is_simple_polygon(points):
    tail = np.roll(points, -1, axis=0)
    edges = tail - points
    previous = np.roll(edges, 1, axis=0)
    if np.any((np.abs(_cross(previous, edges)) <= _POLYGON_TOLERANCE)
              & (np.einsum('ij,ij->i', previous, edges) < 0)):
        return False
    for i, (a, b) in enumerate(zip(points, tail)):
        end = len(points) - (i == 0)
        c, d = points[i + 2:end], tail[i + 2:end]
        lower = np.maximum(np.minimum(a, b), np.minimum(c, d))
        upper = np.minimum(np.maximum(a, b), np.maximum(c, d))
        overlaps = (lower <= upper + _POLYGON_TOLERANCE).all(axis=1)
        c, d = c[overlaps], d[overlaps]
        left, right = _cross(b - a, c - a), _cross(b - a, d - a)
        start, end = _cross(d - c, a - c), _cross(d - c, b - c)
        intersects = ((np.minimum(left, right) <= _POLYGON_TOLERANCE)
                      & (np.maximum(left, right) >= -_POLYGON_TOLERANCE)
                      & (np.minimum(start, end) <= _POLYGON_TOLERANCE)
                      & (np.maximum(start, end) >= -_POLYGON_TOLERANCE))
        if intersects.any():
            return False
    return True


def _has_exact_perimeter(triangles, vertex_count):
    edges = _face_edges(triangles)
    _, inverse, counts = np.unique(_edge_codes(edges, vertex_count),
                                   return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        return False
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(np.bincount(inverse, weights=directions)[counts == 2]):
        return False
    perimeter = edges[counts[inverse] == 1]
    rim = np.arange(vertex_count, dtype=np.int64)
    return np.array_equal(np.sort(perimeter[:, 0] * vertex_count + perimeter[:, 1]),
                          rim * vertex_count + np.roll(rim, -1))


def _triangulate_convex(points, winding):
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0):
        return None
    directions = edges / lengths[:, None]
    previous = np.roll(directions, 1, axis=0)
    turns = winding * np.arctan2(_cross(previous, directions),
                                 np.einsum('ij,ij->i', previous, directions))
    # Rotated float32 rest vertices introduce tiny bends along straight runs.
    roundoff = 8 * np.finfo(np.float32).eps * (1 / lengths + 1 / np.roll(lengths, 1))
    tolerance = np.minimum(roundoff, 1e-3)
    if np.any(turns < -tolerance) or not np.isclose(turns.sum(), 2 * np.pi):
        return None
    corners = np.flatnonzero(turns > tolerance)
    if len(corners) < 3:
        return None

    perimeter = list(zip(corners, np.roll(corners, -1)))
    owners = dict.fromkeys(perimeter)
    triangles = []

    def add(triangle):
        for edge in zip(triangle, (triangle[1], triangle[2], triangle[0])):
            if edge in owners:
                owners[edge] = len(triangles)
        triangles.append(triangle)

    for a, b in zip(corners[1:-1], corners[2:]):
        add((corners[0], a, b))

    # Split each corner-to-corner boundary into its authored segments, keeping
    # the opposite corner as the fan anchor. No display positions are modified.
    count = len(points)
    for a, b in perimeter:
        steps = (b - a) % count
        if steps > 1:
            owner = owners[a, b]
            third = next(vertex for vertex in triangles[owner] if vertex != a and vertex != b)
            triangles[owner] = None
            for offset in range(steps):
                add(((a + offset) % count, (a + offset + 1) % count, third))

    faces = np.asarray([triangle for triangle in triangles if triangle is not None])
    if len(faces) != count - 2:
        return None
    a, b, c = points[faces].transpose(1, 0, 2)
    if np.any(winding * _cross(b - a, c - a) <= 64 * np.finfo(np.float64).eps):
        return None
    if not _has_exact_perimeter(faces, count):
        return None
    return faces


def _triangulate(rim, positions):
    points = (positions[rim.vertices] - rim.center) @ rim.axes.T
    span = np.max(np.ptp(points, axis=0))
    if span == 0:
        return None
    points /= span
    signed_area = _cross(points, np.roll(points, -1, axis=0)).sum()
    if abs(signed_area) <= _POLYGON_TOLERANCE:
        return None
    winding = np.sign(signed_area)
    if len(points) > _EAR_CLIP_LIMIT:
        triangles = _triangulate_convex(points, winding)
        return None if triangles is None else rim.vertices[triangles]
    if not _is_simple_polygon(points):
        return None
    remaining = np.arange(len(points))
    triangles = []
    while len(remaining) > 3:
        for i in range(len(remaining)):
            corner_slots = [(i - 1) % len(remaining), i, (i + 1) % len(remaining)]
            corners = remaining[corner_slots]
            a, b, c = points[corners]
            if winding * _cross(b - a, c - a) <= _POLYGON_TOLERANCE:
                continue
            others = points[np.delete(remaining, corner_slots)]
            # Points on an ear's edge also block it, keeping collinear rim samples.
            inside = ((winding * _cross(b - a, others - a) >= -_POLYGON_TOLERANCE)
                      & (winding * _cross(c - b, others - b) >= -_POLYGON_TOLERANCE)
                      & (winding * _cross(a - c, others - c) >= -_POLYGON_TOLERANCE))
            if not inside.any():
                triangles.append(corners)
                remaining = np.delete(remaining, i)
                break
        else:
            return None
    a, b, c = points[remaining]
    if winding * _cross(b - a, c - a) <= _POLYGON_TOLERANCE:
        return None
    triangles.append(remaining)
    return rim.vertices[np.asarray(triangles)]


def _has_shared_diagonal(faces, shell_edges, particle_count):
    keys, counts = np.unique(_edge_codes(_face_edges(faces), particle_count), return_counts=True)
    return np.isin(keys[counts == 2], shell_edges).any()


def build_cut_faces(faces, candidates, positions, labels):
    """Return cap triangles and positive per-rim group IDs for Solid rendering."""
    face_labels = labels[faces]
    caps, groups = [], []
    if len(faces) and np.all(face_labels == face_labels[:, :1]):
        edges = _face_edges(faces)
        shell_edges, inverse, counts = np.unique(
            _edge_codes(edges, len(labels)), return_inverse=True, return_counts=True)
        loops = [] if np.any(counts > 2) else list(_boundary_loops(edges[counts[inverse] == 1], labels))
        rims = {}
        neighbors = _opposing_components(candidates, labels) if len(loops) > 1 else {}
        for owner, loop in loops:
            evidence = [neighbors.get(_edge_key(a, b), set())
                        for a, b in zip(loop, np.roll(loop, -1))]
            if any(len(adjacent) != 1 for adjacent in evidence):
                continue
            targets = set.union(*evidence)
            if len(targets) != 1:
                continue
            rim = _fit_rim(loop, positions, positions[labels == owner])
            if rim is not None:
                rims[owner, targets.pop()] = rim
        for (owner, target), rim in sorted(rims.items()):
            other = rims.get((target, owner))
            if owner >= target or other is None or not _is_same_cut(rim, other, positions):
                continue
            first, second = _triangulate(rim, positions), _triangulate(other, positions)
            if first is None or second is None:
                continue
            # A single layer can verify the opposing rim without needing a
            # duplicate of its existing exterior face.
            closures = [(side, cap) for side, cap in ((rim, first), (other, second)) if side.has_depth]
            if any(_has_shared_diagonal(cap, shell_edges, len(labels)) for _, cap in closures):
                continue
            for side, cap in closures:
                caps.append(cap)
                groups.append(np.full(len(cap), side.vertices.min() + 1, dtype=np.intp))
    if caps:
        return np.concatenate(caps), np.concatenate(groups)
    return np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.intp)
