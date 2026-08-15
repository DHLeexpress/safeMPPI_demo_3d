"""The only nominal geometry used here: 80 triangular faces, tangent obstacles, zero margin."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


@dataclass(frozen=True)
class NominalPolytope:
    A: np.ndarray
    b: np.ndarray
    center: np.ndarray
    margins: np.ndarray


@lru_cache(maxsize=1)
def triangular_geometry():
    """Subdivision-1 icosphere: 42 unit vertices and 80 ordered triangular face normals."""
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array([
        [-1, +phi, 0], [+1, +phi, 0], [-1, -phi, 0], [+1, -phi, 0],
        [0, -1, +phi], [0, +1, +phi], [0, -1, -phi], [0, +1, -phi],
        [+phi, 0, -1], [+phi, 0, +1], [-phi, 0, -1], [-phi, 0, +1],
    ], float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = ConvexHull(vertices).simplices.astype(int)
    vertex_list = [v.copy() for v in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(i, j):
        key = tuple(sorted((int(i), int(j))))
        if key not in midpoint_cache:
            p = vertex_list[key[0]] + vertex_list[key[1]]
            p /= np.linalg.norm(p)
            midpoint_cache[key] = len(vertex_list)
            vertex_list.append(p)
        return midpoint_cache[key]

    subdivided = []
    for i, j, k in faces:
        a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
        subdivided.extend(((i, a, c), (j, b, a), (k, c, b), (a, b, c)))
    vertices = np.asarray(vertex_list, float)
    faces = np.asarray(subdivided, int)
    oriented, normals, offsets = [], [], []
    for face in faces:
        i, j, k = map(int, face)
        p, q, r = vertices[i], vertices[j], vertices[k]
        normal = np.cross(q - p, r - p)
        if normal @ (p + q + r) < 0:
            j, k = k, j
            q, r = r, q
            normal = -normal
        normal /= np.linalg.norm(normal)
        oriented.append((i, j, k))
        normals.append(normal)
        offsets.append(float(normal @ p))
    faces = np.asarray(oriented, int)
    normals = np.asarray(normals, float)
    offsets = np.asarray(offsets, float)
    azimuth = np.mod(np.arctan2(normals[:, 1], normals[:, 0]), 2 * np.pi)
    order = np.lexsort((azimuth, -normals[:, 2]))
    return vertices, faces[order], normals[order], offsets[order]


def build_nominal_polytope(center, spheres, cylinders, bounds, *, sensing_range=2.0,
                           obstacle_margin=0.0, max_obstacles=16):
    """Build P={x:A x<=b}; every obstacle face is tangent to its raw radius when margin=0."""
    if float(obstacle_margin) != 0.0:
        raise ValueError("this package fixes obstacle_margin=0")
    center = np.asarray(center, float).reshape(3)
    spheres = np.asarray(spheres, float).reshape(-1, 4)
    cylinders = np.asarray(cylinders, float).reshape(-1, 3)
    bounds = np.asarray(bounds, float).reshape(3, 2)
    _, _, normals, unit_offsets = triangular_geometry()
    A_rows = [normal.copy() for normal in normals]
    b_rows = [float(normal @ center + sensing_range * offset)
              for normal, offset in zip(normals, unit_offsets)]

    candidates = []
    for sphere in spheres:
        delta = sphere[:3] - center
        distance = float(np.linalg.norm(delta))
        if distance > 1e-9:
            normal = delta / distance
            candidates.append((distance - sphere[3], normal))
    for cylinder in cylinders:
        delta = cylinder[:2] - center[:2]
        distance = float(np.linalg.norm(delta))
        if distance > 1e-9:
            normal = np.array([delta[0] / distance, delta[1] / distance, 0.0])
            candidates.append((distance - cylinder[2], normal))
    for clearance, normal in sorted(candidates, key=lambda item: item[0])[:max_obstacles]:
        if clearance <= sensing_range:
            A_rows.append(normal)
            b_rows.append(float(normal @ center + clearance))

    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 1.0
        A_rows.extend((direction.copy(), -direction.copy()))
        b_rows.extend((float(bounds[axis, 1]), float(-bounds[axis, 0])))

    A = np.asarray(A_rows, float)
    b = np.asarray(b_rows, float)
    margins = b - A @ center
    return NominalPolytope(A=A, b=b, center=center, margins=margins)


def hp_values(polytope: NominalPolytope, points, clamp_margin=1e-3):
    points = np.asarray(points, float).reshape(-1, 3)
    margins = np.maximum(polytope.margins, float(clamp_margin))
    return ((polytope.b[None] - points @ polytope.A.T) / margins[None]).min(axis=1)


def polytope_vertices(polytope: NominalPolytope):
    """Return vertices, or None when center is not strict interior (e.g. the requested corner start)."""
    if np.any(polytope.margins <= 1e-9):
        return None
    try:
        hs = np.hstack([polytope.A, -polytope.b[:, None]])
        vertices = HalfspaceIntersection(hs, polytope.center).intersections
        return vertices if len(vertices) >= 4 else None
    except Exception:
        return None


def polytope_centroid(polytope: NominalPolytope):
    vertices = polytope_vertices(polytope)
    if vertices is None:
        return polytope.center.copy()
    try:
        hull = ConvexHull(vertices)
        anchor = vertices.mean(axis=0)
        volume_sum = 0.0
        weighted = np.zeros(3)
        for simplex in hull.simplices:
            a, b, c = vertices[simplex]
            volume = abs(np.dot(a - anchor, np.cross(b - anchor, c - anchor))) / 6.0
            volume_sum += volume
            weighted += volume * (anchor + a + b + c) / 4.0
        return weighted / volume_sum if volume_sum > 1e-12 else polytope.center.copy()
    except Exception:
        return polytope.center.copy()


def hull_edges(vertices):
    hull = ConvexHull(np.asarray(vertices, float))
    edges = sorted({tuple(sorted((int(face[j]), int(face[(j + 1) % 3]))))
                    for face in hull.simplices for j in range(3)})
    return hull, np.asarray(edges, int)
