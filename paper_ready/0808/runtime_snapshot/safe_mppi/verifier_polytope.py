"""Trajectory-fitted variable-face verifier polytope for spherical obstacles.

This is the dimension-independent certificate underlying the cloned 2-D
``ieee_compact_polytope_verifier_package``.  For one obstacle with center
``d`` and radius ``r``, it asks whether a face ``a.T x <= m`` exists such that

    a.T p_t <= beta_t m,
    r ||a|| <= a.T d - m,
    ||a|| <= 1,
    m >= m_min,

where ``p_t`` and ``d`` are relative to the start of the candidate window and
``beta_t = 1 - (1-gamma)^t``.  The cloned implementation solves the resulting
angular intervals in 2-D and selects the maximum-margin feasible normal.  Here
the same problem is solved exactly in two or three dimensions by enumerating
its at-most-D active constraints on the unit sphere.

As in the cloned package, artificial obstacles bound the fitted polytope at
the sensing radius.  The 3-D counterpart uses the same 80 triangular
directions as the nominal polytope.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import warnings

import numpy as np

from .geometry import triangular_geometry


@dataclass(frozen=True)
class FaceCertificate:
    feasible: bool
    normal: np.ndarray | None
    margin: float
    worst_slack: float


@dataclass(frozen=True)
class VerifierPolytope:
    A: np.ndarray
    b: np.ndarray
    center: np.ndarray
    margins: np.ndarray
    kinds: tuple[str, ...]
    feasible: bool
    worst_slack: float
    sensing_radius: float


@lru_cache(maxsize=None)
def _active_sets(count: int, size: int) -> np.ndarray:
    return np.asarray(list(combinations(range(count), size)), dtype=np.int64)


def fit_variable_face(trajectory: np.ndarray, obstacle_center: np.ndarray,
                      radius: float, gamma: float, m_min: float = 1.0e-4,
                      tolerance: float = 1.0e-8) -> FaceCertificate:
    """Fit one cloned-verifier face in 2-D or 3-D."""
    trajectory = np.asarray(trajectory, float)
    if trajectory.ndim != 2 or trajectory.shape[1] not in {2, 3}:
        raise ValueError("trajectory must have shape [H+1,2] or [H+1,3]")
    if not 0.0 < float(gamma) <= 1.0 or radius < 0.0 or m_min <= 0.0:
        raise ValueError("require gamma in (0,1], radius >= 0, and m_min > 0")

    centered = trajectory - trajectory[0]
    d = np.asarray(obstacle_center, float).reshape(-1) - trajectory[0]
    if len(d) != trajectory.shape[1]:
        raise ValueError("obstacle center dimension must match the trajectory")
    alpha = (1.0 - float(gamma)) ** np.arange(len(centered), dtype=float)
    beta = 1.0 - alpha

    # At a positive-weight optimum ||a||=1 and m=n.T d-r.  The cloned SOCP
    # therefore reduces to maximizing d.T n on the unit sphere subject to
    # C n >= b.  An optimum is either the radial normal or lies on at most D
    # active linear constraints.
    constraints = [d]
    lower = [float(radius) + float(m_min)]
    for point, beta_t in zip(centered[1:], beta[1:]):
        constraints.append(beta_t * d - point)
        lower.append(beta_t * float(radius))
    C = np.asarray(constraints, float)
    b = np.asarray(lower, float)

    candidates: list[np.ndarray] = []
    distance = float(np.linalg.norm(d))
    if distance > 1.0e-12:
        radial = d / distance
        if np.all(C @ radial >= b - tolerance):
            margin = float(radial @ d - radius)
            h_values = (margin - centered[1:] @ radial) / margin
            worst_slack = float(np.min(h_values - alpha[1:]))
            if worst_slack >= -tolerance:
                return FaceCertificate(True, radial, margin, worst_slack)

    dimension = trajectory.shape[1]
    for size in range(1, min(dimension, len(C)) + 1):
        active = _active_sets(len(C), size)
        rows = C[active]
        gram = rows @ np.swapaxes(rows, 1, 2)
        rhs = b[active]
        nonsingular = np.abs(np.linalg.det(gram)) > 1.0e-12
        if not np.any(nonsingular):
            continue
        dual_rhs = np.full_like(rhs, np.nan)
        dual_rhs[nonsingular] = np.linalg.solve(
            gram[nonsingular], rhs[nonsingular, :, None]
        )[:, :, 0]
        base = np.einsum("mki,mk->mi", rows, dual_rhs)
        base_norm2 = np.einsum("mi,mi->m", base, base)
        remaining = np.sqrt(np.maximum(0.0, 1.0 - base_norm2))

        # Project the max-margin direction d into each active set's nullspace.
        row_dot_d = np.einsum("mki,i->mk", rows, d)
        projection_coeff = np.full_like(row_dot_d, np.nan)
        projection_coeff[nonsingular] = np.linalg.solve(
            gram[nonsingular], row_dot_d[nonsingular, :, None]
        )[:, :, 0]
        tangent = d[None] - np.einsum("mki,mk->mi", rows, projection_coeff)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        regular = nonsingular & (base_norm2 <= 1.0 + tolerance) & (tangent_norm > 1.0e-12)
        points = base.copy()
        points[regular] += (
            remaining[regular, None] * tangent[regular] / tangent_norm[regular, None]
        )
        feasible = (
            regular
            & np.all(points @ C.T >= b[None] - tolerance, axis=1)
            & (np.abs(np.einsum("mi,mi->m", points, points) - 1.0) <= 1.0e-6)
        )
        candidates.extend(points[feasible])

        # If d has no component in the active nullspace, every point on that
        # subsphere has the same objective.  Coordinate projections plus all
        # larger active sets cover its feasible interior/boundary.
        degenerate = np.flatnonzero(
            nonsingular & (base_norm2 <= 1.0 + tolerance) & (tangent_norm <= 1.0e-12)
        )
        eye = np.eye(dimension)
        for index in degenerate:
            if remaining[index] <= 1.0e-8:
                if np.all(C @ base[index] >= b - tolerance):
                    candidates.append(base[index])
                continue
            projector = eye - rows[index].T @ np.linalg.solve(
                gram[index], rows[index]
            )
            for axis in eye:
                direction = projector @ axis
                norm = float(np.linalg.norm(direction))
                if norm <= 1.0e-12:
                    continue
                for sign in (-1.0, 1.0):
                    point = base[index] + sign * remaining[index] * direction / norm
                    if np.all(C @ point >= b - tolerance):
                        candidates.append(point)

    if not candidates:
        return FaceCertificate(False, None, 0.0, -float("inf"))

    best: FaceCertificate | None = None
    for candidate in sorted(candidates, key=lambda value: float(value @ d), reverse=True):
        norm = float(np.linalg.norm(candidate))
        if norm <= 1.0e-12:
            continue
        normal = candidate / norm
        margin = float(normal @ d - radius)
        if margin < m_min - tolerance:
            continue
        h_values = (margin - centered[1:] @ normal) / margin
        worst_slack = float(np.min(h_values - alpha[1:]))
        certificate = FaceCertificate(
            bool(worst_slack >= -tolerance), normal, margin, worst_slack
        )
        if certificate.feasible and (best is None or certificate.margin > best.margin):
            best = certificate
    return best or FaceCertificate(False, None, 0.0, -float("inf"))


@lru_cache(maxsize=None)
def _cvxpy_face_problem(
    dimension: int,
    horizon: int,
    radius_value: float,
    gamma: float,
    m_min: float,
):
    """Compile one reusable parameterized SOCP for a face shape."""
    try:
        import cvxpy as cp
    except ImportError as error:
        raise ImportError(
            "the cvxpy verifier backend requires cvxpy with CLARABEL"
        ) from error
    normal = cp.Variable(dimension)
    margin = cp.Variable()
    points = cp.Parameter((horizon, dimension))
    center = cp.Parameter(dimension)
    beta = 1.0 - (1.0 - float(gamma)) ** np.arange(1, horizon + 1)
    problem = cp.Problem(
        cp.Maximize(margin),
        [
            points @ normal <= cp.multiply(beta, margin),
            float(radius_value) * cp.norm(normal, 2)
            <= center @ normal - margin,
            cp.norm(normal, 2) <= 1.0,
            margin >= float(m_min),
        ],
    )
    return problem, normal, margin, points, center


def fit_variable_face_cvxpy(
    trajectory: np.ndarray,
    obstacle_center: np.ndarray,
    radius: float,
    gamma: float,
    m_min: float = 1.0e-4,
    tolerance: float = 1.0e-8,
    fallback_to_analytic: bool = True,
) -> FaceCertificate:
    """Solve the same variable-face problem with a CVXPY/CLARABEL hybrid.

    The exact radial certificate is retained because it is both the SOCP
    optimum and substantially cheaper than invoking a generic solver.  A
    CLARABEL failure or numerically ambiguous result falls back to the existing
    active-set solver by default, preserving fail-closed expansion semantics.
    """
    trajectory = np.asarray(trajectory, float)
    if trajectory.ndim != 2 or trajectory.shape[1] not in {2, 3}:
        raise ValueError("trajectory must have shape [H+1,2] or [H+1,3]")
    if not 0.0 < float(gamma) <= 1.0 or radius < 0.0 or m_min <= 0.0:
        raise ValueError("require gamma in (0,1], radius >= 0, and m_min > 0")
    centered = trajectory - trajectory[0]
    d = np.asarray(obstacle_center, float).reshape(-1) - trajectory[0]
    if len(d) != trajectory.shape[1]:
        raise ValueError("obstacle center dimension must match the trajectory")
    alpha = (1.0 - float(gamma)) ** np.arange(len(centered), dtype=float)
    distance = float(np.linalg.norm(d))
    if distance > 1.0e-12:
        radial = d / distance
        radial_margin = float(radial @ d - radius)
        if radial_margin >= m_min - tolerance:
            h_values = (
                radial_margin - centered[1:] @ radial
            ) / radial_margin
            radial_slack = float(np.min(h_values - alpha[1:]))
            if radial_slack >= -tolerance:
                return FaceCertificate(
                    True, radial, radial_margin, radial_slack
                )
    horizon = len(centered) - 1
    (
        problem,
        normal_variable,
        _margin_variable,
        points_parameter,
        center_parameter,
    ) = _cvxpy_face_problem(
        trajectory.shape[1],
        horizon,
        float(radius),
        float(gamma),
        float(m_min),
    )
    points_parameter.value = centered[1:]
    center_parameter.value = d
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Solution may be inaccurate.*",
                category=UserWarning,
            )
            problem.solve(
                solver="CLARABEL",
                warm_start=True,
                verbose=False,
                tol_gap_abs=1.0e-8,
                tol_gap_rel=1.0e-8,
                tol_feas=1.0e-8,
                max_iter=200,
            )
    except Exception as error:
        if fallback_to_analytic:
            return fit_variable_face(
                trajectory, obstacle_center, radius, gamma,
                m_min=m_min, tolerance=min(tolerance, 1.0e-8),
            )
        raise RuntimeError("CVXPY CLARABEL face solve failed") from error
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        if fallback_to_analytic:
            return fit_variable_face(
                trajectory, obstacle_center, radius, gamma,
                m_min=m_min, tolerance=min(tolerance, 1.0e-8),
            )
        return FaceCertificate(False, None, 0.0, -float("inf"))
    value = np.asarray(normal_variable.value, float).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-10 or not np.isfinite(value).all():
        if fallback_to_analytic:
            return fit_variable_face(
                trajectory, obstacle_center, radius, gamma,
                m_min=m_min, tolerance=min(tolerance, 1.0e-8),
            )
        return FaceCertificate(False, None, 0.0, -float("inf"))
    normal = value / norm
    margin = float(normal @ d - radius)
    if margin < m_min - tolerance:
        if fallback_to_analytic:
            return fit_variable_face(
                trajectory, obstacle_center, radius, gamma,
                m_min=m_min, tolerance=min(tolerance, 1.0e-8),
            )
        return FaceCertificate(False, None, 0.0, -float("inf"))
    h_values = (margin - centered[1:] @ normal) / margin
    alpha = (1.0 - float(gamma)) ** np.arange(1, horizon + 1)
    worst_slack = float(np.min(h_values - alpha))
    if (not np.isfinite(worst_slack) or worst_slack < -tolerance) and fallback_to_analytic:
        return fit_variable_face(
            trajectory, obstacle_center, radius, gamma,
            m_min=m_min, tolerance=min(tolerance, 1.0e-8),
        )
    feasible = bool(np.isfinite(worst_slack) and worst_slack >= -tolerance)
    return FaceCertificate(feasible, normal if feasible else None, margin,
                           worst_slack)


def fit_verifier_polytope(positions: np.ndarray, spheres: np.ndarray,
                          cylinders: np.ndarray, gamma: float,
                          sensing_range: float, m_min: float = 1.0e-4,
                          radius_padding: float = 1.3,
                          artificial_radius: float = 0.16,
                          face_solver: str = "analytic") -> VerifierPolytope:
    """Fit real-obstacle and 80 artificial bounding faces for one window."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape [H+1,3]")
    if artificial_radius <= 0.0:
        raise ValueError("artificial_radius must be positive")
    if face_solver not in {"analytic", "cvxpy"}:
        raise ValueError("face_solver must be analytic or cvxpy")
    fit_face = (
        fit_variable_face
        if face_solver == "analytic" else fit_variable_face_cvxpy
    )
    spheres = np.asarray(spheres, float).reshape(-1, 4)
    cylinders = np.asarray(cylinders, float).reshape(-1, 3)
    relative = positions - positions[0]
    effective_range = max(
        float(sensing_range),
        float(radius_padding) * float(np.linalg.norm(relative, axis=1).max()),
    )

    normals: list[np.ndarray] = []
    margins: list[float] = []
    kinds: list[str] = []
    slacks: list[float] = []
    feasible = True

    def append(certificate: FaceCertificate, kind: str) -> None:
        nonlocal feasible
        if not certificate.feasible or certificate.normal is None:
            feasible = False
            slacks.append(float(certificate.worst_slack))
            return
        normals.append(np.asarray(certificate.normal, float))
        margins.append(float(certificate.margin))
        kinds.append(kind)
        slacks.append(float(certificate.worst_slack))

    for sphere in spheres:
        if np.linalg.norm(sphere[:3] - positions[0]) - sphere[3] > effective_range:
            continue
        append(
            fit_face(
                positions, sphere[:3], float(sphere[3]), gamma, m_min=m_min
            ),
            "real_sphere",
        )

    # Infinite vertical cylinders reduce exactly to the cloned 2-D disk block.
    for cylinder in cylinders:
        if np.linalg.norm(cylinder[:2] - positions[0, :2]) - cylinder[2] > effective_range:
            continue
        certificate = fit_face(
            positions[:, :2], cylinder[:2], float(cylinder[2]), gamma, m_min=m_min
        )
        if certificate.normal is not None:
            certificate = FaceCertificate(
                certificate.feasible,
                np.r_[certificate.normal, 0.0],
                certificate.margin,
                certificate.worst_slack,
            )
        append(certificate, "real_cylinder")

    # The 2-D clone places a small artificial disk just outside each desired
    # radial face. In 3-D, use the 80 icosphere-face directions and their exact
    # inscribed offsets. A radial solution then has margin R_eff*offset, exactly
    # matching the nominal 80-face sensing envelope.
    _, _, bounding_normals, unit_offsets = triangular_geometry()
    alpha = (1.0 - float(gamma)) ** np.arange(len(positions), dtype=float)
    for normal, unit_offset in zip(bounding_normals, unit_offsets):
        radial_margin = effective_range * float(unit_offset)
        radial_h = (
            radial_margin - relative[1:] @ normal
        ) / radial_margin
        radial_slack = float(np.min(radial_h - alpha[1:]))
        if radial_slack >= -1.0e-8:
            append(
                FaceCertificate(
                    True, normal.copy(), radial_margin, radial_slack
                ),
                "artificial",
            )
            continue
        obstacle_center = (
            positions[0]
            + (radial_margin + artificial_radius) * normal
        )
        append(
            fit_face(
                positions, obstacle_center, artificial_radius, gamma, m_min=m_min
            ),
            "artificial",
        )

    A = np.asarray(normals, float).reshape(-1, 3)
    face_margins = np.asarray(margins, float)
    b = A @ positions[0] + face_margins
    worst_slack = float(min(slacks)) if slacks else -float("inf")
    return VerifierPolytope(
        A=A,
        b=b,
        center=positions[0].copy(),
        margins=face_margins,
        kinds=tuple(kinds),
        feasible=bool(feasible),
        worst_slack=worst_slack,
        sensing_radius=float(effective_range),
    )


def certify_window(positions: np.ndarray, spheres: np.ndarray,
                   cylinders: np.ndarray, gamma: float, sensing_range: float,
                   m_min: float = 1.0e-4,
                   radius_padding: float = 1.3,
                   artificial_radius: float = 0.16,
                   face_solver: str = "analytic") -> tuple[bool, float]:
    """Certify one full window with the bounded fitted verifier polytope."""
    polytope = fit_verifier_polytope(
        positions, spheres, cylinders, gamma, sensing_range,
        m_min=m_min,
        radius_padding=radius_padding,
        artificial_radius=artificial_radius,
        face_solver=face_solver,
    )
    return polytope.feasible, polytope.worst_slack


def certify_single_sphere_affine(
    positions: np.ndarray,
    spheres: np.ndarray,
    cylinders: np.ndarray,
    gamma: float,
    sensing_range: float,
    m_min: float = 1.0e-4,
    face_solver: str = "analytic",
) -> tuple[bool, float]:
    """Certify against one sphere using only its fitted affine barrier.

    This is exactly the real-obstacle block used by :func:`certify_window`:
    it finds one face whose contracted level sets contain every plan knot and
    exclude the sphere.  It deliberately omits the 80 artificial faces that
    bound the sensing envelope.  The caller must still perform task-space and
    collision checks; consequently this mode is a task-specific speed option,
    not a claim of equivalence to the bounded GREEN polytope.

    ``sensing_range`` remains in the signature so verifier call sites can swap
    modes without changing their scientific inputs.  The sole obstacle is
    always considered, including before it enters the nominal sensing range.
    """
    del sensing_range
    spheres = np.asarray(spheres, float).reshape(-1, 4)
    cylinders = np.asarray(cylinders, float).reshape(-1, 3)
    if len(spheres) != 1 or len(cylinders) != 0:
        raise ValueError(
            "single_sphere_affine requires exactly one sphere and no cylinders"
        )
    if face_solver not in {"analytic", "cvxpy"}:
        raise ValueError("face_solver must be analytic or cvxpy")
    fit_face = (
        fit_variable_face
        if face_solver == "analytic" else fit_variable_face_cvxpy
    )
    certificate = fit_face(
        np.asarray(positions, float),
        spheres[0, :3],
        float(spheres[0, 3]),
        float(gamma),
        m_min=m_min,
    )
    return certificate.feasible, float(certificate.worst_slack)
