import numpy as np

from rebot_grasp.core.types import GraspCandidate, GraspPlan


def filter_candidates(candidates: list[GraspCandidate], *, max_opening_m: float, opening_margin_m: float, min_score: float) -> list[GraspCandidate]:
    if not np.isfinite([max_opening_m, opening_margin_m, min_score]).all() or max_opening_m <= 0 or opening_margin_m < 0:
        raise ValueError("Invalid candidate filtering limits")
    return sorted((g for g in candidates if g.score >= min_score and g.width_m + opening_margin_m <= max_opening_m), key=lambda g: g.score, reverse=True)


def filter_workspace(plans: list[GraspPlan], lower, upper) -> list[GraspPlan]:
    lower, upper = np.asarray(lower), np.asarray(upper)
    accepted = []
    for plan in plans:
        positions = np.array([
            plan.T_base_pregrasp[:3, 3], plan.T_base_grasp[:3, 3], plan.T_base_retreat[:3, 3],
        ])
        if np.all((lower <= positions) & (positions <= upper)):
            accepted.append(plan)
    return accepted
