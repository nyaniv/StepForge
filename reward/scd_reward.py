"""
Scaled Chamfer Distance reward for RL training.

Implements paper Equations 1–3 exactly.

Eq. 1 — Bidirectional Chamfer Distance:
    CD(P, Q) = mean_{p∈P} min_{q∈Q} ||p-q||² + mean_{q∈Q} min_{p∈P} ||p-q||²

Eq. 2 — Scaled Chamfer Distance:
    SCD(P, Q) = CD(P_aligned, Q_centered) / scale²
    where scale = RMS distance of GT points from their centroid

Eq. 3 — Piecewise linear reward:
    R_geo(scd) = 1.0          if scd ≤ δ_low  (0.01)
    R_geo(scd) = 0.0          if scd ≥ δ_high (0.50)
    R_geo(scd) = (δ_high - scd) / (δ_high - δ_low)  otherwise
"""

import numpy as np
from scipy.spatial import cKDTree

from reward.alignment import align_point_clouds
from reward.step_to_pointcloud import step_to_pointcloud


# ── Eq. 1: Bidirectional Chamfer Distance ─────────────────────────────────────

def chamfer_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Bidirectional Chamfer Distance (paper Eq. 1)."""
    d_PQ = cKDTree(Q).query(P)[0]
    d_QP = cKDTree(P).query(Q)[0]
    return float(np.mean(d_PQ ** 2) + np.mean(d_QP ** 2))


# ── Eq. 2: Scaled Chamfer Distance ────────────────────────────────────────────

def scaled_chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Scaled Chamfer Distance (paper Eq. 2).
    Robust to translation, rotation, and scale.
    """
    gt_centered = gt - gt.mean(axis=0)
    # Scale factor = RMS distance of GT points from their centroid
    scale_factor = float(np.sqrt(np.mean(np.sum(gt_centered ** 2, axis=1))))
    if scale_factor < 1e-8:
        return float("inf")
    pred_aligned = align_point_clouds(pred, gt)
    cd = chamfer_distance(pred_aligned, gt_centered)
    return cd / (scale_factor ** 2)


# ── Eq. 3: Piecewise linear reward ────────────────────────────────────────────

def r_geo(scd: float,
          delta_low: float = 0.01,
          delta_high: float = 0.50) -> float:
    """Piecewise linear reward R_geo (paper Eq. 3)."""
    if scd <= delta_low:
        return 1.0
    if scd >= delta_high:
        return 0.0
    return (delta_high - scd) / (delta_high - delta_low)


# ── Full reward pipeline ───────────────────────────────────────────────────────

def compute_reward(
    generated_step: str,
    gt_step: str,
    n_points: int = 2048,
    delta_low: float = 0.01,
    delta_high: float = 0.50,
    text2cad_src: str | None = None,
) -> float:
    """
    Full reward pipeline.  Returns 0.0 for any failure — never raises.

    Fast-path: if the generated STEP doesn't have the terminator, skip OCC entirely.
    """
    if "END-ISO-10303-21;" not in generated_step:
        return 0.0

    pred_pc = step_to_pointcloud(generated_step, n_points=n_points,
                                 text2cad_src=text2cad_src)
    gt_pc   = step_to_pointcloud(gt_step, n_points=n_points,
                                 text2cad_src=text2cad_src)

    if pred_pc is None or gt_pc is None:
        return 0.0

    try:
        scd = scaled_chamfer_distance(pred_pc, gt_pc)
    except Exception:
        return 0.0

    if not np.isfinite(scd):
        return 0.0

    return r_geo(scd, delta_low=delta_low, delta_high=delta_high)
