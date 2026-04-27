"""
Scaled Chamfer Distance reward for RL training.

Implements paper Equations 1-3 exactly.

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

import atexit
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from reward.alignment import align_point_clouds


# API-3: collapse the 6 reward-shape params that always travel together from
# cfg.rl.reward into one frozen dataclass. Pickles cleanly for mp.Process.
@dataclass(frozen=True)
class RewardConfig:
    n_points: int = 2000
    delta_low: float = 0.01
    delta_high: float = 0.50
    bidirectional: bool = True
    scale_prenorm: bool = True
    deflection: float | None = None

# Minimum unique points before running Open3D registration. Barycentric
# sampling produces continuous points (rarely collide), so this no longer
# guards "few mesh vertices". It still guards a single near-zero-area face
# where float64 quantization collapses the cloud — keep, but the real
# reward-hacking guard is _MIN_TRIANGLES below.
_MIN_UNIQUE_POINTS = 10
_MIN_TRIANGLES = 4


# ── Subprocess-isolated OCC tessellation ─────────────────────────────────────
# Why: OCC's STEP parser segfaults (SIGSEGV) on some malformed completions
# that pass our textual checks. A native segfault kills the rank — try/except
# can't catch it. We isolate step_to_pointcloud in a worker pool so worker
# death is recoverable: pool respawns the worker, we return (None, 0).
#
# Why spawn (not fork): the prior mp.Process used the default fork() context,
# which inherits the parent's CUDA/OpenMP state and deadlocks indefinitely.
# spawn starts a fresh interpreter — no CUDA inheritance, no deadlock.
# Workers persist across calls, so spawn's startup cost is paid once per pool.
_OCC_POOL: ProcessPoolExecutor | None = None
_OCC_POOL_LOCK = threading.Lock()
_OCC_POOL_WORKERS = 2
_OCC_TIMEOUT_S = 30.0


def _occ_worker(step_content: str, n_points: int,
                text2cad_src: str | None,
                deflection: float | None) -> tuple:
    """Subprocess entrypoint. Top-level so it pickles for spawn."""
    from reward.step_to_pointcloud import step_to_pointcloud as _s2p
    return _s2p(step_content, n_points=n_points,
                text2cad_src=text2cad_src,
                return_triangle_count=True,
                deflection=deflection)


def _get_occ_pool() -> ProcessPoolExecutor:
    global _OCC_POOL
    with _OCC_POOL_LOCK:
        if _OCC_POOL is None:
            _OCC_POOL = ProcessPoolExecutor(
                max_workers=_OCC_POOL_WORKERS,
                mp_context=mp.get_context("spawn"),
            )
        return _OCC_POOL


def _shutdown_occ_pool() -> None:
    global _OCC_POOL
    with _OCC_POOL_LOCK:
        if _OCC_POOL is not None:
            try:
                _OCC_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _OCC_POOL = None


atexit.register(_shutdown_occ_pool)


def _safe_step_to_pointcloud(step_content: str, *, n_points: int,
                              text2cad_src: str | None,
                              deflection: float | None,
                              timeout: float = _OCC_TIMEOUT_S) -> tuple:
    """
    Subprocess-isolated step_to_pointcloud. Returns (pc | None, n_triangles).
    Survives OCC native segfaults: pool respawns dead workers automatically;
    BrokenProcessPool poisons the global so the next caller rebuilds it.
    """
    global _OCC_POOL
    pool = _get_occ_pool()
    try:
        fut = pool.submit(_occ_worker, step_content, n_points,
                          text2cad_src, deflection)
        return fut.result(timeout=timeout)
    except BrokenProcessPool:
        with _OCC_POOL_LOCK:
            _OCC_POOL = None
        return (None, 0)
    except FuturesTimeoutError:
        return (None, 0)
    except Exception:
        return (None, 0)


# ── Eq. 1: Bidirectional Chamfer Distance ─────────────────────────────────────

def chamfer_distance(P: np.ndarray, Q: np.ndarray, *, bidirectional: bool = True) -> float:
    """
    Chamfer Distance.

    A1: bidirectional=True implements paper Eq. (1) exactly.
    bidirectional=False matches the OFFICIAL eval code, which calls
    chamferdist.ChamferDistance() without bidirectional=True (the library
    defaults to forward-only mean(d_PQ²)). Paper Tables 1/4 were almost
    certainly produced by the official code, so use False to compare against
    reported MSCD numbers and True to compare against the paper's stated method.
    """
    d_PQ = cKDTree(Q).query(P)[0]
    if not bidirectional:
        return float(np.mean(d_PQ ** 2))
    d_QP = cKDTree(P).query(Q)[0]
    return float(np.mean(d_PQ ** 2) + np.mean(d_QP ** 2))


# ── Eq. 2: Scaled Chamfer Distance ────────────────────────────────────────────

def scaled_chamfer_distance(pred: np.ndarray, gt: np.ndarray, *,
                            bidirectional: bool = True,
                            scale_prenorm: bool = True) -> float:
    """
    Scaled Chamfer Distance (paper Eq. 2).
    Robust to translation, rotation, and scale.

    NOTE: Asymmetric — scale_factor and alignment target both derive from gt.
    Do not swap arguments. SCD(pred, gt) != SCD(gt, pred).

    Paper §3.3: "the scale factor is defined as the root mean square distance
    of ground-truth points from its centroid." Implementation matches.
    """
    # float64 throughout: float32 centering on 1e5-scale CAD coords leaves ~1e-2
    # absolute error, which is the same magnitude as the paper's δ_low=0.01 threshold.
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)

    gt_centered = gt - gt.mean(axis=0)
    # Scale factor = RMS distance of GT points from their centroid
    scale_factor = float(np.sqrt(np.mean(np.sum(gt_centered ** 2, axis=1))))
    if scale_factor < 1e-8:
        return float("inf")
    # C2: Open3D's estimate_normals/FPFH/RANSAC/ICP can raise on geometry that
    # parsed and tessellated fine. NaN means "alignment infrastructure failed"
    # — the worker returns it as scd_nonfinite (NaN-masked), not reward=0.
    try:
        pred_aligned = align_point_clouds(pred, gt_centered, scale_prenorm=scale_prenorm)
    except Exception:
        return float("nan")
    cd = chamfer_distance(pred_aligned, gt_centered, bidirectional=bidirectional)
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


# ── Parse reward (intermediate signal: does OCP accept the generated STEP?) ────

def compute_parse_reward(generated_step: str, text2cad_src: str | None = None,
                         reward_value: float = 0.3) -> float:
    """
    Returns reward_value if the generated STEP parses + tessellates successfully,
    0.0 otherwise. OCC tessellation is run in a spawn-context worker pool so
    a native segfault (seen mid-run on malformed completions) cannot kill the
    training rank.
    """
    if "END-ISO-10303-21;" not in generated_step:
        return 0.0
    pc, n_tris = _safe_step_to_pointcloud(
        generated_step, n_points=64,
        text2cad_src=text2cad_src, deflection=None,
    )
    ok = (pc is not None
          and n_tris >= _MIN_TRIANGLES
          and len(np.unique(pc, axis=0)) >= _MIN_UNIQUE_POINTS)
    return reward_value if ok else 0.0


# ── GT point cloud cache (in-process, shared across the 8 GRPO generations) ───

# In-process LRU cache of GT point clouds. All 8 GRPO generations for a prompt
# share the same GT — without this cache the OCC tessellation runs 8× per prompt
# per step. Keyed by SHA256(n_points|deflection|gt_step). Capped to bound memory.
_GT_PC_CACHE: dict[str, tuple] = {}
_GT_PC_CACHE_MAX = 512
# B1: GRPO sends ≥num_generations threads with the same gt_step every step.
# Unsynchronized eviction lets two threads pop(next(iter(dict))) the same key →
# KeyError → crash via f.result(). Lock the read-modify-write.
_GT_PC_CACHE_LOCK = threading.Lock()


def _gt_cache_key(gt_step: str, rcfg: RewardConfig) -> str:
    # CK-1: cached point cloud depends on (n_points, deflection); include them
    # so a process that calls compute_reward with different params doesn't
    # silently reuse a stale tessellation.
    import hashlib
    prefix = f"{rcfg.n_points}|{rcfg.deflection!r}|".encode()
    return hashlib.sha256(prefix + gt_step.encode(errors="replace")).hexdigest()


# ── Full reward pipeline ───────────────────────────────────────────────────────

def compute_reward(
    generated_step: str,
    gt_step: str,
    *,
    rcfg: RewardConfig = RewardConfig(),
    text2cad_src: str | None = None,
    verbose: bool = False,
) -> tuple[float, float, str, int]:
    """
    Full reward pipeline. Never raises.

    Returns (reward, raw_scd, fail_stage, n_triangles):
      reward     -- r_geo(scd) ∈ [0,1], or 0.0 on pred failure, or NaN on GT failure
      raw_scd    -- pre-clamp Chamfer distance (NaN if not computed)
      fail_stage -- 'ok' | 'no_terminator' | 'pred_parse' | 'gt_parse'
                    | 'pred_degenerate' | 'gt_degenerate' | 'scd_nonfinite' | 'exception'
      n_triangles-- mesh triangle count for the prediction (0 if parse failed)

    OCC tessellation runs in a spawn-context worker pool; native segfaults in
    the OCC C++ parser cannot kill the training rank.
    """
    if "END-ISO-10303-21;" not in generated_step:
        return (0.0, float("nan"), "no_terminator", 0)

    try:
        # Pred point cloud (subprocess-isolated)
        pred_pc, pred_tris = _safe_step_to_pointcloud(
            generated_step, n_points=rcfg.n_points,
            text2cad_src=text2cad_src, deflection=rcfg.deflection)

        if pred_pc is None:
            return (0.0, float("nan"), "pred_parse", 0)
        if pred_tris < _MIN_TRIANGLES:
            return (0.0, float("nan"), "pred_degenerate", pred_tris)
        if len(np.unique(pred_pc, axis=0)) < _MIN_UNIQUE_POINTS:
            return (0.0, float("nan"), "pred_degenerate", pred_tris)

        # GT point cloud — use cache to avoid recomputing for all 8 GRPO generations
        cache_key = _gt_cache_key(gt_step, rcfg)
        cached = _GT_PC_CACHE.get(cache_key)
        if cached is not None:
            gt_pc, gt_tris = cached
        else:
            gt_pc, gt_tris = _safe_step_to_pointcloud(
                gt_step, n_points=rcfg.n_points,
                text2cad_src=text2cad_src, deflection=rcfg.deflection)
            if gt_pc is not None and gt_tris >= _MIN_TRIANGLES:
                with _GT_PC_CACHE_LOCK:
                    if cache_key not in _GT_PC_CACHE:
                        if len(_GT_PC_CACHE) >= _GT_PC_CACHE_MAX:
                            evict = next(iter(_GT_PC_CACHE), None)
                            if evict is not None:
                                _GT_PC_CACHE.pop(evict, None)
                        _GT_PC_CACHE[cache_key] = (gt_pc, gt_tris)

        if gt_pc is None:
            return (float("nan"), float("nan"), "gt_parse", pred_tris)
        if gt_tris < _MIN_TRIANGLES:
            return (float("nan"), float("nan"), "gt_degenerate", pred_tris)
        if len(np.unique(gt_pc, axis=0)) < _MIN_UNIQUE_POINTS:
            return (float("nan"), float("nan"), "gt_degenerate", pred_tris)

        scd = scaled_chamfer_distance(pred_pc, gt_pc,
                                      bidirectional=rcfg.bidirectional,
                                      scale_prenorm=rcfg.scale_prenorm)
        if verbose:
            print(f"[compute_reward] SCD={scd:.4f}")
        if not np.isfinite(scd):
            return (float("nan"), float("nan"), "scd_nonfinite", pred_tris)

        reward = r_geo(scd, delta_low=rcfg.delta_low, delta_high=rcfg.delta_high)
        if verbose:
            print(f"[compute_reward] reward={reward:.4f}")
        return (reward, float(scd), "ok", pred_tris)

    except Exception as e:
        if verbose:
            print(f"[compute_reward] Exception: {e!r}")
        return (0.0, float("nan"), "exception", 0)
