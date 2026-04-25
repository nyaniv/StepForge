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

import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from reward.alignment import align_point_clouds
from reward.step_to_pointcloud import step_to_pointcloud


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

def _parse_worker(queue: mp.Queue, generated_step: str,
                  text2cad_src: str | None) -> None:
    """Return 1 if step_to_pointcloud succeeds, else 0."""
    try:
        pc, n_tris = step_to_pointcloud(generated_step, n_points=64,
                                        text2cad_src=text2cad_src,
                                        return_triangle_count=True)
        ok = (pc is not None
              and n_tris >= _MIN_TRIANGLES
              and len(np.unique(pc, axis=0)) >= _MIN_UNIQUE_POINTS)
        queue.put(1 if ok else 0)
    except Exception:
        queue.put(0)


def compute_parse_reward(generated_step: str, text2cad_src: str | None = None,
                         reward_value: float = 0.3) -> float:
    """
    Returns reward_value if the generated STEP parses successfully in OCP,
    0.0 otherwise.  Runs in a subprocess to isolate segfaults.
    """
    if "END-ISO-10303-21;" not in generated_step:
        return 0.0
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_parse_worker, args=(queue, generated_step, text2cad_src))
    proc.start()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
    if proc.exitcode != 0:
        queue.close()
        return 0.0
    try:
        # Use timeout instead of get_nowait: the OS pipe feeder thread may not
        # have flushed the result yet when proc.join() returns, causing a
        # spurious Empty exception even though the subprocess put() succeeded.
        result = queue.get(timeout=5)
        return reward_value if result == 1 else 0.0
    except Exception:
        return 0.0
    finally:
        queue.close()


# ── Subprocess worker (isolates Open3D segfaults) ─────────────────────────────

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


def _scd_worker(queue: mp.Queue, generated_step: str, gt_step: str,
                rcfg: RewardConfig, text2cad_src: str | None,
                verbose: bool = False,
                gt_pc_precomputed: np.ndarray | None = None,
                gt_tris_precomputed: int | None = None) -> None:
    """
    Run the full reward pipeline (STEP parsing + alignment + Chamfer) in a
    child process so any C++ segfault (OCP tessellation or Open3D) cannot
    kill the training process.
    """
    # Queue payload: (reward, raw_scd, fail_stage, n_tris). raw_scd lets the
    # parent process see whether the model is at SCD=0.55 (one step from
    # gradient — lower delta_high) vs SCD=50.0 (broken). fail_stage lets
    # tensorboard show parse-rate climbing from 5%→80% as a separate scalar.
    try:
        # Signal "worker reached the GT phase" so the parent can attribute
        # a later segfault to GT-side processing instead of pred-side.
        queue.put(("_phase", "pred"))

        if "END-ISO-10303-21;" not in generated_step:
            if verbose:
                print("[scd_worker] FAIL: no terminator")
            queue.put((0.0, float("nan"), "no_terminator", 0))
            return

        pred_pc, pred_tris = step_to_pointcloud(generated_step, n_points=rcfg.n_points,
                                                text2cad_src=text2cad_src, verbose=verbose,
                                                return_triangle_count=True,
                                                deflection=rcfg.deflection)

        queue.put(("_phase", "gt"))
        if gt_pc_precomputed is not None:
            gt_pc, gt_tris = gt_pc_precomputed, gt_tris_precomputed
        else:
            gt_pc, gt_tris = step_to_pointcloud(gt_step, n_points=rcfg.n_points,
                                                text2cad_src=text2cad_src, verbose=verbose,
                                                return_triangle_count=True,
                                                deflection=rcfg.deflection)

        if pred_pc is None:
            if verbose:
                print("[scd_worker] FAIL: pred_pc is None (generated STEP failed to parse)")
            queue.put((0.0, float("nan"), "pred_parse", 0))
            return
        if gt_pc is None:
            if verbose:
                print("[scd_worker] FAIL: gt_pc is None (GT STEP failed to parse)")
            queue.put((float("nan"), float("nan"), "gt_parse", pred_tris))
            return

        if pred_tris < _MIN_TRIANGLES:
            if verbose:
                print(f"[scd_worker] FAIL: pred has only {pred_tris} triangles — likely reward-hacking sphere/plane")
            queue.put((0.0, float("nan"), "pred_degenerate", pred_tris))
            return
        if gt_tris < _MIN_TRIANGLES:
            queue.put((float("nan"), float("nan"), "gt_degenerate", pred_tris))
            return

        pred_unique = len(np.unique(pred_pc, axis=0))
        gt_unique   = len(np.unique(gt_pc, axis=0))
        if pred_unique < _MIN_UNIQUE_POINTS:
            if verbose:
                print(f"[scd_worker] FAIL: pred unique_pts={pred_unique} < {_MIN_UNIQUE_POINTS}")
            queue.put((0.0, float("nan"), "pred_degenerate", pred_tris))
            return
        if gt_unique < _MIN_UNIQUE_POINTS:
            if verbose:
                print(f"[scd_worker] FAIL: gt unique_pts={gt_unique} < {_MIN_UNIQUE_POINTS}")
            queue.put((float("nan"), float("nan"), "gt_degenerate", pred_tris))
            return

        scd = scaled_chamfer_distance(pred_pc, gt_pc, bidirectional=rcfg.bidirectional,
                                      scale_prenorm=rcfg.scale_prenorm)
        if verbose:
            print(f"[scd_worker] SCD={scd:.4f}, finite={np.isfinite(scd)}")
        if not np.isfinite(scd):
            # C4: scd_nonfinite is a GT-side or alignment-infrastructure
            # condition (scale_factor≈0 or align raised) — NaN-mask, not 0.
            queue.put((float("nan"), float("nan"), "scd_nonfinite", pred_tris))
            return
        reward = r_geo(scd, delta_low=rcfg.delta_low, delta_high=rcfg.delta_high)
        if verbose:
            print(f"[scd_worker] reward={reward:.4f}")
        queue.put((reward, float(scd), "ok", pred_tris))
    except Exception as e:
        if verbose:
            print(f"[scd_worker] Exception: {e!r}")
        queue.put((0.0, float("nan"), "exception", 0))


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
                    | 'pred_degenerate' | 'gt_degenerate' | 'scd_nonfinite'
                    | 'exception' | 'timeout' | 'segfault' | 'segfault_gt'
                    | 'spawn_fail'
      n_triangles-- mesh triangle count for the prediction (0 if parse failed)

    Everything (STEP parsing, tessellation, alignment, Chamfer) runs in a
    subprocess so any C++ segfault cannot kill the training process.
    """
    # Fast-path: skip subprocess overhead if terminator is absent
    if "END-ISO-10303-21;" not in generated_step:
        return (0.0, float("nan"), "no_terminator", 0)

    # GT point-cloud cache: all generations for a prompt share the same GT.
    # spawn copies the parent's memory, so the precomputed array reaches the child.
    cache_key = _gt_cache_key(gt_step, rcfg)
    cached = _GT_PC_CACHE.get(cache_key)
    if cached is None:
        gt_pc_pre, gt_tris_pre = None, None
    else:
        gt_pc_pre, gt_tris_pre = cached

    # B2: mp.Queue() and proc.start() can raise OSError on fd/process-slot
    # exhaustion. With 16 ThreadPool workers × DDP ranks × 80 steps that's
    # plausible mid-run; honour the "Never raises" docstring.
    try:
        queue: mp.Queue = mp.Queue()
        # API-2: kwargs= instead of a positional args tuple — immune to reorder.
        proc = mp.Process(
            target=_scd_worker,
            kwargs=dict(
                queue=queue, generated_step=generated_step, gt_step=gt_step,
                rcfg=rcfg, text2cad_src=text2cad_src, verbose=verbose,
                gt_pc_precomputed=gt_pc_pre, gt_tris_precomputed=gt_tris_pre,
            ),
        )
        proc.start()
    except OSError as e:
        if verbose:
            print(f"[compute_reward] subprocess spawn failed: {e!r}")
        return (float("nan"), float("nan"), "spawn_fail", 0)
    proc.join(timeout=60)  # 60 s hard cap per reward call

    # Drain phase markers + result. Phase tells us where a segfault hit.
    last_phase = "pred"
    result = None
    while True:
        try:
            item = queue.get_nowait()
        except Exception:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "_phase":
            last_phase = item[1]
        else:
            result = item

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        queue.close()
        # Best-effort sweep of temp files orphaned by the killed worker.
        import glob as _glob, shutil as _shutil
        for _d in _glob.glob(f"/tmp/stepforge_occ_{proc.pid}_*"):
            _shutil.rmtree(_d, ignore_errors=True)
        return (0.0, float("nan"), "timeout", 0)

    if proc.exitcode != 0:
        if verbose:
            print(f"[compute_reward] subprocess segfault in {last_phase} phase: exitcode={proc.exitcode}")
        queue.close()
        # F1: segfault is the COMMON failure for malformed STEP through OCC;
        # the worker's finally never runs. Sweep its temp dirs here too.
        import glob as _glob, shutil as _shutil
        for _d in _glob.glob(f"/tmp/stepforge_occ_{proc.pid}_*"):
            _shutil.rmtree(_d, ignore_errors=True)
        # GT-side segfault → NaN (masked to batch mean), not 0.0 (spurious negative gradient).
        if last_phase == "gt":
            return (float("nan"), float("nan"), "segfault_gt", 0)
        return (0.0, float("nan"), "segfault", 0)

    # Upstream Bug 3 / RA-2: the OS pipe feeder thread can lag behind
    # proc.join(), so the get_nowait() drain above may have missed items.
    # The worker puts up to three items (two phase markers + result), so a
    # single blocking read is insufficient. Drain with a deadline.
    if result is None:
        deadline = time.monotonic() + 5
        while result is None and time.monotonic() < deadline:
            try:
                item = queue.get(timeout=max(0.01, deadline - time.monotonic()))
            except Exception:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "_phase":
                last_phase = item[1]
            else:
                result = item

    queue.close()
    if result is None:
        if verbose:
            print("[compute_reward] subprocess exited cleanly but queue had no result")
        return (0.0, float("nan"), "exception", 0)

    # Populate GT cache on first successful parse (only when not already cached
    # and the worker actually computed it — i.e. result wasn't an early-out).
    if cached is None and result[2] in ("ok", "scd_nonfinite", "pred_degenerate"):
        with _GT_PC_CACHE_LOCK:
            # B1: re-check under lock — another GRPO rollout for the same prompt
            # may have populated this key between our unlocked read and now.
            if cache_key not in _GT_PC_CACHE:
                if len(_GT_PC_CACHE) >= _GT_PC_CACHE_MAX:
                    evict = next(iter(_GT_PC_CACHE), None)
                    if evict is not None:
                        _GT_PC_CACHE.pop(evict, None)
                # We can't recover the GT pc from the result tuple — recompute once in
                # the parent (no segfault risk: this exact GT just succeeded in the child).
                gt_pc, gt_tris = step_to_pointcloud(gt_step, n_points=rcfg.n_points,
                                                    text2cad_src=text2cad_src,
                                                    return_triangle_count=True,
                                                    deflection=rcfg.deflection)
                if gt_pc is not None:
                    _GT_PC_CACHE[cache_key] = (gt_pc, gt_tris)

    return result
