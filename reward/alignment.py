"""
Multi-stage point cloud alignment: center → FPFH+RANSAC → ICP.

From paper Section 3.3:
  "We apply multi-stage alignment: center alignment, global registration via
   FPFH + RANSAC, and ICP refinement."

This alignment is applied before computing Scaled Chamfer Distance to make
the metric invariant to translation, rotation, and scale.
"""

import numpy as np
import open3d as o3d

# RANSAC's internal C++ RNG is otherwise non-deterministic across mp.spawn
# subprocesses — identical (pred, gt) pairs would yield different SCD values
# across epochs, adding noise to the GRPO gradient. Open3D's seed is
# process-global so set it once at import time.
o3d.utility.random.seed(42)


def align_point_clouds(pred: np.ndarray, gt: np.ndarray, *,
                       scale_prenorm: bool = True) -> np.ndarray:
    """
    Align pred point cloud to gt using three-stage pipeline.

    Args:
        pred: (N, 3) predicted point cloud
        gt:   (M, 3) ground-truth point cloud
        scale_prenorm: A2 — if True, apply W16 scale pre-normalization before
            FPFH/RANSAC. Paper §3.3 lists exactly three stages (centroid →
            FPFH+RANSAC → ICP); this is a fourth. Set False for paper-exact
            alignment; True is more robust to mm-vs-m unit mismatch.

    Returns:
        (N, 3) aligned pred point cloud
    """
    # Stage 1: Center alignment (subtract centroid from both).
    # float64 — Open3D upcasts internally and we don't want to lose precision
    # before centering on large-offset CAD coordinates.
    pred_c = np.asarray(pred, dtype=np.float64) - np.mean(pred, axis=0, dtype=np.float64)
    gt_c   = np.asarray(gt,   dtype=np.float64) - np.mean(gt,   axis=0, dtype=np.float64)

    gt_rms   = float(np.sqrt(np.mean(np.sum(gt_c   ** 2, axis=1))))
    pred_rms = float(np.sqrt(np.mean(np.sum(pred_c ** 2, axis=1))))
    # A2: gate behind scale_prenorm so eval can run paper-exact 3-stage alignment.
    # W16: Pre-normalize pred to gt's scale. RANSAC and ICP run with
    # with_scaling=False, so a 1000× unit mismatch (mm vs m) silently
    # degenerates: FPFH neighborhoods at the wrong radius find nothing.
    if scale_prenorm and pred_rms > 1e-8 and gt_rms > 1e-8:
        pred_c = pred_c * (gt_rms / pred_rms)

    pred_o3d = o3d.geometry.PointCloud()
    pred_o3d.points = o3d.utility.Vector3dVector(pred_c)
    gt_o3d = o3d.geometry.PointCloud()
    gt_o3d.points = o3d.utility.Vector3dVector(gt_c)

    # Adaptive voxel: 2% of GT's RMS radius.  A fixed absolute value (e.g.
    # 0.05) is meaningless across parts of different scales — RANSAC neighborhoods
    # either collapse to a single voxel or become near-identical, both causing
    # silent alignment failure.  gt_c is already centred so mean≈0 and RMS
    # radius is the natural scale reference (same formula used in scd_reward.py).
    voxel = max(gt_rms * 0.02, 1e-6)  # floor prevents division-by-zero on degenerate clouds

    # Estimate normals (required for FPFH)
    for pcd in [pred_o3d, gt_o3d]:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30)
        )

    # Stage 2: FPFH feature extraction + RANSAC global registration
    pred_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pred_o3d,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100),
    )
    gt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        gt_o3d,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100),
    )

    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pred_o3d,
        gt_o3d,
        pred_fpfh,
        gt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    # When RANSAC fails it returns identity with fitness≈0. Applying identity
    # is harmless, but ICP's tight max_correspondence_distance (0.8% of scale)
    # can't recover from a bad init — widen it to give ICP a chance.
    ransac_ok = ransac_result.fitness > 0.1
    if ransac_ok:
        pred_o3d.transform(ransac_result.transformation)
        icp_dist = voxel * 0.4
    else:
        icp_dist = voxel * 2.0

    # Stage 3: ICP refinement.
    # A5: 3× coarse-to-fine matches official eval (step_chamfer_reward.py:226-237).
    # Each iteration halves the correspondence threshold and seeds from the
    # previous transform, recovering from poorer RANSAC inits than a single pass.
    init = np.eye(4)
    for d in (icp_dist, icp_dist / 2, icp_dist / 4):
        icp_result = o3d.pipelines.registration.registration_icp(
            pred_o3d,
            gt_o3d,
            max_correspondence_distance=d,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        init = icp_result.transformation
    pred_o3d.transform(init)

    return np.asarray(pred_o3d.points)
