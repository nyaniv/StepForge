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


def align_point_clouds(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Align pred point cloud to gt using three-stage pipeline.

    Args:
        pred: (N, 3) predicted point cloud
        gt:   (M, 3) ground-truth point cloud

    Returns:
        (N, 3) aligned pred point cloud
    """
    # Stage 1: Center alignment (subtract centroid from both)
    pred_c = pred - pred.mean(axis=0)
    gt_c   = gt   - gt.mean(axis=0)

    pred_o3d = o3d.geometry.PointCloud()
    pred_o3d.points = o3d.utility.Vector3dVector(pred_c)
    gt_o3d = o3d.geometry.PointCloud()
    gt_o3d.points = o3d.utility.Vector3dVector(gt_c)

    # Adaptive voxel: 2% of GT's RMS radius.  A fixed absolute value (e.g.
    # 0.05) is meaningless across parts of different scales — RANSAC neighborhoods
    # either collapse to a single voxel or become near-identical, both causing
    # silent alignment failure.  gt_c is already centred so mean≈0 and RMS
    # radius is the natural scale reference (same formula used in scd_reward.py).
    gt_rms = float(np.sqrt(np.mean(np.sum(gt_c ** 2, axis=1))))
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
    pred_o3d.transform(ransac_result.transformation)

    # Stage 3: ICP refinement
    icp_result = o3d.pipelines.registration.registration_icp(
        pred_o3d,
        gt_o3d,
        max_correspondence_distance=voxel * 0.4,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_o3d.transform(icp_result.transformation)

    return np.asarray(pred_o3d.points, dtype=np.float32)
