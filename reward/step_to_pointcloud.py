"""
Convert a STEP file string to a sampled 3D point cloud.

Used by the RL reward pipeline (scd_reward.py) and evaluation (evaluate.py).
Writes a temp file, reads with OpenCASCADE STEPControl_Reader, tessellates,
samples points uniformly, deletes the temp file.

Never raises — returns None on any failure so reward can safely return 0.0.
"""

import os
import re
import sys
import tempfile

import hashlib

import numpy as np


def step_to_pointcloud(step_content: str, n_points: int = 2048,
                       text2cad_src: str | None = None,
                       verbose: bool = False) -> np.ndarray | None:
    """
    Convert STEP text → sampled 3D point cloud.

    Args:
        step_content: complete STEP file as a string
        n_points: number of points to sample from the mesh
        text2cad_src: optional path to Text2CAD/CadSeqProc (for OCC import fallback)
        verbose: if True, print reason for failure instead of silently returning None

    Returns:
        (n_points, 3) float32 array, or None if STEP is invalid/unrenderable.
    """
    if text2cad_src:
        parent = os.path.dirname(text2cad_src)
        for p in [text2cad_src, parent]:
            if p not in sys.path:
                sys.path.insert(0, p)

    # Strip /* ... */ comments — added by dfs_reserializer as tree annotations
    # but not supported by OCC/OCP's strict STEP parser.
    step_content = re.sub(r"/\*.*?\*/", "", step_content)

    # The data pipeline (load_step_data_section) stores only the DATA section,
    # so the model learns to output DATA-section-only content (no HEADER).
    # OCC requires a complete file with a valid HEADER block to parse.
    # Detect which format we have and reconstruct a valid file accordingly.
    _MINIMAL_HEADER = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('Open CASCADE Model'),'2;1');\n"
        "FILE_NAME('','',(''),(''),'','','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
    )
    if step_content.lstrip().startswith("DATA;"):
        # DATA-section-only input — prepend a minimal valid STEP header.
        step_content = _MINIMAL_HEADER + step_content.lstrip()
    else:
        # Old format: header present but missing its ENDSEC; terminator.
        step_content = step_content.replace("DATA;", "ENDSEC;\nDATA;", 1)

    # Write to temp file (OCC requires a file path, not a string)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".step")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(step_content)

        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopoDS import topods_Face
        except ImportError:
            from OCP.STEPControl import STEPControl_Reader
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.BRep import BRep_Tool
            from OCP.TopoDS import TopoDS
            topods_Face = TopoDS.Face_s

        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_path)
        if int(status) != 1:  # 1 = IFSelect_RetDone
            if verbose:
                print(f"[step_to_pointcloud] ReadFile failed: status={int(status)}")
            return None

        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            if verbose:
                print("[step_to_pointcloud] OneShape returned null shape")
            return None

        # Tessellate with deflection 0.01
        BRepMesh_IncrementalMesh(shape, 0.01).Perform()

        all_pts = []
        n_faces = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            n_faces += 1
            face = topods_Face(explorer.Current())
            tri = (BRep_Tool.Triangulation_s(face, face.Location())
               if hasattr(BRep_Tool, "Triangulation_s")
               else BRep_Tool.Triangulation(face, face.Location()))
            if tri is not None:
                for i in range(1, tri.NbNodes() + 1):
                    node = tri.Node(i)
                    all_pts.append([node.X(), node.Y(), node.Z()])
            explorer.Next()

        if not all_pts:
            if verbose:
                print(f"[step_to_pointcloud] No points after tessellation (faces={n_faces})")
            return None

        pts = np.array(all_pts, dtype=np.float32)
        unique_pts = len(np.unique(pts, axis=0))
        if verbose:
            print(f"[step_to_pointcloud] OK: faces={n_faces}, raw_pts={len(pts)}, unique_pts={unique_pts}")
        # Seed from content hash so the same STEP always produces the same sample.
        # Must use hashlib (not Python's hash()) — hash() is salted per process,
        # giving different seeds across spawn'd subprocesses for identical inputs.
        seed = int.from_bytes(
            hashlib.sha256(step_content.encode(errors="replace")).digest()[:4], "big"
        )
        idx = np.random.default_rng(seed).choice(len(pts), size=n_points, replace=(len(pts) < n_points))
        return pts[idx]

    except Exception as e:
        if verbose:
            print(f"[step_to_pointcloud] Exception: {e!r}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
