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


def step_to_pointcloud(step_content: str, n_points: int = 2000, *,
                       text2cad_src: str | None = None,
                       verbose: bool = False,
                       return_triangle_count: bool = False,
                       return_mesh: bool = False,
                       deflection: float | None = None):
    """
    Convert STEP text → sampled 3D point cloud (and/or triangulated mesh).

    Args:
        step_content: complete STEP file as a string
        n_points: number of points to sample from the mesh
        text2cad_src: optional path to Text2CAD/CadSeqProc (for OCC import fallback)
        verbose: if True, print reason for failure instead of silently returning None
        return_triangle_count: if True, return (pts, n_triangles) instead of pts.
            scd_reward.py uses this for the reward-hacking guard — barycentric
            sampling produces continuous unique points even from a single
            triangle, so unique-point count can't detect a degenerate mesh.
        return_mesh: if True, also return the world-space triangle array of
            shape (T, 3, 3) for direct mesh rendering. Adds the mesh as the
            last element of the returned tuple. Has no effect on the point
            cloud computation; it's just an additional output.

    Returns:
        (n_points, 3) float64 array sampled uniformly over surface area,
        or None if STEP is invalid/unrenderable.
        If return_triangle_count: (pts | None, n_triangles).
        If return_mesh: same shape with mesh appended as last element;
        mesh is a (T, 3, 3) ndarray of triangle vertices or None on failure.
    """
    def _ret(pts, n_tris=0, mesh=None):
        out = (pts,)
        if return_triangle_count:
            out = out + (n_tris,)
        if return_mesh:
            out = out + (mesh,)
        return out if (return_triangle_count or return_mesh) else pts
    if text2cad_src:
        parent = os.path.dirname(text2cad_src)
        for p in [text2cad_src, parent]:
            if p not in sys.path:
                sys.path.insert(0, p)

    # Strip /* ... */ comments — added by dfs_reserializer as tree annotations
    # but not supported by OCC/OCP's strict STEP parser. re.DOTALL so that a
    # multi-line annotation doesn't survive and corrupt entity boundaries.
    step_content = re.sub(r"/\*.*?\*/", "", step_content, flags=re.DOTALL)

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
    stripped = step_content.lstrip()
    if stripped.startswith("DATA;"):
        # DATA-section-only input — prepend a minimal valid STEP header.
        step_content = _MINIMAL_HEADER + stripped
    elif "ENDSEC;" not in step_content.split("DATA;", 1)[0]:
        # Header present but missing its ENDSEC; terminator. The split-guard
        # avoids producing ENDSEC;\nENDSEC;\nDATA; on already-correct files.
        step_content = step_content.replace("DATA;", "ENDSEC;\nDATA;", 1)

    # Write to temp file (OCC requires a file path, not a string).
    # PID + monotonic suffix on the dir name lets a parent that hard-kills
    # this subprocess sweep /tmp/stepforge_occ_* afterwards.
    tmp_dir = tempfile.mkdtemp(prefix=f"stepforge_occ_{os.getpid()}_")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".step", dir=tmp_dir)
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
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.BRepBndLib import brepbndlib_Add
            _bnd_add = brepbndlib_Add
            _OCP = False
        except ImportError as _occ_err:
            try:
                from OCP.STEPControl import STEPControl_Reader
                from OCP.BRepMesh import BRepMesh_IncrementalMesh
                from OCP.TopExp import TopExp_Explorer
                from OCP.TopAbs import TopAbs_FACE
                from OCP.BRep import BRep_Tool
                from OCP.TopoDS import TopoDS
                from OCP.TopLoc import TopLoc_Location
                from OCP.Bnd import Bnd_Box
                from OCP.BRepBndLib import BRepBndLib
                topods_Face = TopoDS.Face_s
                _bnd_add = BRepBndLib.Add_s
                _OCP = True
            except ImportError as _ocp_err:
                # Neither OCC nor OCP importable. Under mp.spawn this surfaces
                # as 100% pred_parse_fail with no other symptom — re-raise so
                # the failure is loud at startup, not silent at step 80.
                raise ImportError(
                    f"Neither pythonocc (OCC) nor OCP is importable in this process. "
                    f"Under multiprocessing.spawn the child Python may not see the "
                    f"conda environment. OCC error: {_occ_err}. OCP error: {_ocp_err}."
                ) from _ocp_err

        reader = STEPControl_Reader()
        status = reader.ReadFile(tmp_path)
        if int(status) != 1:  # 1 = IFSelect_RetDone
            if verbose:
                print(f"[step_to_pointcloud] ReadFile failed: status={int(status)}")
            return _ret(None)

        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            if verbose:
                print("[step_to_pointcloud] OneShape returned null shape")
            return _ret(None)

        # C11: Adaptive deflection. A fixed 0.01 collapses small parts to single
        # triangles and OOMs on large mm-scale parts. Scale by bbox diagonal.
        bbox = Bnd_Box()
        _bnd_add(shape, bbox)
        if bbox.IsVoid():
            if verbose:
                print("[step_to_pointcloud] Bounding box is void")
            return _ret(None)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        diag = float(np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2))
        # A3: explicit deflection overrides C11's adaptive value. Use 0.1 to match
        # official eval (step_chamfer_reward.py:95); leave None for OOM-safe adaptive.
        defl = deflection if deflection is not None else max(diag * 1e-3, 1e-6)
        BRepMesh_IncrementalMesh(shape, defl, False, 0.5, True).Perform()

        # C3: BRep_Tool.Triangulation returns nodes in face-LOCAL coordinates.
        # The location is an OUTPUT parameter — must apply its transform.
        # W3: Uniform surface sampling weighted by triangle area, not raw vertices.
        all_tris = []   # (3, 3) world-space triangle vertices
        tri_areas = []
        n_faces = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            n_faces += 1
            face = topods_Face(explorer.Current())
            loc = TopLoc_Location()
            tri = (BRep_Tool.Triangulation_s(face, loc) if _OCP
                   else BRep_Tool.Triangulation(face, loc))
            if tri is not None:
                trsf = loc.Transformation()
                identity = trsf.Form() == 0  # gp_Identity
                n_nodes = tri.NbNodes()
                nodes = np.empty((n_nodes, 3), dtype=np.float64)
                for i in range(1, n_nodes + 1):
                    p = tri.Node(i)
                    if not identity:
                        p = p.Transformed(trsf)
                    nodes[i-1] = (p.X(), p.Y(), p.Z())
                # Collect triangle index triples first, then vectorize area computation.
                n_t = tri.NbTriangles()
                idx = np.empty((n_t, 3), dtype=np.int64)
                for i in range(1, n_t + 1):
                    t = tri.Triangle(i)
                    a, b, c = t.Get() if hasattr(t, "Get") else (t.Value(1), t.Value(2), t.Value(3))
                    idx[i-1] = (a-1, b-1, c-1)
                face_tris = nodes[idx]                                   # (n_t, 3, 3)
                e1 = face_tris[:, 1] - face_tris[:, 0]
                e2 = face_tris[:, 2] - face_tris[:, 0]
                face_areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
                keep = face_areas > 0
                if keep.any():
                    all_tris.append(face_tris[keep])
                    tri_areas.append(face_areas[keep])
            explorer.Next()

        if not all_tris:
            if verbose:
                print(f"[step_to_pointcloud] No triangles after tessellation (faces={n_faces})")
            return _ret(None)

        tris  = np.concatenate(all_tris, axis=0)                # (T, 3, 3)
        areas = np.concatenate(tri_areas, axis=0).astype(np.float64)
        areas /= areas.sum()

        # Seed from content hash so the same STEP always produces the same sample.
        # Must use hashlib (not Python's hash()) — hash() is salted per process,
        # giving different seeds across spawn'd subprocesses for identical inputs.
        seed = int.from_bytes(
            hashlib.sha256(step_content.encode(errors="replace")).digest()[:4], "big"
        )
        rng = np.random.default_rng(seed)

        # W3: Barycentric sampling — choose triangles by area, then uniform points within.
        choices = rng.choice(len(tris), size=n_points, p=areas)
        u = rng.random(n_points)
        v = rng.random(n_points)
        flip = (u + v) > 1
        u[flip] = 1 - u[flip]
        v[flip] = 1 - v[flip]
        T = tris[choices]
        pts = T[:, 0] + u[:, None]*(T[:, 1]-T[:, 0]) + v[:, None]*(T[:, 2]-T[:, 0])

        # S7: Drop non-finite points from degenerate faces before they poison Chamfer.
        finite_mask = np.isfinite(pts).all(axis=1)
        if not finite_mask.all():
            pts = pts[finite_mask]
            if len(pts) == 0:
                if verbose:
                    print("[step_to_pointcloud] All sampled points were non-finite")
                return _ret(None, len(tris))

        if verbose:
            unique_pts = len(np.unique(pts, axis=0))
            defl_str = f"{deflection:.3g}" if deflection is not None else "adaptive"
            print(f"[step_to_pointcloud] OK: faces={n_faces}, tris={len(tris)}, "
                  f"sampled={len(pts)}, unique={unique_pts}, deflection={defl_str}")
        return _ret(pts, len(tris), tris)

    except ImportError:
        raise
    except Exception as e:
        if verbose:
            print(f"[step_to_pointcloud] Exception: {e!r}")
        return _ret(None)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass
