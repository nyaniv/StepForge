"""
Round-trip integrity test for the STEP reserialization pipeline.

For each fixture STEP file:
  1. Tessellate the original via step_to_pointcloud → point cloud A
  2. Run the active pipeline (round_step_numbers → step_restructurer)
  3. Tessellate the output → point cloud B
  4. Assert SCD(A, B) ≈ 0 and entity-count is preserved

This is the single test that would have caught the C7 dangling-ref bug,
the C9 string-literal mutation, and the silent complex-entity drop. None
of those crash; they all produce syntactically valid but geometrically
wrong STEP that flows straight into rag_dataset.json as the SFT label.

Usage:
    pytest tests/test_roundtrip.py -v
    pytest tests/test_roundtrip.py -v --fixtures-dir /path/to/abc/steps
"""

import os
import re
import sys
import glob

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.round_step_numbers import round_float_numbers
from data.step_restructurer import StepRestructurer
from reward.step_to_pointcloud import step_to_pointcloud
from reward.scd_reward import scaled_chamfer_distance


FIXTURES_DIR = os.environ.get(
    "STEPFORGE_FIXTURES",
    os.path.join(os.path.dirname(__file__), "fixtures", "step"),
)
SCD_TOLERANCE = 1e-4


def _count_entities(step_text: str) -> int:
    return len(re.findall(r"^\s*#\d+\s*=", step_text, re.MULTILINE))


def _strip_comments(step_text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", step_text, flags=re.DOTALL)


def _discover_fixtures():
    if not os.path.isdir(FIXTURES_DIR):
        return []
    return sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.step")))[:20]


_fixture_paths = _discover_fixtures()


@pytest.mark.skipif(not _fixture_paths,
                    reason=f"no .step fixtures in {FIXTURES_DIR} "
                           f"(set STEPFORGE_FIXTURES env var)")
@pytest.mark.parametrize("step_path", _fixture_paths,
                         ids=[os.path.basename(p) for p in _fixture_paths])
def test_pipeline_roundtrip_is_geometrically_identical(step_path, tmp_path):
    with open(step_path, encoding="utf-8", errors="replace") as f:
        original = f.read()

    pc_original = step_to_pointcloud(original, n_points=2000)
    if pc_original is None:
        pytest.skip(f"fixture {os.path.basename(step_path)} not renderable by OCC")

    n_orig = _count_entities(original)

    # Match the active pipeline order (step_restructurer.py docstring):
    # 1. step_restructurer (DFS reorder + annotate)  2. round_step_numbers.
    # In production, dataset_construct_rag.py now inlines round_float_numbers
    # on read, so the effective transform is restructure → round.
    src = tmp_path / "in.step"
    out = tmp_path / "out.step"
    src.write_text(original, encoding="utf-8")
    # G1: StepRestructurer() takes no constructor args; matches the active
    # pipeline call pattern in batch_restructure.py:_restructure_semantic.
    StepRestructurer().restructure_step_file(str(src), str(out))
    assert out.exists(), f"restructurer produced no output at {out}"
    transformed = round_float_numbers(out.read_text(encoding="utf-8"))

    n_trans = _count_entities(transformed)
    assert n_trans == n_orig, (
        f"entity count drifted: {n_orig} → {n_trans} "
        f"(silent drop or duplication in the reserialization pipeline)"
    )

    pc_transformed = step_to_pointcloud(_strip_comments(transformed), n_points=2000)
    assert pc_transformed is not None, (
        "reserialized STEP no longer renders — pipeline corrupted geometry"
    )

    scd = scaled_chamfer_distance(pc_transformed, pc_original)
    assert np.isfinite(scd), f"SCD is non-finite ({scd})"
    assert scd < SCD_TOLERANCE, (
        f"SCD={scd:.6e} > {SCD_TOLERANCE:.0e} — reserialization changed geometry. "
        f"Round-trip should be geometrically lossless modulo float rounding."
    )


def test_string_literals_survive_round_step_numbers():
    src = "#1 = PRODUCT('Version 2.1234567 build', '', #2);"
    out = round_float_numbers(src)
    assert "'Version 2.1234567 build'" in out, (
        f"round_float_numbers mutated a string literal: {out!r}"
    )


def test_string_literals_survive_dfs_reserializer():
    from data.dfs_reserializer import reserialize
    header = "ISO-10303-21;\nHEADER;\nENDSEC;"
    entities = {
        1: {"type": "PRODUCT", "args": "'Bracket #99 v2.1234567', '', #2", "refs": [2]},
        2: {"type": "APPLICATION_CONTEXT", "args": "'core data'", "refs": []},
    }
    # G3: step_parser.py never creates empty-set entries; entity 1 has no
    # incoming refs so it's a root and should not appear as a key.
    referenced_by = {2: {1}}
    out = reserialize(header, entities, referenced_by)
    assert "'Bracket #99 v2.1234567'" in out, (
        f"reserialize mutated string literal contents (refs and floats inside "
        f"strings must pass through verbatim):\n{out}"
    )
