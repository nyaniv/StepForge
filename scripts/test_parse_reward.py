"""
Standalone test: verify compute_parse_reward works correctly on Gautschi.

Run from StepForge root:
    python scripts/test_parse_reward.py

Loads one real STEP file from the processed dataset and tests:
1. step_to_pointcloud (OCC rendering)
2. compute_parse_reward (subprocess isolation)

Prints pass/fail + timing for each.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config_gautschi.yaml")
text2cad_src = cfg.paths.text2cad_src
processed_dir = cfg.paths.processed_dir

print(f"text2cad_src = {text2cad_src}")
print(f"processed_dir = {processed_dir}")

# Load a real example from the training set
train_json = os.path.join(processed_dir, "train.json")
print(f"Loading {train_json}...")
with open(train_json) as f:
    records = json.load(f)

record = records[0]
gt_step = record.get("output", "")
print(f"GT STEP length: {len(gt_step)} chars")
print(f"GT STEP first 200 chars: {repr(gt_step[:200])}")

# Test 1: step_to_pointcloud on ground-truth
print("\n--- Test 1: step_to_pointcloud (ground truth) ---")
from reward.step_to_pointcloud import step_to_pointcloud

t0 = time.time()
pts = step_to_pointcloud(gt_step, n_points=100, text2cad_src=text2cad_src, verbose=True)
elapsed = time.time() - t0
if pts is not None:
    print(f"PASS: got point cloud shape={pts.shape} in {elapsed:.2f}s")
else:
    print(f"FAIL: step_to_pointcloud returned None in {elapsed:.2f}s")

# Test 2: compute_parse_reward on ground-truth
print("\n--- Test 2: compute_parse_reward (ground truth) ---")
from reward.scd_reward import compute_parse_reward

t0 = time.time()
r = compute_parse_reward(gt_step, text2cad_src=text2cad_src)
elapsed = time.time() - t0
print(f"Result = {r}  (expected 0.3)  elapsed={elapsed:.2f}s")
if r == 0.3:
    print("PASS")
else:
    print("FAIL — subprocess may have crashed. Check stderr above.")

# Test 3: simulate what the model outputs (DATA; prefix)
print("\n--- Test 3: compute_parse_reward (model-style DATA; prefix) ---")
# Simulate model output: strip DATA; prefix then re-add it (as _fix() does)
stripped = gt_step
if stripped.lstrip().startswith("DATA;"):
    stripped = stripped.lstrip()[len("DATA;"):].lstrip("\n")
simulated_model_output = "DATA;\n" + stripped

t0 = time.time()
r2 = compute_parse_reward(simulated_model_output, text2cad_src=text2cad_src)
elapsed = time.time() - t0
print(f"Result = {r2}  (expected 0.3)  elapsed={elapsed:.2f}s")
if r2 == 0.3:
    print("PASS")
else:
    print("FAIL — DATA; prefix handling may be broken.")

# Test 4: minimal broken STEP (should return 0.0)
print("\n--- Test 4: compute_parse_reward (broken STEP) ---")
r3 = compute_parse_reward("DATA;\n#1=NONSENSE();\nENDSEC;\nEND-ISO-10303-21;", text2cad_src=text2cad_src)
print(f"Result = {r3}  (expected 0.0)")

print("\nDone.")
