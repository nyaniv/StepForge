"""
STEP-LLM Gradio App — Text to CAD.

Usage:
    cd /path/to/StepLLM
    CHECKPOINT=checkpoints/rl/final python app.py

Then open http://localhost:7860
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from loguru import logger
from omegaconf import OmegaConf

from inference.generate import generate_step
from retrieval.retriever import Retriever

# ── Load model once at startup ─────────────────────────────────────────────

CONFIG     = os.environ.get("CONFIG",     "configs/config.yaml")
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/rl/final")
HF_TOKEN   = os.environ.get("HUGGINGFACE_TOKEN")

cfg = OmegaConf.load(CONFIG)

logger.info(f"Loading model from {CHECKPOINT} ...")
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT,
    max_seq_length=cfg.model.max_seq_length,
    load_in_4bit=True,
    token=HF_TOKEN,
)
FastLanguageModel.for_inference(model)
tokenizer.pad_token = tokenizer.eos_token

retriever = Retriever(
    index_path=cfg.paths.faiss_index_path,
    metadata_path=cfg.paths.faiss_metadata_path,
    model_name=cfg.retrieval.model,
)
logger.info("Ready.")


# ── STEP → STL conversion (for 3D preview) ────────────────────────────────

def step_to_stl(step_content: str, stl_path: str) -> bool:
    """Convert STEP string to STL file using pythonOCC. Returns True on success."""
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.StlAPI import StlAPI_Writer
    except ImportError:
        logger.warning("pythonOCC not available — 3D preview disabled")
        return False

    step_tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".step", mode="w", delete=False) as f:
            f.write(step_content)
            step_tmp = f.name

        reader = STEPControl_Reader()
        if reader.ReadFile(step_tmp) != IFSelect_RetDone:
            return False
        reader.TransferRoots()
        shape = reader.OneShape()

        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.Write(shape, stl_path)
        return os.path.getsize(stl_path) > 80

    except Exception as e:
        logger.warning(f"STL conversion failed: {e}")
        return False
    finally:
        if step_tmp and os.path.exists(step_tmp):
            os.unlink(step_tmp)


# ── Main inference function ────────────────────────────────────────────────

MAX_RETRIES = 3


def generate(caption: str):
    caption = caption.strip()
    if not caption:
        raise gr.Error("Please enter a description.")

    last_error = "Unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Attempt {attempt}/{MAX_RETRIES}: {caption}")
        try:
            step_content = generate_step(
                caption=caption,
                model=model,
                tokenizer=tokenizer,
                retriever=retriever,
            )

            if not step_content or "ISO-10303-21;" not in step_content:
                last_error = "Model did not produce a valid STEP file."
                continue

            # Save STEP file for download
            prefix = caption[:20].replace(" ", "_")
            step_file = tempfile.NamedTemporaryFile(
                suffix=".step", mode="w", delete=False, prefix=prefix + "_"
            )
            step_file.write(step_content)
            step_file.close()

            # Convert to STL for 3D preview
            stl_file = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
            stl_file.close()
            stl_ok = step_to_stl(step_content, stl_file.name)

            return (
                stl_file.name if stl_ok else None,
                step_file.name,
            )

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt} failed: {e}")

    raise gr.Error(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


# ── Gradio UI ──────────────────────────────────────────────────────────────

EXAMPLES = [
    ["a hollow cylinder"],
    ["a rectangular plate with four holes at the corners"],
    ["a 90-degree pipe elbow connector"],
    ["a hexagonal bolt head"],
    ["a flat disc with a central hole"],
]

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        label="Object description",
        placeholder="e.g. a hollow cylinder with a flange at the base",
        lines=2,
    ),
    outputs=[
        gr.Model3D(
            clear_color=[0.15, 0.15, 0.15, 1.0],
            label="3D Preview",
        ),
        gr.File(label="Download STEP File"),
    ],
    examples=EXAMPLES,
    title="STEP-LLM: Text → CAD",
    description=(
        "Generate a STEP CAD file from a plain English description. "
        "The 3D preview renders the geometry — download the STEP file "
        "to open in Fusion 360, FreeCAD, or any CAD tool."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue().launch(server_port=7860, theme=gr.themes.Soft())
