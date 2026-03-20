"""
Local web server for STEP-LLM inference.

Usage:
    cd /path/to/StepForge
    CHECKPOINT=checkpoints/rl/final python app/server.py

Then open http://localhost:8000 in your browser.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel
from starlette.background import BackgroundTask

from inference.generate import generate_step
from retrieval.retriever import Retriever

app = FastAPI()

model = None
tokenizer = None
retriever = None
cfg = None


@app.on_event("startup")
async def load_model():
    global model, tokenizer, retriever, cfg

    config_path = os.environ.get("CONFIG", "configs/config.yaml")
    checkpoint = os.environ.get("CHECKPOINT", "checkpoints/rl/final")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")

    cfg = OmegaConf.load(config_path)

    logger.info(f"Loading model from {checkpoint} ...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=True,
        token=hf_token,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.pad_token = tokenizer.eos_token

    retriever = Retriever(
        index_path=cfg.paths.faiss_index_path,
        metadata_path=cfg.paths.faiss_metadata_path,
        model_name=cfg.retrieval.model,
    )
    logger.info("Ready.")


class GenerateRequest(BaseModel):
    caption: str


@app.post("/generate")
async def generate(req: GenerateRequest):
    caption = req.caption.strip()
    if not caption:
        raise HTTPException(400, "Caption cannot be empty")

    logger.info(f"Generating: {caption}")
    step_content = generate_step(
        caption=caption,
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
    )

    if not step_content or "ISO-10303-21;" not in step_content:
        raise HTTPException(500, "Model did not produce a valid STEP file")

    tmp = tempfile.NamedTemporaryFile(suffix=".step", delete=False, mode="w")
    tmp.write(step_content)
    tmp.close()

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", caption[:40])
    filename = (safe_name or "output") + ".step"

    def cleanup():
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return FileResponse(
        tmp.name,
        filename=filename,
        media_type="application/octet-stream",
        background=BackgroundTask(cleanup),
    )


app.mount(
    "/",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="static",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
