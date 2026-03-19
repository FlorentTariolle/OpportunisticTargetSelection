"""FastAPI backend for the adversarial attack demonstrator."""

import asyncio
import base64
import io
import os
import sys

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.demo.app import (
    STANDARD_MODELS,
    predict_image,
    run_attack,
)
from src.models.loader import ROBUSTBENCH_MODELS

import torch

app = FastAPI(title="Adversarial Attack Demonstrator")

# Static file mounts
static_dir = os.path.join(os.path.dirname(__file__), "static")
data_dir = os.path.join(project_root, "data")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data", StaticFiles(directory=data_dir), name="data")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/api/gpu")
async def get_gpu():
    if torch.cuda.is_available():
        return {"available": True, "name": torch.cuda.get_device_name(0)}
    return {"available": False, "name": None}


DEMO_STANDARD_MODELS = ["resnet18", "resnet50", "vgg16", "alexnet", "vit_b_16"]
DEMO_ROBUST_MODELS = ["Salman2020Do_R18", "Salman2020Do_R50"]


@app.get("/api/models/{source}")
async def get_models(source: str):
    if source == "robust":
        return {"models": DEMO_ROBUST_MODELS}
    return {"models": DEMO_STANDARD_MODELS}


@app.get("/api/examples")
async def get_examples():
    examples = []
    if os.path.exists(data_dir):
        # basketball.jpg first
        for filename in sorted(os.listdir(data_dir)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                examples.append({"name": filename, "url": f"/data/{filename}"})
    # Move basketball to front if present
    examples.sort(key=lambda e: (0 if e["name"] == "basketball.jpg" else 1, e["name"]))
    return {"examples": examples}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form("resnet50"),
    source: str = Form("standard"),
):
    pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    label, confidence, class_index = await asyncio.to_thread(
        predict_image, pil_image, model_name, source
    )
    return {"label": label, "confidence": confidence, "class_index": class_index}


@app.post("/api/attack")
async def attack(
    image: UploadFile = File(...),
    method: str = Form("SimBA"),
    epsilon_n: int = Form(8),
    max_iterations: int = Form(10000),
    model_name: str = Form("resnet50"),
    source: str = Form("standard"),
    mode: str = Form("untargeted"),
    target_class: int = Form(-1),
    opportunistic: str = Form("false"),
    switch_iteration: int = Form(10),
    loss: str = Form("ce"),
    seed: int = Form(42),
):
    pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

    epsilon = epsilon_n / 255.0
    targeted = mode == "targeted"
    tc = max(0, min(999, target_class)) if targeted and target_class >= 0 else None
    use_opportunistic = opportunistic in ("true", "on", "1") and not targeted

    adv_img, pert_img, conf_graph, result_text = await asyncio.to_thread(
        run_attack,
        pil_image,
        method,
        epsilon,
        max_iterations,
        model_name,
        targeted,
        tc,
        use_opportunistic,
        switch_iteration,
        loss,
        source,
        seed,
    )

    return JSONResponse({
        "adversarial_image": pil_to_base64(adv_img) if adv_img else None,
        "perturbation_image": pil_to_base64(pert_img) if pert_img else None,
        "confidence_graph": pil_to_base64(conf_graph) if conf_graph else None,
        "result_text": result_text or "",
    })
