import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

BLENDER_BINARY = "/opt/blender/blender"
DEFAULT_RESOLUTION = 1024
DEFAULT_CAMERA = "TopView"
DEFAULT_THREAD_THICKNESS = 0.2

repo_dir = os.path.dirname(os.path.abspath(__file__))
blend_file = os.path.join(repo_dir, "BlenderSetup.blend")
blender_script = os.path.join(repo_dir, "blender_render_still.py")
processed_dir = os.path.join(repo_dir, "processed_files")
service_log = os.path.join(processed_dir, "render_service.log")

if not os.path.exists(BLENDER_BINARY):
    raise RuntimeError(f"Blender binary not found at {BLENDER_BINARY}")
if not os.path.exists(blend_file):
    raise RuntimeError(f"Blend file not found at {blend_file}")
if not os.path.exists(blender_script):
    raise RuntimeError(f"Blender still script not found at {blender_script}")
os.makedirs(processed_dir, exist_ok=True)

logger = logging.getLogger("embroidery_render_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(service_log)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False

app = FastAPI(title="Embroidery Still Render Service (simple)")


def _dedupe_path(candidate_path: str) -> str:
    if not os.path.exists(candidate_path):
        return candidate_path
    stem, ext = os.path.splitext(candidate_path)
    counter = 1
    while True:
        updated = f"{stem}_{counter}{ext}"
        if not os.path.exists(updated):
            return updated
        counter += 1


def run_blender_render(input_path: str, output_path: str) -> float:
    cmd = [
        BLENDER_BINARY,
        "-b",
        blend_file,
        "-P",
        blender_script,
        "--",
        "-i",
        input_path,
        "-o",
        output_path,
        "--resolution",
        str(DEFAULT_RESOLUTION),
        "--camera",
        DEFAULT_CAMERA,
        "--thread_thickness",
        str(DEFAULT_THREAD_THICKNESS),
    ]
    logger.info("Running Blender: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Blender stdout:\n%s", proc.stdout.decode("utf-8", errors="ignore"))
        if proc.stderr:
            logger.info("Blender stderr:\n%s", proc.stderr.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as exc:
        logger.error("Blender failed (returncode=%s): %s", exc.returncode, exc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError(exc.stderr.decode("utf-8", errors="ignore") or str(exc)) from exc
    return time.perf_counter() - start


@app.post("/render", response_class=StreamingResponse)
async def render_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File upload must include a filename.")
    base_name, extension = os.path.splitext(file.filename)
    if not extension:
        extension = ".pes"
    safe_base = base_name.strip() or "upload"
    logger.info("Request received for %s", file.filename)

    with tempfile.TemporaryDirectory(prefix="embroidery_render_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{extension}")
        output_path = os.path.join(tmpdir, "render.png")

        data = await file.read()
        with open(input_path, "wb") as f:
            f.write(data)

        try:
            elapsed = run_blender_render(input_path, output_path)
        except RuntimeError as exc:
            logger.exception("Render failed for '%s'", file.filename)
            raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

        if not os.path.exists(output_path):
            logger.error("Render output missing at %s", output_path)
            raise HTTPException(status_code=500, detail="Render completed but no image was generated.")

        timestamp = int(time.time())
        input_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}{extension.lower()}"))
        output_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}.png"))
        shutil.copyfile(input_path, input_backup)
        shutil.copyfile(output_path, output_backup)
        logger.info("Backed up input to %s and output to %s", input_backup, output_backup)

    with open(output_backup, "rb") as png_file:
        png_bytes = png_file.read()

    headers = {
        "Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename)[0]}.png"',
        "X-Render-Time": f"{elapsed:.2f}",
        "X-Input-Backup": os.path.relpath(input_backup, repo_dir),
        "X-Output-Backup": os.path.relpath(output_backup, repo_dir),
    }
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png", headers=headers)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "message": "Embroidery render API",
        "submit_endpoint": "/render",
        "docs": "/docs",
        "example_curl": 'curl -F "file=@path/to/design.PES" http://localhost:8000/render --output design.png',
    }
