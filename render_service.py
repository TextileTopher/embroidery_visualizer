import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def run_blender_render(
    input_path: str,
    fast_output_path: str,
    *,
    resolution: int,
    camera: str,
    thread_thickness: float,
    fast_samples: Optional[int] = None,
    legacy_output_path: Optional[str] = None,
    legacy_resolution: Optional[int] = None,
    legacy_samples: Optional[int] = None,
) -> float:
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
        fast_output_path,
        "--resolution",
        str(resolution),
        "--camera",
        camera,
        "--thread_thickness",
        str(thread_thickness),
    ]
    if fast_samples is not None:
        cmd.extend(["--fast_samples", str(fast_samples)])
    if legacy_output_path:
        cmd.extend(["--legacy_output", legacy_output_path])
        if legacy_resolution is not None:
            cmd.extend(["--legacy_resolution", str(legacy_resolution)])
        if legacy_samples is not None:
            cmd.extend(["--legacy_samples", str(legacy_samples)])

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
async def render_image(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
    resolution: Optional[int] = Form(None),
    camera: str = Form(DEFAULT_CAMERA),
    thread_thickness: float = Form(DEFAULT_THREAD_THICKNESS),
    fast_samples: Optional[int] = Form(None),
    legacy_resolution: Optional[int] = Form(None),
    legacy_samples: Optional[int] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File upload must include a filename.")
    base_name, extension = os.path.splitext(file.filename)
    if not extension:
        extension = ".pes"
    safe_base = base_name.strip() or "upload"
    logger.info("Request received for %s", file.filename)

    requested_mode = (mode or "fast").strip().lower()
    if requested_mode not in {"fast", "legacy", "both"}:
        raise HTTPException(status_code=400, detail="mode must be one of: fast, legacy, both.")

    effective_resolution = resolution or DEFAULT_RESOLUTION
    legacy_requested = requested_mode in {"legacy", "both"}

    input_backup = None
    fast_backup = None
    legacy_backup = None
    timestamp = int(time.time())
    elapsed = 0.0

    with tempfile.TemporaryDirectory(prefix="embroidery_render_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{extension}")
        fast_output_path = os.path.join(tmpdir, "fast.png")
        legacy_output_path = os.path.join(tmpdir, "legacy.png") if legacy_requested else None

        data = await file.read()
        with open(input_path, "wb") as f:
            f.write(data)

        input_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}{extension.lower()}"))
        shutil.copyfile(input_path, input_backup)
        logger.info("Backed up input to %s", input_backup)

        try:
            elapsed = run_blender_render(
                input_path,
                fast_output_path,
                resolution=effective_resolution,
                camera=camera,
                thread_thickness=thread_thickness,
                fast_samples=fast_samples,
                legacy_output_path=legacy_output_path,
                legacy_resolution=legacy_resolution,
                legacy_samples=legacy_samples,
            )
        except RuntimeError as exc:
            logger.exception("Render failed for '%s'", file.filename)
            raise HTTPException(status_code=500, detail=f"Render failed: {exc}") from exc

        if not os.path.exists(fast_output_path):
            logger.error("Fast render output missing at %s", fast_output_path)
            raise HTTPException(status_code=500, detail="Fast render completed but no image was generated.")

        fast_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}_fast.png"))
        shutil.copyfile(fast_output_path, fast_backup)

        if legacy_requested:
            if not legacy_output_path or not os.path.exists(legacy_output_path):
                logger.error("Legacy render output missing at %s", legacy_output_path)
                raise HTTPException(status_code=500, detail="Legacy render was requested but no image was generated.")
            legacy_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}_legacy.png"))
            shutil.copyfile(legacy_output_path, legacy_backup)
            logger.info(
                "Backed up fast output to %s and legacy output to %s",
                fast_backup,
                legacy_backup,
            )
        else:
            logger.info("Backed up fast output to %s", fast_backup)

    headers = {
        "X-Render-Time": f"{elapsed:.2f}",
        "X-Mode": requested_mode,
    }
    if input_backup:
        headers["X-Input-Backup"] = os.path.relpath(input_backup, repo_dir)
    if fast_backup:
        headers["X-Fast-Output-Backup"] = os.path.relpath(fast_backup, repo_dir)
    if legacy_backup:
        headers["X-Legacy-Output-Backup"] = os.path.relpath(legacy_backup, repo_dir)

    if requested_mode == "both":
        archive_name = f"{safe_base}_{timestamp}.zip"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if fast_backup:
                archive.write(fast_backup, arcname=f"{safe_base}_fast.png")
            if legacy_backup:
                archive.write(legacy_backup, arcname=f"{safe_base}_legacy.png")
        zip_buffer.seek(0)
        headers["Content-Disposition"] = f'attachment; filename="{archive_name}"'
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    if requested_mode == "legacy":
        if not legacy_backup:
            raise HTTPException(status_code=500, detail="Legacy render requested but no image was produced.")
        target_backup = legacy_backup
        filename = f"{safe_base}_legacy.png"
    else:
        target_backup = fast_backup
        filename = f"{safe_base}_fast.png"

    if not target_backup or not os.path.exists(target_backup):
        raise HTTPException(status_code=500, detail="Requested render completed but could not locate the output file.")

    with open(target_backup, "rb") as png_file:
        png_bytes = png_file.read()

    headers["Content-Disposition"] = f'attachment; filename="{filename}"'
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
        "example_fast": 'curl -F "mode=fast" -F "file=@path/to/design.PES" http://localhost:8000/render --output design_fast.png',
        "example_legacy": 'curl -F "mode=legacy" -F "file=@path/to/design.PES" http://localhost:8000/render --output design_legacy.png',
        "example_both": 'curl -F "mode=both" -F "file=@path/to/design.PES" http://localhost:8000/render --output design_outputs.zip',
    }
