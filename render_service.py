import io
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

def _load_env_file(path: str, *, override: bool) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                if not key:
                    continue
                if override or key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


from apply_texture_background import apply_texture_overlay, replace_white_with_texture

BLENDER_BINARY = "/opt/blender/blender"
DEFAULT_RESOLUTION = 1024
DEFAULT_CAMERA = "TopView"
DEFAULT_THREAD_THICKNESS = 0.2
DEFAULT_LOW_QUALITY_RESOLUTION = 768
DEFAULT_HIGH_QUALITY_RESOLUTION = 2048
DEFAULT_LOW_QUALITY_SAMPLES = 64
DEFAULT_HIGH_QUALITY_SAMPLES = 512

repo_dir = os.path.dirname(os.path.abspath(__file__))
blend_file = os.path.join(repo_dir, "BlenderSetup.blend")
blender_still_script = os.path.join(repo_dir, "blender_render_still.py")
blender_video_script = os.path.join(repo_dir, "blender_script.py")
processed_dir = os.path.join(repo_dir, "processed_files")
service_log = os.path.join(processed_dir, "render_service.log")
textures_dir = os.path.join(repo_dir, "assets", "textures")
TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DEFAULT_BACKGROUND_TOLERANCE = 8
CANVAS_TEXTURE_FILE = os.path.join(textures_dir, "canvas1.png")

env_path = os.path.join(repo_dir, ".env")
venv_env_path = os.path.join(repo_dir, ".venv", ".env")
if load_dotenv:
    load_dotenv(dotenv_path=venv_env_path, override=False)
    load_dotenv(dotenv_path=env_path, override=True)

_load_env_file(venv_env_path, override=False)
_load_env_file(env_path, override=True)

# Allow AWS_* env names to populate (and override) SPACES_* variables.
ENV_FALLBACKS = {
    "SPACES_ENDPOINT": "AWS_S3_ENDPOINT_URL",
    "SPACES_ACCESS_KEY": "AWS_ACCESS_KEY_ID",
    "SPACES_SECRET_KEY": "AWS_SECRET_ACCESS_KEY",
    "SPACES_REGION": "AWS_REGION",
}
for target, source in ENV_FALLBACKS.items():
    value = os.getenv(source)
    if value:
        os.environ[target] = value

if not os.path.exists(BLENDER_BINARY):
    raise RuntimeError(f"Blender binary not found at {BLENDER_BINARY}")
if not os.path.exists(blend_file):
    raise RuntimeError(f"Blend file not found at {blend_file}")
if not os.path.exists(blender_still_script):
    raise RuntimeError(f"Blender still script not found at {blender_still_script}")
if not os.path.exists(blender_video_script):
    raise RuntimeError(f"Blender video script not found at {blender_video_script}")
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

MAX_INPUT_BYTES = 50 * 1024 * 1024
S3_MAX_ATTEMPTS = 20
OUTPUT_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%SZ"

SPACES_ENDPOINT = os.getenv("SPACES_ENDPOINT")
SPACES_REGION = os.getenv("SPACES_REGION")
SPACES_ACCESS_KEY = os.getenv("SPACES_ACCESS_KEY")
SPACES_SECRET_KEY = os.getenv("SPACES_SECRET_KEY")
PUBLIC_READ_ACL = {"ACL": "public-read"}


def _normalize_region(endpoint: Optional[str], region: Optional[str]) -> str:
    if not endpoint:
        return region or "sfo3"
    parsed = urlparse(endpoint)
    host = parsed.netloc or ""
    # Default fallback if parsing fails
    default_region = "sfo3"
    if not region:
        return default_region
    normalized = region.strip()
    aws_like = bool(re.fullmatch(r"[a-z]{2}-[a-z]+-\d", normalized))
    if aws_like and default_region not in normalized:
        logger.warning(
            "SPACES_REGION '%s' does not match DigitalOcean region inferred from endpoint '%s'; defaulting to '%s'.",
            normalized,
            endpoint,
            default_region,
        )
        return default_region
    return normalized


def _log_endpoint_warnings(endpoint: Optional[str]) -> None:
    if not endpoint:
        logger.warning("SPACES_ENDPOINT is not configured; /render-from-space will fail until it is set.")
        return
    parsed = urlparse(endpoint)
    host = parsed.netloc or ""
    if ".cdn." in host:
        logger.warning(
            "SPACES_ENDPOINT '%s' contains '.cdn.'; use the S3 API endpoint such as https://sfo3.digitaloceanspaces.com instead.",
            endpoint,
        )
    suffix = ".digitaloceanspaces.com"
    if host.endswith(suffix):
        prefix = host[: -len(suffix)]
        if prefix.count(".") >= 1:
            logger.warning(
                "SPACES_ENDPOINT '%s' appears to include a bucket name; the endpoint must be regional like https://sfo3.digitaloceanspaces.com.",
                endpoint,
            )


SPACES_REGION = _normalize_region(SPACES_ENDPOINT, SPACES_REGION)
_log_endpoint_warnings(SPACES_ENDPOINT)

_s3_client: Optional[Any] = None


def get_s3_client():
    global _s3_client
    if _s3_client:
        return _s3_client
    if not all([SPACES_ENDPOINT, SPACES_ACCESS_KEY, SPACES_SECRET_KEY]):
        raise RuntimeError("DigitalOcean Spaces configuration is incomplete; set SPACES_ENDPOINT, ACCESS_KEY, and SECRET_KEY.")
    session = boto3.session.Session(
        aws_access_key_id=SPACES_ACCESS_KEY,
        aws_secret_access_key=SPACES_SECRET_KEY,
    )
    config = Config(
        region_name=SPACES_REGION,
        retries={
            "max_attempts": S3_MAX_ATTEMPTS,
            "mode": "standard",
        },
    )
    _s3_client = session.client("s3", endpoint_url=SPACES_ENDPOINT, config=config)
    return _s3_client


class SpacesObjectNotFound(Exception):
    """Raised when a requested Spaces object is not found."""


class SpacesCollisionError(Exception):
    """Raised when a Spaces key collision cannot be resolved."""


def normalize_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    candidate = path.replace("\\", "/").strip("/")
    if not candidate:
        return ""
    segments = candidate.split("/")
    for segment in segments:
        if segment in {"", ".", ".."} or ".." in segment:
            raise ValueError(f"Invalid path segment '{segment}'.")
    return "/".join(segments)


def safe_file_name(name: str) -> str:
    if not name:
        raise ValueError("File name cannot be empty.")
    candidate = name.replace("\\", "/").strip("/")
    if "/" in candidate or ".." in candidate:
        raise ValueError("File name must not contain path separators or '..'.")
    return candidate


def compose_key(path: Optional[str], name: str) -> str:
    sanitized_name = safe_file_name(name)
    if not path:
        return sanitized_name
    normalized_path = normalize_path(path)
    if normalized_path:
        return f"{normalized_path}/{sanitized_name}"
    return sanitized_name


def head_object_size(bucket: str, key: str) -> int:
    client = get_s3_client()
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise SpacesObjectNotFound(f"{bucket}/{key}") from exc
        raise
    return int(response.get("ContentLength", 0))


def download_to_file(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()
    try:
        client.download_file(bucket, key, str(dest))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise SpacesObjectNotFound(f"{bucket}/{key}") from exc
        raise


def _key_exists(bucket: str, key: str) -> bool:
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _apply_timestamp_suffix(key: str, timestamp: str) -> str:
    base, ext = os.path.splitext(key)
    return f"{base}_{timestamp}{ext}"


def upload_with_suffix_on_conflict(bucket: str, key: str, src: Path) -> Dict[str, int]:
    client = get_s3_client()
    timestamp = datetime.now(timezone.utc).strftime(OUTPUT_TIMESTAMP_FORMAT)
    final_key = key
    if _key_exists(bucket, key):
        candidate = _apply_timestamp_suffix(key, timestamp)
        if _key_exists(bucket, candidate):
            raise SpacesCollisionError(f"Unable to resolve key collision for {bucket}/{key}")
        final_key = candidate
    client.upload_file(str(src), bucket, final_key, ExtraArgs=dict(PUBLIC_READ_ACL))
    size_bytes = src.stat().st_size
    return {"final_key": final_key, "size_bytes": size_bytes}


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


def _safe_backup_prefix(filename: str) -> str:
    base, _ = os.path.splitext(filename)
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", base.strip())
    return candidate or "upload"


def _normalize_framing(value: Optional[str]) -> str:
    if not value:
        return "full"
    value = value.strip().lower()
    if value not in {"full", "zoomed"}:
        return "full"
    return value


def _list_available_textures() -> List[str]:
    if not os.path.isdir(textures_dir):
        return []
    files = []
    for entry in os.listdir(textures_dir):
        path = os.path.join(textures_dir, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in TEXTURE_EXTENSIONS:
            files.append(entry)
    return sorted(files)


def _resolve_texture_file(name: str) -> Optional[str]:
    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    candidate = os.path.join(textures_dir, name)
    if os.path.exists(candidate):
        return candidate
    return None


def _apply_solid_background(image_path: str, color: Tuple[int, int, int], tolerance: int) -> None:
    with Image.open(image_path) as render:
        color_tile = Image.new("RGB", (8, 8), color)
        composited = replace_white_with_texture(render, color_tile, tolerance)
        composited.save(image_path)


def apply_background_choice(image_path: str, choice: Optional[str], tolerance: Optional[int]) -> Optional[str]:
    if not choice:
        return "white"
    normalized = choice.strip().lower()
    if not normalized or normalized == "white":
        return "white"

    tolerance = tolerance if tolerance is not None else DEFAULT_BACKGROUND_TOLERANCE

    if normalized == "black":
        _apply_solid_background(image_path, (0, 0, 0), tolerance)
        return "black"

    if normalized == "random":
        textures = _list_available_textures()
        if not textures:
            logger.warning("No textures found in %s; keeping white background.", textures_dir)
            return "white"
        selection = random.choice(textures)
        texture_path = _resolve_texture_file(selection)
        if not texture_path:
            logger.warning("Randomly selected texture %s missing; keeping white background.", selection)
            return "white"
        apply_texture_overlay(image_path, texture_path, image_path, tolerance=tolerance)
        return selection

    texture_path = _resolve_texture_file(choice)
    if not texture_path:
        logger.warning("Texture '%s' not found; keeping white background.", choice)
        return "white"

    apply_texture_overlay(image_path, texture_path, image_path, tolerance=tolerance)
    return os.path.basename(texture_path)


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
        blender_still_script,
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


def run_blender_video(
    input_path: str,
    temp_image_path: str,
    output_video_path: str,
    *,
    video_quality: Optional[str] = None,
    video_framing: Optional[str] = None,
) -> float:
    cmd = [
        BLENDER_BINARY,
        "-b",
        blend_file,
        "-P",
        blender_video_script,
        "--",
        "-i",
        input_path,
        "-o",
        temp_image_path,
        "-v",
        output_video_path,
    ]
    if video_quality:
        cmd.extend(["--video_quality", video_quality])
    if video_framing:
        cmd.extend(["--video_framing", video_framing])
    logger.info("Running Blender (video): %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Video Blender stdout:\n%s", proc.stdout.decode("utf-8", errors="ignore"))
        if proc.stderr:
            logger.info("Video Blender stderr:\n%s", proc.stderr.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as exc:
        logger.error("Video render failed (returncode=%s): %s", exc.returncode, exc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError(exc.stderr.decode("utf-8", errors="ignore") or str(exc)) from exc
    return time.perf_counter() - start


class SpacesInputSpec(BaseModel):
    bucket_name: str = Field(..., min_length=1)
    file_path: Optional[str] = None
    file_name: str = Field(..., min_length=1)

    @field_validator("file_path", mode="before")
    @classmethod
    def _normalize_file_path(cls, value):
        if value in (None, "", "null"):
            return None
        return normalize_path(str(value))

    @field_validator("file_name")
    @classmethod
    def _validate_file_name(cls, value: str) -> str:
        sanitized = safe_file_name(value)
        if not sanitized.lower().endswith(".pes"):
            raise ValueError("file_name must end with .pes")
        return sanitized


class SpacesOutputSpec(BaseModel):
    bucket_name: Optional[str] = None
    base_path: Optional[str] = None

    @field_validator("base_path", mode="before")
    @classmethod
    def _normalize_base_path(cls, value):
        if value in (None, "", "null"):
            return None
        return normalize_path(str(value))


class OperationFlags(BaseModel):
    image_generate_low: bool = False
    image_generate_high: bool = False
    video_generate_low: bool = False
    video_generate_high: bool = False
    video_generate_ultra_high: bool = False
    generate_low: bool = Field(default=False, description="Legacy alias for image_generate_low")
    generate_high: bool = Field(default=False, description="Legacy alias for image_generate_high")
    generate_video: bool = Field(default=False, description="Legacy alias for video_generate_*")

    @model_validator(mode="after")
    def _ensure_primary_render(self):
        if not (self.image_generate_low or self.generate_low or self.image_generate_high or self.generate_high):
            raise ValueError("At least one of generate_low or generate_high must be true.")
        return self


class ImageRenderOptions(BaseModel):
    resolution: Optional[int] = Field(default=None, gt=0)
    legacy_resolution: Optional[int] = Field(default=None, gt=0)
    fast_samples: Optional[int] = Field(default=None, gt=0)
    legacy_samples: Optional[int] = Field(default=None, gt=0)
    low_quality_resolution: Optional[int] = Field(default=None, gt=0)
    high_quality_resolution: Optional[int] = Field(default=None, gt=0)
    low_quality_samples: Optional[int] = Field(default=None, gt=0)
    high_quality_samples: Optional[int] = Field(default=None, gt=0)
    camera: Optional[str] = Field(default=None, min_length=1)
    thread_thickness: Optional[float] = Field(default=None, gt=0)
    background: Optional[str] = Field(default=None, description="Background selection (white, black, texture filename, or random)")
    background_tolerance: Optional[int] = Field(default=None, gt=0)


# Backwards compatibility alias
RenderOptions = ImageRenderOptions


class RenderFromSpaceRequest(BaseModel):
    input: SpacesInputSpec
    operations: OperationFlags
    output: SpacesOutputSpec
    render_options: Optional[RenderOptions] = None  # legacy alias for image_render_options
    image_render_options: Optional[ImageRenderOptions] = None
    debug: bool = False


class InputSummary(BaseModel):
    bucket_name: str
    object_key: str
    size_bytes: int


class AssetSummary(BaseModel):
    object_key: str
    size_bytes: int


class OutputSummary(BaseModel):
    bucket_name: str
    base_path: str
    low_quality_png: Optional[AssetSummary] = None
    low_quality_canvas_png: Optional[AssetSummary] = None
    high_quality_png: Optional[AssetSummary] = None
    high_quality_canvas_png: Optional[AssetSummary] = None
    video_low_mp4: Optional[AssetSummary] = None
    video_high_mp4: Optional[AssetSummary] = None
    video_ultra_mp4: Optional[AssetSummary] = None
    video_mp4: Optional[AssetSummary] = None  # legacy field


class StageMetrics(BaseModel):
    low_quality_png_seconds: Optional[float] = None
    high_quality_png_seconds: Optional[float] = None
    video_low_seconds: Optional[float] = None
    video_high_seconds: Optional[float] = None
    video_ultra_seconds: Optional[float] = None
    video_seconds: Optional[float] = None  # legacy aggregate


class RenderMetrics(BaseModel):
    render_time_seconds_total: float
    stages: StageMetrics


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.2f}s"


def _safe_relpath(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return os.path.relpath(path, repo_dir)
    except Exception:
        return str(path)


def _read_log_tail(lines: int = 60) -> Optional[str]:
    if not os.path.exists(service_log):
        return None
    try:
        from collections import deque

        with open(service_log, "r", encoding="utf-8", errors="replace") as log_file:
            tail = deque(log_file, maxlen=lines)
        return "".join(tail)
    except Exception:
        return None


def _generate_canvas_overlay(src: Path, dest: Path, tolerance: int) -> Optional[Path]:
    if not os.path.exists(CANVAS_TEXTURE_FILE):
        return None
    try:
        apply_texture_overlay(
            render_path=str(src),
            texture_path=CANVAS_TEXTURE_FILE,
            output_path=str(dest),
            tolerance=tolerance,
        )
        return dest
    except Exception as exc:
        logger.warning("Failed to create canvas overlay for %s: %s", src, exc)
        return None


def _build_completion_message(
    request_model: "RenderFromSpaceRequest",
    *,
    input_key: str,
    total_elapsed: float,
    stage_metrics: StageMetrics,
    uploaded_assets: Dict[str, AssetSummary],
    output_bucket: str,
    errors: List[str],
    debug_enabled: bool,
) -> str:
    lines = [
        f"Input file name: {request_model.input.file_name}",
        f"Input file location: {input_key}",
        f"Input bucket: {request_model.input.bucket_name}",
        f"Total processing time: {_format_duration(total_elapsed)}",
    ]

    asset_labels = {
        "low_quality_png": ("Low-quality PNG", stage_metrics.low_quality_png_seconds),
        "low_quality_canvas_png": ("Low-quality Canvas PNG", stage_metrics.low_quality_png_seconds),
        "high_quality_png": ("High-quality PNG", stage_metrics.high_quality_png_seconds),
        "high_quality_canvas_png": ("High-quality Canvas PNG", stage_metrics.high_quality_png_seconds),
        "video_low_mp4": ("Low-quality MP4", stage_metrics.video_low_seconds),
        "video_high_mp4": ("High-quality MP4", stage_metrics.video_high_seconds),
        "video_ultra_mp4": ("Ultra-high-quality MP4", stage_metrics.video_ultra_seconds),
        "video_mp4": ("Video MP4", stage_metrics.video_seconds),
    }

    for key, (label, elapsed) in asset_labels.items():
        asset = uploaded_assets.get(key)
        if not asset:
            continue
        lines.extend(
            [
                f"{label} return file name: {Path(asset.object_key).name}",
                f"{label} return file location: {asset.object_key}",
                f"{label} return bucket: {output_bucket}",
                f"{label} processing time: {_format_duration(elapsed)}",
            ]
        )

    if errors:
        lines.append(f"Errors: {'; '.join(errors)}")
    if debug_enabled:
        lines.append("Debug command executed: response includes debug diagnostics.")
    else:
        lines.append('Debug command: add "debug": true to the request payload to return detailed diagnostics.')
    lines.append("File processing complete.")
    return "\n".join(lines)


class RenderFromSpaceResponse(BaseModel):
    job_id: str
    input: InputSummary
    outputs: OutputSummary
    metrics: RenderMetrics
    message: str
    debug: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


@app.post("/render-from-space")
async def render_from_space(payload: Dict[str, Any]):
    job_id = str(uuid.uuid4())
    logger.info("job %s: received /render-from-space request", job_id)
    try:
        request_model = RenderFromSpaceRequest(**payload)
    except ValidationError as exc:
        logger.warning("job %s: validation error: %s", job_id, exc)
        raise HTTPException(
            status_code=400,
            detail={"job_id": job_id, "message": "Invalid request payload.", "errors": exc.errors()},
        ) from exc

    try:
        input_key = compose_key(request_model.input.file_path, request_model.input.file_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"job_id": job_id, "message": str(exc)}) from exc

    debug_enabled = bool(request_model.debug)

    output_bucket = request_model.output.bucket_name or request_model.input.bucket_name
    if not output_bucket:
        raise HTTPException(
            status_code=400,
            detail={"job_id": job_id, "message": "Output bucket is required when input bucket is not provided."},
        )

    base_root = request_model.output.base_path
    if base_root is None:
        base_root = request_model.input.file_path or ""
    final_base_path = base_root or ""
    compose_base_path = final_base_path or None

    try:
        input_size = head_object_size(request_model.input.bucket_name, input_key)
    except SpacesObjectNotFound as exc:
        logger.warning("job %s: input not found at %s", job_id, exc)
        raise HTTPException(
            status_code=404,
            detail={
                "job_id": job_id,
                "message": f"Input object '{input_key}' not found in bucket '{request_model.input.bucket_name}'.",
            },
        ) from exc

    if input_size > MAX_INPUT_BYTES:
        logger.warning(
            "job %s: input size %s exceeds limit of %s bytes",
            job_id,
            input_size,
            MAX_INPUT_BYTES,
        )
        raise HTTPException(
            status_code=413,
            detail={
                "job_id": job_id,
                "message": f"Input object is {input_size} bytes which exceeds the 50 MB limit.",
            },
        )

    input_backup_path: Optional[Path] = None
    low_quality_backup_path: Optional[Path] = None
    high_quality_backup_path: Optional[Path] = None
    video_backup_path: Optional[Path] = None

    overall_start = time.perf_counter()
    stage_metrics: Dict[str, float] = {}
    errors: List[str] = []
    uploaded_assets: Dict[str, AssetSummary] = {}
    safe_base_prefix = _safe_backup_prefix(request_model.input.file_name)
    timestamp_suffix = int(time.time())

    fast_requested = request_model.operations.image_generate_low or request_model.operations.generate_low
    legacy_requested = request_model.operations.image_generate_high or request_model.operations.generate_high
    image_options = (
        request_model.image_render_options
        or request_model.render_options
        or ImageRenderOptions()
    )
    canvas_tolerance = image_options.background_tolerance or DEFAULT_BACKGROUND_TOLERANCE
    default_video_quality = "high"
    default_video_framing = _normalize_framing("full")
    video_low_framing = default_video_framing
    video_high_framing = default_video_framing
    video_ultra_framing = default_video_framing
    video_low_requested = request_model.operations.video_generate_low
    video_high_requested = request_model.operations.video_generate_high
    video_ultra_requested = getattr(request_model.operations, "video_generate_ultra_high", False)
    if request_model.operations.generate_video and not (
        video_low_requested or video_high_requested or video_ultra_requested
    ):
        if default_video_quality in {"fast", "low"}:
            video_low_requested = True
        elif default_video_quality == "ultra":
            video_ultra_requested = True
        else:
            video_high_requested = True
    video_requested = video_low_requested or video_high_requested or video_ultra_requested

    try:
        with tempfile.TemporaryDirectory(prefix=f"render_{job_id}_") as tmpdir:
            tmp_path = Path(tmpdir)
            local_input_path = tmp_path / request_model.input.file_name
            try:
                download_to_file(request_model.input.bucket_name, input_key, local_input_path)
                logger.info(
                    "job %s: downloaded input to %s (size=%s)", job_id, local_input_path, input_size
                )
            except SpacesObjectNotFound as exc:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "job_id": job_id,
                        "message": f"Input object '{input_key}' not found in bucket '{request_model.input.bucket_name}'.",
                    },
                ) from exc

            input_backup_path = Path(
                _dedupe_path(
                    os.path.join(
                        processed_dir,
                        f"{safe_base_prefix}_{timestamp_suffix}{os.path.splitext(request_model.input.file_name)[1].lower()}",
                    )
                )
            )
            shutil.copyfile(local_input_path, input_backup_path)
            logger.info("job %s: backed up input to %s", job_id, input_backup_path)

            low_output_path = tmp_path / "low_quality.png"
            high_output_path = tmp_path / "high_quality.png" if legacy_requested else None
            video_jobs: List[Tuple[str, str, Path, Path, str]] = []
            if video_low_requested:
                video_jobs.append(
                    (
                        "fast",
                        video_low_framing,
                        tmp_path / "video_fast_frame.png",
                        tmp_path / "video_fast.mp4",
                        "low-quality",
                    )
                )
            if video_high_requested:
                video_jobs.append(
                    (
                        "high",
                        video_high_framing,
                        tmp_path / "video_high_frame.png",
                        tmp_path / "video_high.mp4",
                        "high-quality",
                    )
                )
            if video_ultra_requested:
                video_jobs.append(
                    (
                        "ultra",
                        video_ultra_framing,
                        tmp_path / "video_ultra_frame.png",
                        tmp_path / "video_ultra.mp4",
                        "ultra-high-quality",
                    )
                )

            still_render_elapsed = None
            if fast_requested or legacy_requested:
                low_resolution = (
                    image_options.low_quality_resolution
                    or image_options.resolution
                    or DEFAULT_LOW_QUALITY_RESOLUTION
                )
                high_resolution = (
                    image_options.high_quality_resolution
                    or image_options.legacy_resolution
                    or DEFAULT_HIGH_QUALITY_RESOLUTION
                )
                low_samples = (
                    image_options.low_quality_samples
                    or image_options.fast_samples
                    or DEFAULT_LOW_QUALITY_SAMPLES
                )
                high_samples = (
                    image_options.high_quality_samples
                    or image_options.legacy_samples
                    or DEFAULT_HIGH_QUALITY_SAMPLES
                )
                try:
                    still_render_elapsed = run_blender_render(
                        str(local_input_path),
                        str(low_output_path),
                        resolution=low_resolution,
                        camera=image_options.camera or DEFAULT_CAMERA,
                        thread_thickness=image_options.thread_thickness or DEFAULT_THREAD_THICKNESS,
                        fast_samples=low_samples,
                        legacy_output_path=str(high_output_path) if high_output_path else None,
                        legacy_resolution=high_resolution if legacy_requested else None,
                        legacy_samples=high_samples if legacy_requested else None,
                    )
                    logger.info("job %s: still render completed in %.2fs", job_id, still_render_elapsed)
                except RuntimeError as exc:
                    logger.exception("job %s: still render failed", job_id)
                    errors.append(f"still_render: {exc}")
                else:
                    if fast_requested:
                        stage_metrics["low_quality_png_seconds"] = still_render_elapsed
                    if legacy_requested:
                        stage_metrics["high_quality_png_seconds"] = still_render_elapsed

                    background_choice = image_options.background or "white"
                    if fast_requested and low_output_path.exists():
                        applied = apply_background_choice(
                            str(low_output_path),
                            background_choice,
                            image_options.background_tolerance,
                        )
                        logger.info("job %s: applied background '%s' to low-quality render", job_id, applied)
                    if legacy_requested and high_output_path and high_output_path.exists():
                        applied = apply_background_choice(
                            str(high_output_path),
                            background_choice,
                            image_options.background_tolerance,
                        )
                        logger.info("job %s: applied background '%s' to high-quality render", job_id, applied)

                    if fast_requested:
                        if not low_output_path.exists():
                            errors.append("low_quality_render: output not produced")
                        else:
                            low_backup = Path(
                                _dedupe_path(
                                    os.path.join(
                                        processed_dir, f"{safe_base_prefix}_{timestamp_suffix}_low_quality.png"
                                    )
                                )
                            )
                            shutil.copyfile(low_output_path, low_backup)
                            logger.info("job %s: backed up low-quality render to %s", job_id, low_backup)
                            low_quality_backup_path = low_backup
                            low_filename = f"{safe_base_prefix}_low_quality.png"
                            low_key = compose_key(compose_base_path, low_filename)
                            try:
                                upload_result = upload_with_suffix_on_conflict(
                                    output_bucket, low_key, low_output_path
                                )
                                uploaded_assets["low_quality_png"] = AssetSummary(
                                    object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                                )
                                elapsed = stage_metrics.get("low_quality_png_seconds") or still_render_elapsed
                                msg = (
                                    f"[job {job_id}] low-quality PNG ready in {elapsed:.2f}s -> {upload_result['final_key']}"
                                    if elapsed
                                    else f"[job {job_id}] low-quality PNG uploaded -> {upload_result['final_key']}"
                                )
                                print(msg, flush=True)
                                logger.info(msg)
                                logger.info(
                                    "job %s: uploaded low-quality output to %s/%s",
                                    job_id,
                                    output_bucket,
                                    upload_result["final_key"],
                                )
                            except SpacesCollisionError as exc:
                                logger.error("job %s: collision uploading low-quality render: %s", job_id, exc)
                                raise HTTPException(
                                    status_code=409,
                                    detail={"job_id": job_id, "message": str(exc)},
                                ) from exc
                            except ClientError as exc:
                                logger.exception("job %s: failed to upload low-quality render", job_id)
                                errors.append(f"low_quality_upload: {exc}")

                            canvas_overlay_path = tmp_path / "low_quality_canvas.png"
                            canvas_result = _generate_canvas_overlay(low_output_path, canvas_overlay_path, canvas_tolerance)
                            if canvas_result and canvas_result.exists():
                                try:
                                    canvas_backup = Path(
                                        _dedupe_path(
                                            os.path.join(
                                                processed_dir,
                                                f"{safe_base_prefix}_{timestamp_suffix}_low_quality_canvas.png",
                                            )
                                        )
                                    )
                                    shutil.copyfile(canvas_result, canvas_backup)
                                    canvas_filename = f"{safe_base_prefix}_low_quality_canvas.png"
                                    canvas_key = compose_key(compose_base_path, canvas_filename)
                                    upload_result = upload_with_suffix_on_conflict(
                                        output_bucket, canvas_key, canvas_result
                                    )
                                    uploaded_assets["low_quality_canvas_png"] = AssetSummary(
                                        object_key=upload_result["final_key"],
                                        size_bytes=upload_result["size_bytes"],
                                    )
                                    logger.info(
                                        "job %s: uploaded low-quality canvas output to %s/%s",
                                        job_id,
                                        output_bucket,
                                        upload_result["final_key"],
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "job %s: failed to upload low-quality canvas overlay: %s",
                                        job_id,
                                        exc,
                                    )

                    if legacy_requested and high_output_path:
                        if not high_output_path.exists():
                            errors.append("high_quality_render: output not produced")
                        else:
                            high_backup = Path(
                                _dedupe_path(
                                    os.path.join(
                                        processed_dir, f"{safe_base_prefix}_{timestamp_suffix}_high_quality.png"
                                    )
                                )
                            )
                            shutil.copyfile(high_output_path, high_backup)
                            logger.info("job %s: backed up high-quality render to %s", job_id, high_backup)
                            high_quality_backup_path = high_backup
                            high_filename = f"{safe_base_prefix}_high_quality.png"
                            high_key = compose_key(compose_base_path, high_filename)
                            try:
                                upload_result = upload_with_suffix_on_conflict(
                                    output_bucket, high_key, high_output_path
                                )
                                uploaded_assets["high_quality_png"] = AssetSummary(
                                    object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                                )
                                elapsed = stage_metrics.get("high_quality_png_seconds") or still_render_elapsed
                                msg = (
                                    f"[job {job_id}] high-quality PNG ready in {elapsed:.2f}s -> {upload_result['final_key']}"
                                    if elapsed
                                    else f"[job {job_id}] high-quality PNG uploaded -> {upload_result['final_key']}"
                                )
                                print(msg, flush=True)
                                logger.info(msg)
                                logger.info(
                                    "job %s: uploaded high-quality output to %s/%s",
                                    job_id,
                                    output_bucket,
                                    upload_result["final_key"],
                                )
                            except SpacesCollisionError as exc:
                                logger.error("job %s: collision uploading high-quality render: %s", job_id, exc)
                                raise HTTPException(
                                    status_code=409,
                                    detail={"job_id": job_id, "message": str(exc)},
                                ) from exc
                            except ClientError as exc:
                                logger.exception("job %s: failed to upload high-quality render", job_id)
                                errors.append(f"high_quality_upload: {exc}")

                        canvas_high_path = tmp_path / "high_quality_canvas.png"
                        canvas_high_result = _generate_canvas_overlay(
                            high_output_path, canvas_high_path, canvas_tolerance
                        )
                        if canvas_high_result and canvas_high_result.exists():
                            try:
                                canvas_high_backup = Path(
                                    _dedupe_path(
                                        os.path.join(
                                            processed_dir,
                                            f"{safe_base_prefix}_{timestamp_suffix}_high_quality_canvas.png",
                                        )
                                    )
                                )
                                shutil.copyfile(canvas_high_result, canvas_high_backup)
                                canvas_high_filename = f"{safe_base_prefix}_high_quality_canvas.png"
                                canvas_high_key = compose_key(compose_base_path, canvas_high_filename)
                                upload_result = upload_with_suffix_on_conflict(
                                    output_bucket, canvas_high_key, canvas_high_result
                                )
                                uploaded_assets["high_quality_canvas_png"] = AssetSummary(
                                    object_key=upload_result["final_key"],
                                    size_bytes=upload_result["size_bytes"],
                                )
                                logger.info(
                                    "job %s: uploaded high-quality canvas output to %s/%s",
                                    job_id,
                                    output_bucket,
                                    upload_result["final_key"],
                                )
                            except Exception as exc:
                                logger.warning(
                                    "job %s: failed to upload high-quality canvas overlay: %s",
                                    job_id,
                                    exc,
                                )

            if video_jobs:
                total_video_jobs = len(video_jobs)
                completed_video_jobs = 0
                for quality, framing, frame_path, video_path, label in video_jobs:
                    try:
                        percent_complete = (completed_video_jobs / total_video_jobs) * 100
                        print(
                            f"[job {job_id}] Video progress: {percent_complete:.0f}% -> starting {label} ({framing})",
                            flush=True,
                        )
                        video_elapsed = run_blender_video(
                            str(local_input_path),
                            str(frame_path),
                            str(video_path),
                            video_quality=quality,
                            video_framing=framing,
                        )
                        if quality == "fast":
                            key = "video_low_seconds"
                        elif quality == "high":
                            key = "video_high_seconds"
                        else:
                            key = "video_ultra_seconds"
                        stage_metrics[key] = video_elapsed
                        stage_metrics["video_seconds"] = video_elapsed
                        msg = (
                            f"[job {job_id}] {label} ({framing}) video render completed in {video_elapsed:.2f}s -> {video_path}"
                        )
                        print(msg, flush=True)
                        logger.info(msg)
                        completed_video_jobs += 1
                        percent_complete = (completed_video_jobs / total_video_jobs) * 100
                        print(
                            f"[job {job_id}] Video progress: {percent_complete:.0f}% complete",
                            flush=True,
                        )
                    except RuntimeError as exc:
                        logger.exception("job %s: video render (%s) failed", job_id, quality)
                        errors.append(f"video_render_{label.replace('-', '_')}: {exc}")
                        continue

                    if not video_path.exists():
                        errors.append(f"video_render_{label.replace('-', '_')}: output not produced")
                        continue

                    video_backup = Path(
                        _dedupe_path(
                            os.path.join(
                                processed_dir,
                                f"{safe_base_prefix}_{timestamp_suffix}_video_{label.replace('-', '_')}.mp4",
                            )
                        )
                    )
                    shutil.copyfile(video_path, video_backup)
                    logger.info("job %s: backed up %s video render to %s", job_id, label, video_backup)
                    video_backup_path = video_backup
                    filename = f"{safe_base_prefix}_video_{label.replace('-', '_')}.mp4"
                    video_key = compose_key(compose_base_path, filename)
                    try:
                        upload_result = upload_with_suffix_on_conflict(
                            output_bucket, video_key, video_path
                        )
                        if quality == "fast":
                            summary_key = "video_low_mp4"
                        elif quality == "high":
                            summary_key = "video_high_mp4"
                        else:
                            summary_key = "video_ultra_mp4"
                        uploaded_assets[summary_key] = AssetSummary(
                            object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                        )
                        # Maintain legacy field for compatibility (prefer high quality)
                        if (
                            quality in ("high", "ultra")
                            or "video_mp4" not in uploaded_assets
                        ):
                            uploaded_assets["video_mp4"] = uploaded_assets[summary_key]
                        upload_msg = (
                            f"[job {job_id}] {label} video uploaded -> {upload_result['final_key']}"
                        )
                        print(upload_msg, flush=True)
                        logger.info(upload_msg)
                        logger.info(
                            "job %s: uploaded %s video output to %s/%s",
                            job_id,
                            label,
                            output_bucket,
                            upload_result["final_key"],
                        )
                    except SpacesCollisionError as exc:
                        logger.error("job %s: collision uploading %s video render: %s", job_id, quality, exc)
                        raise HTTPException(
                            status_code=409,
                            detail={"job_id": job_id, "message": str(exc)},
                        ) from exc
                    except ClientError as exc:
                        logger.exception("job %s: failed to upload %s video render", job_id, quality)
                        errors.append(f"video_upload_{quality}: {exc}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("job %s: unexpected error during render-from-space", job_id)
        raise HTTPException(
            status_code=500, detail={"job_id": job_id, "message": "Unexpected error processing render request."}
        ) from exc

    total_elapsed = time.perf_counter() - overall_start
    stage_metrics_model = StageMetrics(**stage_metrics)

    debug_payload: Optional[Dict[str, Any]] = None
    if debug_enabled:
        uploaded_assets_dump = {key: value.dict() for key, value in uploaded_assets.items()}
        stage_metrics_dump = stage_metrics_model.dict(exclude_none=True)
        debug_payload = {
            "job_id": job_id,
            "input_bucket": request_model.input.bucket_name,
            "input_object_key": input_key,
            "input_size_bytes": input_size,
            "input_backup": _safe_relpath(input_backup_path),
            "low_quality_backup": _safe_relpath(low_quality_backup_path),
            "high_quality_backup": _safe_relpath(high_quality_backup_path),
            "video_backup": _safe_relpath(video_backup_path),
            "output_bucket": output_bucket,
            "base_path": final_base_path,
            "compose_base_path": compose_base_path,
            "stage_metrics": stage_metrics_dump,
            "uploaded_assets": uploaded_assets_dump,
            "errors": list(errors),
            "service_log_tail": _read_log_tail(),
        }
        debug_payload = {key: value for key, value in debug_payload.items() if value not in (None, {}, [])}

    completion_message = _build_completion_message(
        request_model,
        input_key=input_key,
        total_elapsed=total_elapsed,
        stage_metrics=stage_metrics_model,
        uploaded_assets=uploaded_assets,
        output_bucket=output_bucket,
        errors=errors,
        debug_enabled=debug_enabled,
    )
    response = RenderFromSpaceResponse(
        job_id=job_id,
        input=InputSummary(
            bucket_name=request_model.input.bucket_name,
            object_key=input_key,
            size_bytes=input_size,
        ),
        outputs=OutputSummary(
            bucket_name=output_bucket,
            base_path=final_base_path,
            low_quality_png=uploaded_assets.get("low_quality_png"),
            low_quality_canvas_png=uploaded_assets.get("low_quality_canvas_png"),
            high_quality_png=uploaded_assets.get("high_quality_png"),
            high_quality_canvas_png=uploaded_assets.get("high_quality_canvas_png"),
            video_low_mp4=uploaded_assets.get("video_low_mp4"),
            video_high_mp4=uploaded_assets.get("video_high_mp4"),
            video_ultra_mp4=uploaded_assets.get("video_ultra_mp4"),
            video_mp4=uploaded_assets.get("video_mp4"),
        ),
        metrics=RenderMetrics(render_time_seconds_total=total_elapsed, stages=stage_metrics_model),
        message=completion_message,
        debug=debug_payload,
        errors=errors,
    )
    logger.info(
        "job %s: completed render-from-space (total %.2fs, errors=%s)",
        job_id,
        total_elapsed,
        bool(errors),
    )
    return response.dict(exclude_none=True)


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
    include_video: Optional[str] = Form(None),
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

    include_video_flag = False
    if include_video is not None:
        include_video_flag = include_video.strip().lower() in {"1", "true", "yes", "on"}

    effective_resolution = resolution or DEFAULT_RESOLUTION
    legacy_requested = requested_mode in {"legacy", "both"}

    input_backup = None
    fast_backup = None
    legacy_backup = None
    video_backup = None
    timestamp = int(time.time())
    elapsed = 0.0
    video_elapsed = 0.0

    with tempfile.TemporaryDirectory(prefix="embroidery_render_") as tmpdir:
        input_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{extension}")
        fast_output_path = os.path.join(tmpdir, "fast.png")
        legacy_output_path = os.path.join(tmpdir, "legacy.png") if legacy_requested else None
        video_temp_image_path = os.path.join(tmpdir, "video_frame.png") if include_video_flag else None
        video_output_path = os.path.join(tmpdir, "render.mp4") if include_video_flag else None

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

        if include_video_flag:
            if not video_temp_image_path:
                video_temp_image_path = os.path.join(tmpdir, "video_frame.png")
            if not video_output_path:
                video_output_path = os.path.join(tmpdir, "render.mp4")
            try:
                video_elapsed = run_blender_video(input_path, video_temp_image_path, video_output_path)
            except RuntimeError as exc:
                logger.exception("Video render failed for '%s'", file.filename)
                raise HTTPException(status_code=500, detail=f"Video render failed: {exc}") from exc

            if not os.path.exists(video_output_path):
                logger.error("Video render output missing at %s", video_output_path)
                raise HTTPException(status_code=500, detail="Video render completed but no video was generated.")

            video_backup = _dedupe_path(os.path.join(processed_dir, f"{safe_base}_{timestamp}_video.mp4"))
            shutil.copyfile(video_output_path, video_backup)
            logger.info("Backed up video output to %s", video_backup)

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
    if video_backup:
        headers["X-Video-Output-Backup"] = os.path.relpath(video_backup, repo_dir)
    if include_video_flag:
        headers["X-Video-Render-Time"] = f"{video_elapsed:.2f}"

    needs_archive = include_video_flag or requested_mode == "both"

    if needs_archive:
        if include_video_flag and not video_backup:
            raise HTTPException(status_code=500, detail="Video render requested but no video was produced.")
        archive_name = f"{safe_base}_{timestamp}.zip"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if requested_mode in {"fast", "both"}:
                if not fast_backup or not os.path.exists(fast_backup):
                    raise HTTPException(status_code=500, detail="Fast render completed but could not locate the output file.")
                archive.write(fast_backup, arcname=f"{safe_base}_fast.png")
            if requested_mode in {"legacy", "both"}:
                if not legacy_backup or not os.path.exists(legacy_backup):
                    raise HTTPException(status_code=500, detail="Legacy render requested but no image was produced.")
                archive.write(legacy_backup, arcname=f"{safe_base}_legacy.png")
            if include_video_flag and video_backup and os.path.exists(video_backup):
                archive.write(video_backup, arcname=f"{safe_base}_video.mp4")
        zip_bytes = zip_buffer.getvalue()
        archive_backup = _dedupe_path(os.path.join(processed_dir, archive_name))
        with open(archive_backup, "wb") as archive_file:
            archive_file.write(zip_bytes)
        headers["X-Archive-Backup"] = os.path.relpath(archive_backup, repo_dir)
        headers["Content-Disposition"] = f'attachment; filename="{archive_name}"'
        return StreamingResponse(io.BytesIO(zip_bytes), media_type="application/zip", headers=headers)

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
        "example_fast_with_video": 'curl -F "mode=fast" -F "include_video=true" -F "file=@path/to/design.PES" http://localhost:8000/render --output design_bundle.zip',
    }
