import io
import logging
import os
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
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv(dotenv_path=os.path.join(repo_dir, ".env"))
    load_dotenv(dotenv_path=os.path.join(repo_dir, ".venv", ".env"))

BLENDER_BINARY = "/opt/blender/blender"
DEFAULT_RESOLUTION = 1024
DEFAULT_CAMERA = "TopView"
DEFAULT_THREAD_THICKNESS = 0.2

repo_dir = os.path.dirname(os.path.abspath(__file__))
blend_file = os.path.join(repo_dir, "BlenderSetup.blend")
blender_still_script = os.path.join(repo_dir, "blender_render_still.py")
blender_video_script = os.path.join(repo_dir, "blender_script.py")
processed_dir = os.path.join(repo_dir, "processed_files")
service_log = os.path.join(processed_dir, "render_service.log")

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


def run_blender_video(input_path: str, temp_image_path: str, output_video_path: str) -> float:
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
    generate_low: bool = False
    generate_high: bool = False
    generate_video: bool = False

    @model_validator(mode="after")
    def _ensure_primary_render(self):
        if not (self.generate_low or self.generate_high):
            raise ValueError("At least one of generate_low or generate_high must be true.")
        return self


class RenderOptions(BaseModel):
    resolution: Optional[int] = Field(default=None, gt=0)
    legacy_resolution: Optional[int] = Field(default=None, gt=0)
    fast_samples: Optional[int] = Field(default=None, gt=0)
    legacy_samples: Optional[int] = Field(default=None, gt=0)
    camera: Optional[str] = Field(default=None, min_length=1)
    thread_thickness: Optional[float] = Field(default=None, gt=0)


class RenderFromSpaceRequest(BaseModel):
    input: SpacesInputSpec
    operations: OperationFlags
    output: SpacesOutputSpec
    render_options: Optional[RenderOptions] = None
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
    fast_png: Optional[AssetSummary] = None
    legacy_png: Optional[AssetSummary] = None
    video_mp4: Optional[AssetSummary] = None


class StageMetrics(BaseModel):
    fast_png_seconds: Optional[float] = None
    legacy_png_seconds: Optional[float] = None
    video_seconds: Optional[float] = None


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
        "fast_png": "Fast PNG",
        "legacy_png": "Legacy PNG",
        "video_mp4": "Video MP4",
    }
    stage_lookup = {
        "fast_png": stage_metrics.fast_png_seconds,
        "legacy_png": stage_metrics.legacy_png_seconds,
        "video_mp4": stage_metrics.video_seconds,
    }

    for key, label in asset_labels.items():
        asset = uploaded_assets.get(key)
        if not asset:
            continue
        lines.extend(
            [
                f"{label} return file name: {Path(asset.object_key).name}",
                f"{label} return file location: {asset.object_key}",
                f"{label} return bucket: {output_bucket}",
                f"{label} processing time: {_format_duration(stage_lookup.get(key))}",
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
    fast_backup_path: Optional[Path] = None
    legacy_backup_path: Optional[Path] = None
    video_backup_path: Optional[Path] = None

    overall_start = time.perf_counter()
    stage_metrics: Dict[str, float] = {}
    errors: List[str] = []
    uploaded_assets: Dict[str, AssetSummary] = {}
    safe_base_prefix = _safe_backup_prefix(request_model.input.file_name)
    timestamp_suffix = int(time.time())

    fast_requested = request_model.operations.generate_low
    legacy_requested = request_model.operations.generate_high
    video_requested = request_model.operations.generate_video
    options = request_model.render_options or RenderOptions()

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

            fast_output_path = tmp_path / "fast.png"
            legacy_output_path = tmp_path / "legacy.png" if legacy_requested else None
            video_frame_path = tmp_path / "video_frame.png" if video_requested else None
            video_output_path = tmp_path / "render.mp4" if video_requested else None

            still_render_elapsed = None
            if fast_requested or legacy_requested:
                try:
                    still_render_elapsed = run_blender_render(
                        str(local_input_path),
                        str(fast_output_path),
                        resolution=options.resolution or DEFAULT_RESOLUTION,
                        camera=options.camera or DEFAULT_CAMERA,
                        thread_thickness=options.thread_thickness or DEFAULT_THREAD_THICKNESS,
                        fast_samples=options.fast_samples,
                        legacy_output_path=str(legacy_output_path) if legacy_output_path else None,
                        legacy_resolution=options.legacy_resolution,
                        legacy_samples=options.legacy_samples,
                    )
                    logger.info("job %s: still render completed in %.2fs", job_id, still_render_elapsed)
                except RuntimeError as exc:
                    logger.exception("job %s: still render failed", job_id)
                    errors.append(f"still_render: {exc}")
                else:
                    if fast_requested:
                        stage_metrics["fast_png_seconds"] = still_render_elapsed
                    if legacy_requested:
                        stage_metrics["legacy_png_seconds"] = still_render_elapsed

                    if fast_requested:
                        if not fast_output_path.exists():
                            errors.append("fast_render: output not produced")
                        else:
                            fast_backup = Path(
                                _dedupe_path(
                                    os.path.join(processed_dir, f"{safe_base_prefix}_{timestamp_suffix}_fast.png")
                                )
                            )
                            shutil.copyfile(fast_output_path, fast_backup)
                            logger.info("job %s: backed up fast render to %s", job_id, fast_backup)
                            fast_backup_path = fast_backup
                            fast_filename = f"{safe_base_prefix}_fast.png"
                            fast_key = compose_key(compose_base_path, fast_filename)
                            try:
                                upload_result = upload_with_suffix_on_conflict(
                                    output_bucket, fast_key, fast_output_path
                                )
                                uploaded_assets["fast_png"] = AssetSummary(
                                    object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                                )
                                logger.info(
                                    "job %s: uploaded fast output to %s/%s",
                                    job_id,
                                    output_bucket,
                                    upload_result["final_key"],
                                )
                            except SpacesCollisionError as exc:
                                logger.error("job %s: collision uploading fast render: %s", job_id, exc)
                                raise HTTPException(
                                    status_code=409,
                                    detail={"job_id": job_id, "message": str(exc)},
                                ) from exc
                            except ClientError as exc:
                                logger.exception("job %s: failed to upload fast render", job_id)
                                errors.append(f"fast_upload: {exc}")

                    if legacy_requested and legacy_output_path:
                        if not legacy_output_path.exists():
                            errors.append("legacy_render: output not produced")
                        else:
                            legacy_backup = Path(
                                _dedupe_path(
                                    os.path.join(processed_dir, f"{safe_base_prefix}_{timestamp_suffix}_legacy.png")
                                )
                            )
                            shutil.copyfile(legacy_output_path, legacy_backup)
                            logger.info("job %s: backed up legacy render to %s", job_id, legacy_backup)
                            legacy_backup_path = legacy_backup
                            legacy_filename = f"{safe_base_prefix}_legacy.png"
                            legacy_key = compose_key(compose_base_path, legacy_filename)
                            try:
                                upload_result = upload_with_suffix_on_conflict(
                                    output_bucket, legacy_key, legacy_output_path
                                )
                                uploaded_assets["legacy_png"] = AssetSummary(
                                    object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                                )
                                logger.info(
                                    "job %s: uploaded legacy output to %s/%s",
                                    job_id,
                                    output_bucket,
                                    upload_result["final_key"],
                                )
                            except SpacesCollisionError as exc:
                                logger.error("job %s: collision uploading legacy render: %s", job_id, exc)
                                raise HTTPException(
                                    status_code=409,
                                    detail={"job_id": job_id, "message": str(exc)},
                                ) from exc
                            except ClientError as exc:
                                logger.exception("job %s: failed to upload legacy render", job_id)
                                errors.append(f"legacy_upload: {exc}")

            if video_requested and video_output_path and video_frame_path:
                try:
                    video_elapsed = run_blender_video(str(local_input_path), str(video_frame_path), str(video_output_path))
                    stage_metrics["video_seconds"] = video_elapsed
                    logger.info("job %s: video render completed in %.2fs", job_id, video_elapsed)
                except RuntimeError as exc:
                    logger.exception("job %s: video render failed", job_id)
                    errors.append(f"video_render: {exc}")
                else:
                    if not video_output_path.exists():
                        errors.append("video_render: output not produced")
                    else:
                        video_backup = Path(
                            _dedupe_path(
                                os.path.join(processed_dir, f"{safe_base_prefix}_{timestamp_suffix}_video.mp4")
                            )
                        )
                        shutil.copyfile(video_output_path, video_backup)
                        logger.info("job %s: backed up video render to %s", job_id, video_backup)
                        video_backup_path = video_backup
                        video_filename = f"{safe_base_prefix}_video.mp4"
                        video_key = compose_key(compose_base_path, video_filename)
                        try:
                            upload_result = upload_with_suffix_on_conflict(
                                output_bucket, video_key, video_output_path
                            )
                            uploaded_assets["video_mp4"] = AssetSummary(
                                object_key=upload_result["final_key"], size_bytes=upload_result["size_bytes"]
                            )
                            logger.info(
                                "job %s: uploaded video output to %s/%s",
                                job_id,
                                output_bucket,
                                upload_result["final_key"],
                            )
                        except SpacesCollisionError as exc:
                            logger.error("job %s: collision uploading video render: %s", job_id, exc)
                            raise HTTPException(
                                status_code=409,
                                detail={"job_id": job_id, "message": str(exc)},
                            ) from exc
                        except ClientError as exc:
                            logger.exception("job %s: failed to upload video render", job_id)
                            errors.append(f"video_upload: {exc}")
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
            "fast_backup": _safe_relpath(fast_backup_path),
            "legacy_backup": _safe_relpath(legacy_backup_path),
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
            fast_png=uploaded_assets.get("fast_png"),
            legacy_png=uploaded_assets.get("legacy_png"),
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
