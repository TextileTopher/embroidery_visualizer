## Embroidery Render API

This FastAPI service receives an embroidery upload (`.pes`) and renders one or
two still PNGs with Blender. Every request is backed up to
`processed_files/` so you always retain the original PES and the generated
renders.

### Prerequisites

- Blender available at `/opt/blender/blender` with `BlenderSetup.blend` and the
  bundled scripts left in this repository.
- Python 3.10+ with the runtime dependencies listed in `requirements.txt`.
- Additional packages for the API layer:
  ```bash
  pip install fastapi uvicorn python-multipart
  ```
- The Blender-side importer (`ImporterScript`) already configured for your
  environment.
- DigitalOcean Spaces credentials stored in `embroidery_visualizer/.env`:
  ```
  SPACES_ENDPOINT=https://sfo3.digitaloceanspaces.com
  SPACES_REGION=sfo3
  SPACES_ACCESS_KEY=your_access_key
  SPACES_SECRET_KEY=your_secret_key
  ```
  `render_service.py` automatically loads `.env` (and `.venv/.env` when present)
  at startup, so Uvicorn can be launched without exporting variables manually.

### Recommended Environment Setup

```bash
cd /home/topher/embroidery_visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### Start the Service

From the repository root:

```bash
uvicorn render_service:app --host 0.0.0.0 --port 8000
```

- Keep this terminal open; press `Ctrl+C` to stop the server.
- `--host 0.0.0.0` allows machines on your local network to reach the service.
- Blender runs headlessly per request, so no long-lived Blender daemon needs to
  be managed.

### JSON Workflow (DigitalOcean Spaces)

`POST /render-from-space` downloads a PES file from DigitalOcean Spaces,
executes the requested renders, and uploads the outputs back to Spaces.

```json
{
  "input": {
    "bucket_name": "digitizer",
    "file_path": "test",
    "file_name": "shark_colored.pes"
  },
  "operations": {
    "generate_low": true,
    "generate_high": false,
    "generate_video": false
  },
  "output": {
    "bucket_name": "digitizer",
    "base_path": "test"
  },
  "render_options": {
    "resolution": 1024,
    "legacy_resolution": 2048,
    "fast_samples": 128,
    "legacy_samples": 512,
    "camera": "TopView",
    "thread_thickness": 0.2
  }
}
```

- `input.bucket_name`, `input.file_name`, and `output.bucket_name` are required.
- `file_path` and `base_path` are optional; defaults place outputs alongside the
  input asset.
- `file_name` must end with `.pes` (case-insensitive).
- At least one of `generate_low` or `generate_high` must be `true`.
- Objects larger than 50 MB are rejected with HTTP 413 (size checked via
  `head_object` before download).
- Outputs upload immediately after each render finishes. If a key exists, the
  service appends a UTC timestamp suffix before the extension; if a unique name
  still cannot be produced, the call fails with HTTP 409.

Response payload:

```json
{
  "job_id": "uuid",
  "input": {
    "bucket_name": "digitizer",
    "object_key": "test/shark_colored.pes",
    "size_bytes": 229134
  },
  "outputs": {
    "bucket_name": "digitizer",
    "base_path": "test",
    "fast_png": {
      "object_key": "test/shark_colored_fast.png",
      "size_bytes": 4202718
    }
  },
  "metrics": {
    "render_time_seconds_total": 13.0,
    "stages": {
      "fast_png_seconds": 12.5
    }
  },
  "errors": []
}
```

- Only successful assets appear under `outputs`; failures are described in the
  `errors` array (e.g. `"video_upload: AccessDenied"`).
- `metrics.stages` includes per-render timing when available.
- `job_id` is logged with every line in `processed_files/render_service.log`.

### Request Workflow

`POST /render` accepts a `multipart/form-data` payload with the uploaded PES
file and optional render controls. Include the following fields:

| Field              | Required | Type      | Default | Description |
|--------------------|----------|-----------|---------|-------------|
| `file`             | Yes      | File      | –       | PES file to render. |
| `mode`             | No       | Text      | `fast`  | `fast`, `legacy`, or `both`. |
| `resolution`       | No       | Integer   | `1024`  | Resolution for the fast render. |
| `camera`           | No       | Text      | `TopView` | Camera object in the blend file. |
| `thread_thickness` | No       | Float     | `0.2`   | Thread thickness passed to the importer. |
| `fast_samples`     | No       | Integer   | Blender default | Cycles samples override for the fast render. |
| `legacy_resolution`| No       | Integer   | `resolution * 2` | Optional override for the legacy render resolution. |
| `legacy_samples`   | No       | Integer   | Blender default | Cycles samples override for the legacy render. |
| `include_video`    | No       | Text      | `false` | Set to `true` to request a rendered MP4 animation. |

> **Tip:** When `mode=legacy` the fast render still runs under the hood, but the
> service only streams the legacy PNG back. `mode=both` returns a `.zip`
> containing both outputs.

If a field is omitted the service uses the defaults listed above. Validation is
handled by FastAPI—invalid values return HTTP 422.

### Response Behaviour

- `mode=fast` → Response is `image/png` containing the fast render.
- `mode=legacy` → Response is `image/png` containing the legacy render.
- `mode=both` → Response is `application/zip` with `*_fast.png` and
  `*_legacy.png` entries.
- Adding `include_video=true` returns an `application/zip` bundle containing
  the requested still(s) plus `*_video.mp4`. The service also stores each asset
  (PNG/MP4/ZIP) individually inside `processed_files/` for auditing.

Common response headers:

- `Content-Disposition` – Suggested filename for the downloaded asset.
- `X-Render-Time` – Total seconds spent inside Blender.
- `X-Mode` – The mode that was executed (`fast`, `legacy`, `both`).
- `X-Input-Backup` – Relative path to the stored PES copy in `processed_files/`.
- `X-Fast-Output-Backup` – Relative path to the fast PNG backup (if produced).
- `X-Legacy-Output-Backup` – Relative path to the legacy PNG backup (if
  produced).
- `X-Video-Output-Backup` – Relative path to the MP4 backup when `include_video`
  is enabled.
- `X-Archive-Backup` – Relative path to the ZIP bundle stored for reference
  when a ZIP response is returned.

Every upload persists to disk alongside the renders (deduplicated with a
timestamp suffix) so you can re-download the originals later.

### Sample Requests

```bash
# Fast render only (default)
curl -F "file=@input_PES/cat.PES" \
     -F "mode=fast" \
     http://SERVER_IP:8000/render \
     --output cat_fast.png

# Legacy render (streams the legacy PNG back)
curl -F "file=@input_PES/cat.PES" \
     -F "mode=legacy" \
     -F "legacy_resolution=2048" \
     http://SERVER_IP:8000/render \
     --output cat_legacy.png

# Both renders packaged into a zip
curl -F "file=@input_PES/cat.PES" \
     -F "mode=both" \
     http://SERVER_IP:8000/render \
     --output cat_outputs.zip

# Fast render plus video bundle
curl -F "file=@input_PES/cat.PES" \
     -F "mode=fast" \
     -F "include_video=true" \
     http://SERVER_IP:8000/render \
     --output cat_fast_with_video.zip
```

To customise sampling:

```bash
curl -F "file=@input_PES/cat.PES" \
     -F "mode=both" \
     -F "fast_samples=128" \
     -F "legacy_samples=512" \
     http://SERVER_IP:8000/render \
     --output cat_outputs.zip
```

### Programmatic Example

```python
import requests

with open("cat.PES", "rb") as pes:
    response = requests.post(
        "http://SERVER_IP:8000/render",
        files={"file": pes},
        data={"mode": "fast"},
    )
response.raise_for_status()

with open("cat_fast.png", "wb") as out_file:
    out_file.write(response.content)

print("Render mode:", response.headers.get("X-Mode"))
print("Render time:", response.headers.get("X-Render-Time", "n/a"), "seconds")
print("Input backup:", response.headers.get("X-Input-Backup"))
print("Video backup:", response.headers.get("X-Video-Output-Backup"))
print("Video render time:", response.headers.get("X-Video-Render-Time"))
print("Archive backup:", response.headers.get("X-Archive-Backup"))
```

### Additional Endpoints

- `GET /` – Describes the service and provides handy `curl` recipes.
- `GET /health` – Returns `{"status": "ok"}` for probes.
- `GET /docs` – Swagger UI for manual testing.

### Logs and Diagnostics

- `processed_files/render_service.log` captures Blender stdout/stderr per job.
- Uploaded PES files and generated PNGs are written to `processed_files/` with
  deduplicated timestamps (e.g., `design_1717251234_fast.png`).

### Shutdown

Press `Ctrl+C` in the terminal running Uvicorn. Ensure the process exits
cleanly before restarting to avoid lingering Blender child processes.
