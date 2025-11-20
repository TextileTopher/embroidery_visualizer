# Embroidery Visualizer

Headless Blender workflow for converting Brother PES embroidery files into photorealistic stills and fly-through videos. The repository contains the Blender scripts, an HTTP render service, and a handful of CLI helpers that all share the same `BlenderSetup.blend` scene.

---

## 1. Prepare the Server

### Prerequisites

- **Blender 4.3** installed at `/opt/blender/blender` (the scripts expect the bundled Python runtime at `/opt/blender/4.3/python/bin/python3.11`).
- **DigitalOcean Spaces (S3-compatible)** bucket that stores your PES uploads (example bucket `digitizer` with `test/cat.PES` inside).
- **Python 3.10+** for running the FastAPI service.

### Clone & Install

```bash
git clone https://github.com/<your-org>/embroidery_visualizer.git
cd embroidery_visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure Credentials

Create `.env` in the repo root (the service also checks `.venv/.env`) with either DigitalOcean or AWS-style keys. Example:

```ini
SPACES_ENDPOINT=https://sfo3.digitaloceanspaces.com
SPACES_REGION=sfo3
SPACES_ACCESS_KEY=DOxxxxxxxxxxxx
SPACES_SECRET_KEY=xxxxxxxxxxxxxxxxxxx
```

> The regional endpoint (`https://sfo3.digitaloceanspaces.com`) is required—CDN URLs such as `https://digitizer.sfo3.cdn.digitaloceanspaces.com` cannot serve S3 API requests.

---

## 2. Start the Render Service

Run Uvicorn from the project root (inside the virtualenv):

```bash
source .venv/bin/activate
uvicorn render_service:app --host 0.0.0.0 --port 9009
```

- Console output and Blender logs are written to `processed_files/render_service.log`.
- Every request stores the original PES and all generated assets under `processed_files/` for auditing.

Stop the server with `Ctrl+C` when finished.

---

## 3. Trigger a Test Render (cat.PES)

Assuming `cat.PES` lives in `digitizer/test/cat.PES` (downloadable from `https://digitizer.sfo3.cdn.digitaloceanspaces.com/test/cat.PES`), this curl command renders the stills plus both high-quality videos (legacy + ultra). Replace `SERVER_IP` with your host/IP.

```bash
curl -X POST http://SERVER_IP:9009/render-from-space \
  -H 'Content-Type: application/json' \
  -d '{
        "input": {
          "bucket_name": "digitizer",
          "file_path": "test",
          "file_name": "cat.PES"
        },
        "operations": {
          "image_generate_low": true,
          "image_generate_high": true,
          "video_generate_low": false,
          "video_generate_high": true,
          "video_generate_ultra_high": true
        },
        "output": {
          "bucket_name": "digitizer",
          "base_path": "test"
        }
      }'
```

The response lists the uploaded asset keys. Watch the server console for `[job …]` progress lines; each finished PNG/MP4 is uploaded immediately.

---

## 4. Optional Local Tools

- `python3 embroidery.py -i input_PES/cat.PES -o output/cat.png -v output/cat.mp4` renders a single file without using the API.
- `python3 render_still.py ...` generates fast + legacy PNGs for quick QA.
- `apply_texture_background.py` composites the PNG on top of fabric textures stored in `assets/textures/`.

---

## 5. Operation Presets

| Operation Flag | Output | Resolution | Samples | Extras |
| --- | --- | --- | --- | --- |
| `image_generate_low` | PNG | 768×768 | 64 Cycles samples | TopView camera, off-white canvas background |
| `image_generate_high` | PNG | 2048×2048 | 512 samples | Same framing + denoised legacy pass |
| `video_generate_low` | MP4 | 1024×1024 | 32 samples | Fast square crop, minimal lighting |
| `video_generate_high` | MP4 | 1× native CameraAnim resolution (default 1920×1080) | 10 samples | Matches the original machine footage, trimmed env to keep focus |
| `video_generate_ultra_high` | MP4 | 1.5× native CameraAnim resolution (clamped to 3840×2160) | 256 samples + adaptive sampling | Canvas table + rim lights, depth of field, thread hair, motion blur, BEST FFmpeg preset |

- Legacy `generate_video=true` still triggers the high-quality preset.
- All videos render 160 frames at 24 fps (≈6.6 s) and use the `output/video_*` naming inside `processed_files/` before upload.
- Each still PNG is uploaded twice: the original white-background render plus a `canvas1.png` textured composite.

---

## 6. Project Layout

```
.
├── BlenderSetup.blend      # Base scene shared by every render
├── blender_script.py       # Runs inside Blender (imports PES, renders stills/videos)
├── render_service.py       # FastAPI endpoint that orchestrates renders via Blender CLI
├── apply_texture_background.py / compare_palette.py  # CLI helpers
├── ImporterScript/         # Blender add-on backed by pyembroidery
├── assets/textures/        # Fabric backdrops
├── input_PES/              # Drop PES files here for CLI tests
├── output/                 # Local renders land here when running CLI utilities
└── processed_files/        # Server backups + logs (populated automatically)
```

---

## 7. Logs & Troubleshooting

- `logs/high_quality_video_revert.log` captures every time the high/ultra scripts tweak camera settings or backgrounds.
- If Blender crashes, inspect the per-job temp folder path printed in the server log before it is cleaned up.
- The custom `ImporterScript` ships with the needed `pyembroidery` wheel, so Blender does not require extra manual installs.

Happy stitching!
