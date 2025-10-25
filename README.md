# Embroidery Visualizer

Embroidery Visualizer is a Blender-based toolkit for converting Brother PES embroidery files into 3D thread visualizations and rendered media. It automates PES import inside Blender, normalizes the geometry for consistent sizing, and exports both still images and animation flythroughs that can be shared with clients or stitched into production workflows.

## Features

- **Custom Blender importer** – Loads PES stitch data through the bundled `ImporterScript` add-on, builds geometry nodes for thread depth, and assigns materials that reflect the original thread palette.
- **Automated scene preparation** – Recenters imported collections, converts curves to meshes, and scales each design to a configurable 5-inch (0.127 m) bounding box for uniform presentation.
- **Media export pipeline** – Runs Blender headlessly to render a still image (`.png`) and an animation (`.mp4`) for every PES file.
- **Fast/legacy still pipeline** – `render_still.py` runs Blender in headless mode to generate a quick Cycles render plus an optional high-quality legacy pass for comparisons.
- **HTTP render service** – `render_service.py` exposes a FastAPI endpoint that accepts PES uploads and returns the requested still(s) while backing up all artefacts to `processed_files/`.
- **Batch processing** – Iterates over the `input_PES/` directory, launches the Blender Python runtime for each design, and logs completion status, outputs, and runtimes to `processing_log.csv`.
- **PES diagnostics (optional)** – Generates binary, hex, and statistical breakdowns of PES files for debugging or reverse engineering.

## Project Structure

```
.
├── ImporterScript/        # Bundled Blender add-on that provides the PES importer operator
├── BlenderSetup.blend     # Base Blender scene used for rendering
├── blender_script.py      # Invoked inside Blender; handles import, centering, scaling, exports
├── embroidery.py          # CLI entry point that launches Blender headlessly for one PES file
├── batchrun.py            # Batch runner that processes every PES file and records results
├── analyze_pes.py         # Diagnostic utility that inspects PES headers and binary content
├── input_PES/             # Drop PES files here for processing
├── output/                # Rendered PNG and MP4 files are written here (created on demand)
└── pes_analysis/          # Output folder for PES diagnostics (created on demand)
```

## Requirements

- Blender 4.3 with Python located at `/opt/blender/4.3/python/bin/python3.11` (adjust paths in the scripts if Blender is installed elsewhere).
- The `pyembroidery` Python package (bundled with the add-on wheels in `ImporterScript/wheels/`).
- System packages required by Blender for headless rendering (OpenGL-capable drivers, ffmpeg, etc.).

## Usage

### Render a Single PES File

1. Place your `.pes` file in the `input_PES/` directory.
2. From a shell inside this repository, run:
   ```bash
   /opt/blender/4.3/python/bin/python3.11 embroidery.py \
       -i your_design.pes \
       -o your_design.png \
       -v your_design.mp4
   ```
3. The rendered outputs appear in the `output/` directory.

The script calls Blender in background mode with `BlenderSetup.blend`, runs `blender_script.py`, and forwards the CLI arguments for file names.

### Fast vs Legacy Still Rendering (CLI)

Use `render_still.py` when you need a quick preview alongside an optional
high-quality comparison render. The fast pass is always generated; add
`--legacy_output` to request the legacy image.

```bash
python3 render_still.py \
    -i input_PES/cat.PES \
    -o output/cat_fast.png \
    --legacy_output output/cat_legacy.png \
    --legacy_resolution 2048
```

- `--legacy_resolution` defaults to `resolution * 2` when omitted.
- Use `--fast_samples` or `--legacy_samples` to override Cycles sampling.
- Both PNGs are written to the path you provide, and Blender logs render
  metrics to the console.

### Batch Process All PES Files

1. Populate `input_PES/` with one or more `.pes` files.
2. Run the batch script (it invokes the same single-file pipeline for each design):
   ```bash
   python3 batchrun.py
   ```
3. Monitor progress in the terminal. Each run updates `processing_log.csv` with the PNG/MP4 names, runtime, completion date, and status for auditing.

### Serve the Render API

`render_service.py` exposes the still rendering pipeline over HTTP so teammates
can request images remotely. Launch it with Uvicorn from the repository root:

```bash
uvicorn render_service:app --host 0.0.0.0 --port 8000
```

- Upload a PES file via `POST /render` (see `API_USAGE.md` for field details).
- Choose `mode=fast`, `mode=legacy`, or `mode=both` to control which PNGs are
  returned. When both are requested the response is a `.zip` containing both
  renders.
- Add `include_video=true` to request the animation MP4; the server bundles the
  still(s) and the video into a ZIP download.
- The service saves every uploaded PES plus the generated PNGs into
  `processed_files/` with timestamped names so you never lose an artefact.
- Response headers (`X-Input-Backup`, `X-Fast-Output-Backup`,
  `X-Legacy-Output-Backup`, `X-Video-Output-Backup`, `X-Archive-Backup`)
  record where each file was stored.

### Optional PES Analysis

For low-level inspection of PES file structure, execute:
```bash
python3 analyze_pes.py
```
The script scans the sample files listed in its `files_to_analyze` array and writes binary, hex, header, and statistical reports to `pes_analysis/`. Customize the file list or pass your own paths when using the helper functions.

## Blender Script Workflow

`blender_script.py` performs several steps once Blender launches:

1. Loads the bundled importer operator and parses CLI arguments passed after `--`.
2. Imports the requested PES file using `bpy.ops.import_scene.embroidery` with geometry-node-based thread thickness.
3. Converts the generated curves to meshes, recenters the collection at world origin, and aligns the design to the table surface.
4. Computes the bounding box, scales the model so the largest dimension equals 0.127 meters (5 inches), and reapplies centering.
5. Leaves the prepared scene ready for render settings defined in `BlenderSetup.blend` (camera, lighting, animation).

Adjust parameters such as `thread_thickness`, `desired_dimension`, or material generation inside `ImporterScript/__init__.py` to tailor the look of rendered stitches. The importer script constructs thread materials using geometry nodes and object attributes to color-code individual threads.

## Logging and Outputs

- `output/`: Contains the final `.png` and `.mp4` files for each design. The MP4 is typically a flythrough or animated preview configured in the Blender scene.
- `processing_log.csv`: Created by `batchrun.py` to record job metadata and success/failure states for traceability.
- `processed_files/`: Stores backed-up PES uploads, rendered PNGs, optional MP4 videos, and the ZIP bundles the API returns along with `render_service.log` (Blender stdout/stderr).
- `pes_analysis/`: Stores optional diagnostic reports created by `analyze_pes.py`.

## Troubleshooting

- **Blender path errors** – Update `blender_path` in `embroidery.py` and `BLENDER_PYTHON` in `batchrun.py` if Blender is installed elsewhere.
- **Missing output files** – The batch script marks entries as failed if the PNG or MP4 is not produced; check the Blender console output captured in the log for details.
- **Geometry alignment issues** – Modify the centering and scaling logic near the end of `blender_script.py` to tweak how designs are positioned in the scene.
- **Unexpected ZIP downloads from the API** – `mode=both` intentionally returns a `.zip` containing fast and legacy renders. Switch to `mode=fast` or `mode=legacy` if you only need a single PNG.

## License

This project bundles the `ImporterScript` add-on (see `ImporterScript/LICENSE`). Confirm compatibility with your usage before redistribution.
