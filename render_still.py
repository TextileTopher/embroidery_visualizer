import argparse
import os
import subprocess
import sys
import time

# Path to Blender executable (adjust if your installation lives elsewhere).
BLENDER_BINARY = "/opt/blender/blender"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a single still PNG from a PES file using Blender."
    )
    # The fast render always runs; optional legacy flags produce a slower,
    # higher-quality comparison image by forwarding parameters to
    # blender_render_still.py.
    parser.add_argument(
        "-i",
        "--input_pes",
        required=True,
        help="Input PES file name (looked up in input_PES/) or absolute path.",
    )
    parser.add_argument(
        "-o",
        "--output_image",
        required=True,
        help="Output PNG file name (written under output/) or absolute path.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square resolution for the render (default: 1024).",
    )
    parser.add_argument(
        "--camera",
        default="TopView",
        help="Camera object to render from (default: TopView).",
    )
    parser.add_argument(
        "--thread_thickness",
        type=float,
        default=0.2,
        help="Thread thickness passed to the importer (default: 0.2).",
    )
    parser.add_argument(
        "--fast_samples",
        type=int,
        default=None,
        help="Optional Cycles sample override for the fast render.",
    )
    parser.add_argument(
        "--legacy_output",
        default=None,
        help="Optional second output PNG for a high-quality comparison render.",
    )
    parser.add_argument(
        "--legacy_resolution",
        type=int,
        default=None,
        help="Resolution to use for the legacy render (defaults to fast resolution × 2).",
    )
    parser.add_argument(
        "--legacy_samples",
        type=int,
        default=None,
        help="Optional Cycles sample override for the legacy render.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    blend_file = os.path.join(repo_dir, "BlenderSetup.blend")
    blender_script = os.path.join(repo_dir, "blender_render_still.py")

    cmd = [
        BLENDER_BINARY,
        "-b",
        blend_file,
        "-P",
        blender_script,
        "--",
        "-i",
        args.input_pes,
        "-o",
        args.output_image,
        "--resolution",
        str(args.resolution),
        "--camera",
        args.camera,
        "--thread_thickness",
        str(args.thread_thickness),
    ]

    if args.fast_samples is not None:
        cmd.extend(["--fast_samples", str(args.fast_samples)])
    if args.legacy_output:
        cmd.extend(["--legacy_output", args.legacy_output])
        if args.legacy_resolution is not None:
            cmd.extend(["--legacy_resolution", str(args.legacy_resolution)])
        if args.legacy_samples is not None:
            cmd.extend(["--legacy_samples", str(args.legacy_samples)])

    print("Running Blender command:", " ".join(cmd))
    start_time = time.perf_counter()
    subprocess.run(cmd, check=True)
    total_elapsed = time.perf_counter() - start_time
    print(f"Blender exited successfully in {total_elapsed:.2f} seconds.")


if __name__ == "__main__":
    sys.exit(main())
