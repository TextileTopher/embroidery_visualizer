import argparse
import os
import subprocess
import sys
import time

from apply_texture_background import apply_texture_overlay, resolve_path as resolve_texture_path

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
        "--high_quality_output",
        dest="legacy_output",
        default=None,
        help="Alias for --legacy_output (preferred).",
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
    parser.add_argument(
        "--fabric_texture",
        default="assets/textures/canvas1.png",
        help="Texture to replace the white background (set to 'none' to skip).",
    )
    parser.add_argument(
        "--fabric_tolerance",
        type=int,
        default=8,
        help="How close to pure white a pixel must be before it is replaced.",
    )
    return parser.parse_args()


def resolve_existing_output(repo_dir, user_path, fallback_folder="output"):
    if os.path.isabs(user_path):
        return os.path.abspath(user_path)

    candidate = os.path.join(repo_dir, user_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    fallback = os.path.join(repo_dir, fallback_folder, user_path)
    if os.path.exists(fallback):
        return os.path.abspath(fallback)

    # File might not exist yet; prefer folder under fallback.
    return os.path.abspath(fallback)


def maybe_apply_texture(repo_dir, image_path, texture_arg, tolerance):
    if not texture_arg or texture_arg.lower() == "none":
        return

    texture_path = resolve_texture_path(repo_dir, texture_arg, "assets/textures")
    print(f"Applying fabric texture {texture_path} to {image_path}")
    apply_texture_overlay(
        render_path=image_path,
        texture_path=texture_path,
        output_path=image_path,
        tolerance=tolerance,
    )
    print(f"Updated textured output: {image_path}")


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

    output_abs = resolve_existing_output(repo_dir, args.output_image)
    maybe_apply_texture(repo_dir, output_abs, args.fabric_texture, args.fabric_tolerance)

    if args.legacy_output:
        legacy_output_abs = resolve_existing_output(repo_dir, args.legacy_output)
        maybe_apply_texture(repo_dir, legacy_output_abs, args.fabric_texture, args.fabric_tolerance)


if __name__ == "__main__":
    sys.exit(main())
