import sys
import os
import argparse
import subprocess

# Path to Blender executable (if always the same)
blender_path = "/opt/blender/blender"

# topher@todd:~/embviz2$ /opt/blender/4.3/python/bin/python3.11 embroidery.py -i big.pes -o big.png -v big.mp4

# Determine the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to blend file and blender script (assuming they're in the same folder)
blend_file = os.path.join(script_dir, "BlenderSetup.blend")
script_file = os.path.join(script_dir, "blender_script.py")

# Parse arguments
parser = argparse.ArgumentParser(description="Launch Blender with given PES, image, and optional video files.")
parser.add_argument("-i", "--input_pes", required=True, help="Input PES file name (located in input_PES folder)")
parser.add_argument("-o", "--output_image", required=True, help="Output image file name (will go in output folder)")
parser.add_argument(
    "-v",
    "--output_video",
    required=False,
    default=None,
    help="Optional output video file name (will go in output folder). Omit to skip animation.",
)
parser.add_argument(
    "--backgrounds",
    default="white",
    help="Comma-separated list of backgrounds to render (choices: white,black,side_white). Default: white.",
)
parser.add_argument(
    "--resolutions",
    default="1024",
    help="Comma-separated list of square resolutions (e.g., '1024,2048'). Default: 1024.",
)
parser.add_argument(
    "--show_jump_wires",
    action="store_true",
    help="Render jump stitches between sections (slower). Off by default for speed.",
)
parser.add_argument(
    "--enable_hair",
    action="store_true",
    help="Enable thread fuzz particle hair (slower).",
)
parser.add_argument(
    "--skip_video",
    action="store_true",
    help="Force skip of animation render even if --output_video is supplied.",
)
parser.add_argument(
    "--video_quality",
    choices=["fast", "high"],
    default="high",
    help="Quality preset for animation renders (default: high).",
)
parser.add_argument(
    "--video_framing",
    choices=["zoomed", "full"],
    default="full",
    help="Framing preset for animation renders (default: full).",
)

args = parser.parse_args()

# Construct the blender command
cmd = [
    blender_path,
    "-b", blend_file,
    "-P", script_file,
    "--",  # Everything after here is passed to blender_script.py
    "-i", args.input_pes,
    "-o", args.output_image,
]

if args.output_video:
    cmd.extend(["-v", args.output_video])
    cmd.extend(["--video_quality", args.video_quality])
    cmd.extend(["--video_framing", args.video_framing])
if args.skip_video or not args.output_video:
    cmd.append("--skip_video")

cmd.extend(["--backgrounds", args.backgrounds])
cmd.extend(["--resolutions", args.resolutions])
if args.show_jump_wires:
    cmd.append("--show_jump_wires")
if args.enable_hair:
    cmd.append("--enable_hair")

print("Running Blender command:", " ".join(cmd))

# Run the command
subprocess.run(cmd, check=True)
