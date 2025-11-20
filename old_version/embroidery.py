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
parser = argparse.ArgumentParser(description="Launch Blender with given PES, image, and video files.")
parser.add_argument("-i", "--input_pes", required=True, help="Input PES file name (located in input_PES folder)")
parser.add_argument("-o", "--output_image", required=True, help="Output image file name (will go in output folder)")
parser.add_argument("-v", "--output_video", required=True, help="Output video file name (will go in output folder)")

args = parser.parse_args()

# Construct the blender command
cmd = [
    blender_path,
    "-b", blend_file,
    "-P", script_file,
    "--",  # Everything after here is passed to blender_script.py
    "-i", args.input_pes,
    "-o", args.output_image,
    "-v", args.output_video
]

print("Running Blender command:", " ".join(cmd))

# Run the command
subprocess.run(cmd, check=True)