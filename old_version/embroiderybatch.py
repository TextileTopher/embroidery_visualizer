import sys
import os
import argparse
import subprocess
import glob
import time
import csv
from datetime import datetime

def main():
    # Core paths
    blender_path = "/opt/blender/blender"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blend_file = os.path.join(script_dir, "BlenderSetup.blend")
    script_file = os.path.join(script_dir, "blender_script.py")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Launch Blender with given PES files.")
    parser.add_argument("-i", "--input_pes", help="Input PES file name")
    parser.add_argument("-o", "--output_image", help="Output image file name")
    parser.add_argument("-v", "--output_video", help="Output video file name")
    parser.add_argument("-b", "--batch", action="store_true", help="Process all PES files")
    args = parser.parse_args()

    if args.batch:
        # Get all PES files
        input_folder = os.path.join(script_dir, "input_PES")
        pes_files = glob.glob(os.path.join(input_folder, "*.pes"))
        
        if not pes_files:
            print("No PES files found in input_PES folder")
            return

        # CSV setup
        csv_path = os.path.join(script_dir, "processing_log.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File', 'Start Time', 'End Time', 'Duration', 'Status'])

        # Process each file
        for pes_file in pes_files:
            base_name = os.path.splitext(os.path.basename(pes_file))[0]
            print(f"\nProcessing {base_name}.pes...")
            
            start_time = datetime.now()
            
            # Construct the Blender command directly
            cmd = [
                blender_path,
                "-b", blend_file,
                "-P", script_file,
                "--",  # Everything after here is passed to blender_script.py
                "-i", f"{base_name}.pes",
                "-o", f"{base_name}.png",
                "-v", f"{base_name}.mp4"
            ]
            
            print("Running:", " ".join(cmd))
            
            try:
                # Run and wait
                process = subprocess.Popen(
                    cmd,
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Wait for completion and capture output
                stdout, stderr = process.communicate()
                
                # Check if process completed successfully
                if process.returncode == 0:
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log to CSV
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            base_name + '.pes',
                            start_time.strftime('%Y-%m-%d %H:%M:%S'),
                            end_time.strftime('%Y-%m-%d %H:%M:%S'),
                            f"{duration:.2f}",
                            'Complete'
                        ])
                    
                    print(f"Completed in {duration:.2f} seconds")
                else:
                    print(f"Error: Process returned code {process.returncode}")
                    print("Output:", stdout.decode())
                    print("Error:", stderr.decode())
                    
            except Exception as e:
                print(f"Error processing {base_name}.pes: {str(e)}")
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        base_name + '.pes',
                        start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        '0',
                        f'Error: {str(e)}'
                    ])
    else:
        # Single file mode - run directly
        if not all([args.input_pes, args.output_image, args.output_video]):
            parser.error("Single file mode requires -i, -o, and -v arguments")
        
        cmd = [
            blender_path,
            "-b", blend_file,
            "-P", script_file,
            "--",
            "-i", args.input_pes,
            "-o", args.output_image,
            "-v", args.output_video
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()