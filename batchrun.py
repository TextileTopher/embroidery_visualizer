import os
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime
import sys

# Define the correct Python interpreter path
BLENDER_PYTHON = "/opt/blender/4.3/python/bin/python3.11"

def initialize_csv(csv_file, pes_files):
    """Initialize CSV with PES files if it doesn't exist or add new PES files."""
    existing_files = set()
    
    if csv_file.exists():
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_files = {row['PES name'] for row in reader}
    
    file_exists = csv_file.exists()
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['PES name', 'PNG name', 'MP4 name', 'Total seconds', 'Date Completed', 'Status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for pes_file in pes_files:
            if pes_file.name not in existing_files:
                writer.writerow({
                    'PES name': pes_file.name,
                    'PNG name': '',
                    'MP4 name': '',
                    'Total seconds': '',
                    'Date Completed': '',
                    'Status': 'Pending'
                })

def update_csv_entry(csv_file, pes_name, png_name, mp4_name, processing_time, status):
    """Update CSV entry for completed file."""
    rows = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        if row['PES name'] == pes_name:
            row['PNG name'] = png_name if status == 'Success' else ''
            row['MP4 name'] = mp4_name if status == 'Success' else ''
            row['Total seconds'] = f"{processing_time:.2f}"
            row['Date Completed'] = datetime.now().strftime('%m-%d-%Y')
            row['Status'] = status
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def get_unprocessed_files(csv_file, pes_files):
    """Get list of files that haven't been successfully processed yet."""
    processed_files = set()
    if csv_file.exists():
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            processed_files = {row['PES name'] for row in reader if row['Status'] == 'Success'}
    
    return [f for f in pes_files if f.name not in processed_files]

def process_pes_files(input_folder="input_PES", output_folder="output"):
    script_dir = Path(__file__).parent.absolute()
    input_path = Path(input_folder).absolute()
    output_path = Path(output_folder).absolute()
    csv_file = script_dir / "processing_log.csv"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' not found")
    if not output_path.exists():
        os.makedirs(output_path)
    
    pes_files = list(input_path.glob("*.pes"))
    
    if not pes_files:
        print(f"No .pes files found in {input_folder}")
        return
    
    initialize_csv(csv_file, pes_files)
    unprocessed_files = get_unprocessed_files(csv_file, pes_files)
    
    if not unprocessed_files:
        print("All files have been processed!")
        return
    
    program_start_time = time.time()
    
    for pes_file in unprocessed_files:
        base_name = pes_file.stem
        png_output = output_path / f"{base_name}.png"
        mp4_output = output_path / f"{base_name}.mp4"
        pes_relative = pes_file.name
        
        print(f"\nProcessing {pes_file.name}...")
        file_start_time = time.time()
        
        try:
            # Use the correct Python interpreter and embroidery.py script
            command = [
                BLENDER_PYTHON,
                str(script_dir / "embroidery.py"),
                "-i", pes_relative,
                "-o", str(png_output.name),
                "-v", str(mp4_output.name)
            ]
            
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir  # Run from the directory containing embroidery.py
            )
            
            processing_time = time.time() - file_start_time
            
            # Check if output files were actually created
            if png_output.exists() and mp4_output.exists():
                status = 'Success'
                print(f"Successfully processed {pes_file.name} in {processing_time:.2f} seconds")
            else:
                status = 'Failed - No output files'
                print(f"Error: Output files not created for {pes_file.name}")
                print("Command output:")
                print(result.stdout)
                print(result.stderr)
            
            update_csv_entry(csv_file, pes_file.name, png_output.name, 
                           mp4_output.name, processing_time, status)
            
        except subprocess.CalledProcessError as e:
            processing_time = time.time() - file_start_time
            print(f"Error processing {pes_file.name}:")
            print(f"Command output:\n{e.stdout}\n{e.stderr}")
            update_csv_entry(csv_file, pes_file.name, '', '', 
                           processing_time, f'Failed - Process Error')
            
        except Exception as e:
            processing_time = time.time() - file_start_time
            print(f"Unexpected error processing {pes_file.name}: {str(e)}")
            update_csv_entry(csv_file, pes_file.name, '', '', 
                           processing_time, f'Failed - {str(e)}')
        
        total_time = time.time() - program_start_time
        print(f"Total program runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    process_pes_files()