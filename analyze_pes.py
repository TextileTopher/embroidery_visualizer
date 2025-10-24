import os
import struct
import binascii

def analyze_pes_file(input_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the binary data
    with open(input_file, 'rb') as f:
        binary_data = f.read()
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print(f"\nAnalyzing {base_name}...")
    print(f"File size: {len(binary_data)} bytes")
    
    # 1. Complete Binary representation
    binary_output = os.path.join(output_dir, f"{base_name}_binary.txt")
    with open(binary_output, 'w') as f:
        f.write(f"Complete binary representation of {base_name}\n")
        f.write(f"File size: {len(binary_data)} bytes\n")
        f.write("=" * 50 + "\n\n")
        for i, byte in enumerate(binary_data):
            binary_value = bin(byte)[2:].zfill(8)
            f.write(f"Byte {i:08d}: {binary_value}\n")
    
    # 2. Complete Hex dump with ASCII representation
    hex_output = os.path.join(output_dir, f"{base_name}_hex.txt")
    with open(hex_output, 'w') as f:
        f.write(f"Complete hex dump of {base_name}\n")
        f.write(f"File size: {len(binary_data)} bytes\n")
        f.write("=" * 50 + "\n\n")
        f.write("Offset    Hexadecimal                                      ASCII\n")
        f.write("-" * 78 + "\n")
        
        offset = 0
        while offset < len(binary_data):
            # Get 16 bytes chunk
            chunk = binary_data[offset:offset+16]
            
            # Hex representation
            hex_values = ' '.join([f"{b:02x}" for b in chunk])
            hex_values = hex_values.ljust(48)  # Pad for alignment
            
            # ASCII representation
            ascii_values = ''.join([chr(b) if 32 <= b <= 126 else '.' for b in chunk])
            
            # Write line with offset, hex, and ASCII
            f.write(f"{offset:08x}  {hex_values}  |{ascii_values}|\n")
            offset += 16
    
    # 3. Detailed PES header analysis
    header_output = os.path.join(output_dir, f"{base_name}_header.txt")
    with open(header_output, 'w') as f:
        f.write(f"PES File Header Analysis for {base_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # PES magic number and version check
        f.write(f"First 8 bytes (hex): {binascii.hexlify(binary_data[:8]).decode()}\n")
        f.write(f"First 8 bytes (ASCII): {repr(binary_data[:8])}\n\n")
        
        # Detailed header parse
        f.write("Detailed Header Breakdown:\n")
        f.write("-" * 30 + "\n")
        for i in range(0, min(256, len(binary_data)), 4):
            hex_value = binascii.hexlify(binary_data[i:i+4]).decode()
            int_value = int.from_bytes(binary_data[i:i+4], 'little')
            try:
                float_value = struct.unpack('f', binary_data[i:i+4])[0]
                f.write(f"Offset {i:03d}-{i+3:03d}: {hex_value} (int: {int_value}, float: {float_value:.4f})\n")
            except:
                f.write(f"Offset {i:03d}-{i+3:03d}: {hex_value} (int: {int_value})\n")
    
    # 4. Statistical analysis
    stats_output = os.path.join(output_dir, f"{base_name}_stats.txt")
    with open(stats_output, 'w') as f:
        f.write(f"File Statistics for {base_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # File size
        f.write(f"Total file size: {len(binary_data)} bytes\n")
        f.write(f"Number of blocks (16 bytes): {len(binary_data)//16}\n")
        f.write(f"Remaining bytes: {len(binary_data)%16}\n\n")
        
        # Byte frequency analysis
        byte_freq = {}
        for byte in binary_data:
            byte_freq[byte] = byte_freq.get(byte, 0) + 1
        
        f.write("Byte frequency analysis:\n")
        f.write("-" * 30 + "\n")
        for byte, count in sorted(byte_freq.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(binary_data)) * 100
            char = chr(byte) if 32 <= byte <= 126 else '.'
            f.write(f"0x{byte:02x} ({char}): {count} times ({percentage:.2f}%)\n")

def analyze_multiple_files(files, output_dir):
    for file in files:
        if os.path.exists(file):
            analyze_pes_file(file, output_dir)
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    # List of files to analyze
    files_to_analyze = [
        "input_PES/big.pes",
        "input_PES/2eggs.PES",
        "input_PES/3roses.pes",
        "input_PES/3wisemen.pes"
    ]
    
    output_dir = "pes_analysis"
    
    print("Starting PES file analysis...")
    analyze_multiple_files(files_to_analyze, output_dir)
    print(f"\nAnalysis complete. Check the {output_dir} directory for results.")