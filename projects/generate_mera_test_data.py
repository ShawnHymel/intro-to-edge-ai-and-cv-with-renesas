"""
Generate test data for MERA interpreter comparison.

This script:
1. Reads test_sample.h and extracts the pixel data
2. Converts to NCHW float32 format (matching model input)
3. Saves as .npy file for use with test_mera_model.py

Usage:
    python generate_mera_test_data.py

Output:
    test_image_nchw.npy - Test image in NCHW format for MERA
"""

import re
import numpy as np
from pathlib import Path

# Path to test_sample.h
TEST_SAMPLE_PATH = Path("./test_sample.h")
OUTPUT_DIR = Path(".")

def extract_test_data_from_header(header_path):
    """Extract uint8 pixel data from test_sample.h"""
    with open(header_path, 'r') as f:
        content = f.read()

    # Find the array data between { and };
    match = re.search(r'const uint8_t test_input\[.*?\]\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find test_input array in header file")

    # Extract numbers
    data_str = match.group(1)
    numbers = re.findall(r'\d+', data_str)
    data = np.array([int(n) for n in numbers], dtype=np.uint8)

    print(f"Extracted {len(data)} values from {header_path}")
    return data

def main():
    # Extract pixel data from header
    print("Reading test_sample.h...")
    pixel_data = extract_test_data_from_header(TEST_SAMPLE_PATH)

    # Verify size (192 * 192 * 3 = 110592)
    expected_size = 192 * 192 * 3
    if len(pixel_data) != expected_size:
        raise ValueError(f"Expected {expected_size} values, got {len(pixel_data)}")

    # Reshape to HWC format: (192, 192, 3)
    img_hwc = pixel_data.reshape(192, 192, 3)
    print(f"Reshaped to HWC: {img_hwc.shape}")

    # Convert to CHW format: (3, 192, 192)
    img_chw = np.transpose(img_hwc, (2, 0, 1))
    print(f"Converted to CHW: {img_chw.shape}")

    # Normalize to [0, 1] float32
    img_float = img_chw.astype(np.float32) / 255.0
    print(f"Normalized to float32: [{img_float.min():.4f}, {img_float.max():.4f}]")

    # Add batch dimension: (1, 3, 192, 192)
    img_nchw = np.expand_dims(img_float, axis=0)
    print(f"Final shape (NCHW): {img_nchw.shape}")

    # Verify values match firmware
    print("\nVerification (should match firmware values):")
    print(f"  [0,0,0,0] = {img_nchw[0,0,0,0]:.4f} (expect 0.6275 for R=160)")
    print(f"  [0,1,0,0] = {img_nchw[0,1,0,0]:.4f} (expect 0.6824 for G=174)")
    print(f"  [0,2,0,0] = {img_nchw[0,2,0,0]:.4f} (expect 0.7882 for B=201)")
    print(f"  Sum of first 1000 R values: {img_nchw[0,0,:,:].flatten()[:1000].sum():.4f}")

    # Calculate full input sum for comparison
    full_sum = 0
    for c in range(3):
        for y in range(192):
            for x in range(192):
                idx = c * 192 * 192 + y * 192 + x
                if idx < 1000:
                    full_sum += img_nchw[0, c, y, x]
    print(f"  Sum matching firmware method: {full_sum:.4f} (firmware shows 791.0457)")

    # Save as npy for MERA testing
    output_path = OUTPUT_DIR / "test_image_nchw.npy"
    np.save(output_path, img_nchw)
    print(f"\nSaved to: {output_path}")
    print(f"  Shape: {img_nchw.shape}")
    print(f"  Dtype: {img_nchw.dtype}")

    # Also save as NPZ with input name for MERA
    # The input name from the model is "input"
    npz_path = OUTPUT_DIR / "test_input_for_mera.npz"
    np.savez(npz_path, input=img_nchw)
    print(f"Saved NPZ for MERA: {npz_path}")

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("""
1. Run the MERA interpreter test:

   python test_mera_model.py \\
       --mera_model <path_to_quantized_model.mera> \\
       --test_image test_image_nchw.npy \\
       --output_dir /tmp/mera_test

2. Compare MERA interpreter int8 outputs with firmware int8 outputs

3. If MERA outputs match firmware: issue is in quantization
   If MERA outputs differ from firmware: issue is in Ethos-U code generation
""")

if __name__ == '__main__':
    main()
