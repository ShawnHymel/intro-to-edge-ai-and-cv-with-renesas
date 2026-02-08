"""
Compare TFLite model outputs with firmware outputs.

This script:
1. Reads the test image from test_sample.h
2. Runs TFLite interpreter
3. Prints int8 outputs in the same format as firmware for easy comparison

Usage:
    python compare_tflite_vs_firmware.py <model.tflite>
"""

import sys
import re
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not available, trying tflite_runtime")
    import tflite_runtime.interpreter as tflite
    tf = None


def extract_test_data_from_header(header_path):
    """Extract uint8 pixel data from test_sample.h"""
    with open(header_path, 'r') as f:
        content = f.read()

    match = re.search(r'const uint8_t test_input\[.*?\]\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find test_input array in header file")

    data_str = match.group(1)
    numbers = re.findall(r'\d+', data_str)
    data = np.array([int(n) for n in numbers], dtype=np.uint8)
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_tflite_vs_firmware.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_sample_path = Path("test_sample.h")

    print("=" * 60)
    print("TFLite vs Firmware Output Comparison")
    print("=" * 60)

    # Load test image
    print(f"\nLoading test image from: {test_sample_path}")
    pixel_data = extract_test_data_from_header(test_sample_path)
    print(f"  Extracted {len(pixel_data)} values")

    # Reshape to HWC: (192, 192, 3)
    img_hwc = pixel_data.reshape(192, 192, 3)

    # Load TFLite model
    print(f"\nLoading TFLite model: {model_path}")
    if tf:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nInput tensor:")
    for inp in input_details:
        print(f"  Name: {inp['name']}")
        print(f"  Shape: {inp['shape']}")
        print(f"  Dtype: {inp['dtype']}")
        if 'quantization_parameters' in inp:
            qp = inp['quantization_parameters']
            print(f"  Scale: {qp['scales']}, Zero point: {qp['zero_points']}")

    print(f"\nOutput tensors:")
    for out in output_details:
        print(f"  Name: {out['name']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Dtype: {out['dtype']}")
        if 'quantization_parameters' in out:
            qp = out['quantization_parameters']
            print(f"  Scale: {qp['scales']}, Zero point: {qp['zero_points']}")

    # Prepare input
    # Check expected input format
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    print(f"\nPreparing input...")
    print(f"  Model expects shape: {input_shape}")

    # Check if model expects grayscale (1 channel) or RGB (3 channels)
    num_channels = input_shape[-1]

    if num_channels == 1:
        # Model expects grayscale - convert RGB to grayscale
        print(f"  Model expects GRAYSCALE (1 channel)")
        print(f"  Converting RGB to grayscale using: Y = 0.299*R + 0.587*G + 0.114*B")
        img_gray = (0.299 * img_hwc[:,:,0] + 0.587 * img_hwc[:,:,1] + 0.114 * img_hwc[:,:,2])
        img_gray = img_gray.astype(np.uint8)
        print(f"  Grayscale shape: {img_gray.shape}")
        print(f"  Grayscale range: [{img_gray.min()}, {img_gray.max()}]")

        if input_dtype == np.int8:
            # Convert uint8 [0,255] to int8 [-128,127]
            input_data = img_gray.astype(np.int16) - 128
            input_data = input_data.astype(np.int8)
            input_data = np.expand_dims(input_data, axis=(0, -1))  # Add batch and channel dims
            print(f"  Converted to int8, shape: {input_data.shape}")
            print(f"  First 12 int8 values: {input_data.flatten()[:12]}")
        else:
            input_data = img_gray.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=(0, -1))
            print(f"  Converted to float32, shape: {input_data.shape}")

    elif input_dtype == np.int8:
        # Model expects int8 RGB input
        # Convert uint8 [0,255] to int8 [-128,127]
        if input_shape[-1] == 3:  # NHWC format
            input_data = img_hwc.astype(np.int16) - 128
            input_data = input_data.astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dim
        else:  # Might be NCHW
            img_chw = np.transpose(img_hwc, (2, 0, 1))
            input_data = img_chw.astype(np.int16) - 128
            input_data = input_data.astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)

        print(f"  Converted to int8, shape: {input_data.shape}")
        print(f"  First 12 int8 values: {input_data.flatten()[:12]}")
    else:
        # Model expects float input
        input_data = img_hwc.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        print(f"  Converted to float32, shape: {input_data.shape}")

    # Run inference
    print("\nRunning TFLite inference...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get outputs
    outputs = []
    for out in output_details:
        outputs.append(interpreter.get_tensor(out['index']))

    # Print outputs in firmware format
    print("\n" + "=" * 60)
    print("RAW INT8 TFLITE OUTPUTS (compare with firmware)")
    print("=" * 60)

    for i, out in enumerate(outputs):
        out_flat = out.flatten()
        size = len(out_flat)

        print(f"\nOutput{i} shape: {out.shape}, size: {size}")

        if out.dtype == np.int8:
            # Grid determination based on size
            if size == 648:
                grid_name = "6x6 grid"
            elif size == 2592:
                grid_name = "12x12 grid"
            else:
                grid_name = f"{size} values"

            print(f"Output{i} ({grid_name}, {size} int8 values) - first 36:")
            for j in range(min(36, size)):
                if j % 12 == 0:
                    print(f"[{j:3d}]: ", end="")
                print(f"{out_flat[j]:4d} ", end="")
                if j % 12 == 11:
                    print()
        else:
            print(f"  Dtype: {out.dtype}")
            print(f"  First 18 values: {out_flat[:18]}")

    # Print key indices for comparison
    print("\n" + "=" * 60)
    print("KEY INDICES FOR COMPARISON")
    print("=" * 60)

    for i, out in enumerate(outputs):
        out_flat = out.flatten()
        size = len(out_flat)

        print(f"\nOutput{i} ({out.shape}):")
        # Print specific indices that are useful for debugging YOLO outputs
        indices = [0, 4, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23]
        for idx in indices:
            if idx < size:
                print(f"  [{idx:3d}] = {out_flat[idx]}")

    print("\n" + "=" * 60)
    print("COMPARISON NOTES")
    print("=" * 60)
    print("""
Compare the values above with your firmware output.

For YOLO outputs, each anchor has 6 values: [tx, ty, tw, th, obj, cls]
- Index 0,1,2,3 = tx, ty, tw, th (bounding box parameters)
- Index 4 = objectness score (should be negative for no detection)
- Index 5 = class score

If TFLite and firmware outputs match: Quantization is correct
If TFLite and firmware outputs differ: There's a RUHMI code generation bug
""")


if __name__ == '__main__':
    main()
