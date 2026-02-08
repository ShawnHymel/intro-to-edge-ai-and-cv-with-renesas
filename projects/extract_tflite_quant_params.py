"""
Extract quantization parameters from a TFLite model.

Usage:
    python extract_tflite_quant_params.py model.tflite
"""

import sys
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not available, trying tflite_runtime")
    import tflite_runtime.interpreter as tflite
    tf = None


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_tflite_quant_params.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"Loading model: {model_path}")

    # Load the TFLite model
    if tf:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Get input details
    print("\n" + "=" * 60)
    print("INPUT TENSORS")
    print("=" * 60)
    input_details = interpreter.get_input_details()
    for inp in input_details:
        print(f"\nName: {inp['name']}")
        print(f"  Index: {inp['index']}")
        print(f"  Shape: {inp['shape']}")
        print(f"  Dtype: {inp['dtype']}")
        if 'quantization' in inp:
            print(f"  Quantization: {inp['quantization']}")
        if 'quantization_parameters' in inp:
            qp = inp['quantization_parameters']
            print(f"  Quantization parameters:")
            print(f"    scales: {qp['scales']}")
            print(f"    zero_points: {qp['zero_points']}")

    # Get output details
    print("\n" + "=" * 60)
    print("OUTPUT TENSORS")
    print("=" * 60)
    output_details = interpreter.get_output_details()
    for out in output_details:
        print(f"\nName: {out['name']}")
        print(f"  Index: {out['index']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Dtype: {out['dtype']}")
        if 'quantization' in out:
            print(f"  Quantization: {out['quantization']}")
        if 'quantization_parameters' in out:
            qp = out['quantization_parameters']
            print(f"  Quantization parameters:")
            print(f"    scales: {qp['scales']}")
            print(f"    zero_points: {qp['zero_points']}")

    # Generate C code snippet
    print("\n" + "=" * 60)
    print("C CODE FOR hal_entry.c")
    print("=" * 60)

    for i, out in enumerate(output_details):
        qp = out.get('quantization_parameters', {})
        scales = qp.get('scales', [0.0])
        zero_points = qp.get('zero_points', [0])

        scale = scales[0] if len(scales) > 0 else 0.0
        zp = zero_points[0] if len(zero_points) > 0 else 0

        print(f"\n/* Output {i}: {out['name']} */")
        print(f"/* Shape: {out['shape']} */")
        print(f"float output{i}_scale = {scale}f;")
        print(f"int output{i}_zero_point = {zp};")


if __name__ == '__main__':
    main()
