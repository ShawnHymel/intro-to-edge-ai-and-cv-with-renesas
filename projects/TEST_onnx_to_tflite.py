"""
Convert ONNX model to quantized int8 TFLite model

This script converts an ONNX model to a fully quantized int8 TFLite model
using onnx2tf for the ONNX-to-TensorFlow conversion and the TFLite converter
for int8 quantization with real calibration data.

Usage:
    python onnx_to_tflite.py model.onnx -o model_quant_int8.tflite \
        -c calibration_data_nhwc.npz -n 100

Author: Shawn Hymel
Date: February 2026
License: Apache-2.0
"""

import os
import shutil
import tempfile
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import onnx2tf
import tensorflow as tf

def convert_onnx_to_saved_model(onnx_path, saved_model_dir):
    """Convert ONNX model to TensorFlow SavedModel."""
    print(f"Converting ONNX to SavedModel...")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {saved_model_dir}")

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
        not_use_onnxsim=True,
    )

    print(f"  SavedModel created successfully")

def make_representative_dataset(calib_data):
    """Create a representative dataset generator for TFLite quantization."""
    def representative_dataset():
        for i in range(len(calib_data)):
            yield [calib_data[i:i + 1].astype(np.float32)]
    return representative_dataset


def convert_saved_model_to_tflite(saved_model_dir, tflite_path, calib_data):
    """Convert TF SavedModel to quantized int8 TFLite model."""
    print(f"Converting SavedModel to quantized int8 TFLite...")
    print(f"  Calibration samples: {len(calib_data)}")
    print(f"  Calibration shape:   {calib_data.shape}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # Full int8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calib_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  TFLite model saved to: {tflite_path}")
    print(f"  Model size: {size_kb:.1f} KB")


def verify_tflite_model(tflite_path):
    """Print input/output details of the TFLite model."""
    print(f"\nVerifying TFLite model...")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Inputs:")
    for inp in input_details:
        print(f"    {inp['name']}: {inp['dtype'].__name__}{list(inp['shape'])}")
        if inp['quantization_parameters']['scales'].size > 0:
            print(f"      scale: {inp['quantization_parameters']['scales'][0]:.6f}")
            print(f"      zero_point: {inp['quantization_parameters']['zero_points'][0]}")

    print(f"  Outputs:")
    for out in output_details:
        print(f"    {out['name']}: {out['dtype'].__name__}{list(out['shape'])}")
        if out['quantization_parameters']['scales'].size > 0:
            print(f"      scale: {out['quantization_parameters']['scales'][0]:.6f}")
            print(f"      zero_point: {out['quantization_parameters']['zero_points'][0]}")

def main():
    parser = ArgumentParser(
        description="Convert ONNX model to quantized int8 TFLite model"
    )
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model_quant_int8.tflite",
        help="Path for output TFLite model (default: model_quant_int8.tflite)",
    )
    parser.add_argument(
        "-c", "--calib_data",
        type=str,
        required=True,
        help="Path to NHWC calibration data NPZ file",
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=None,
        help="Max number of calibration samples to use (default: all)",
    )
    parser.add_argument(
        "-k", "--calib_key",
        type=str,
        default="input",
        help="Key name in NPZ file for calibration data (default: 'input')",
    )
    parser.add_argument(
        "--keep_saved_model",
        action="store_true",
        help="Keep intermediate SavedModel directory (default: delete)",
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default=None,
        help="Directory for intermediate SavedModel (default: temp directory)",
    )

    args = parser.parse_args()

    # Validate inputs
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    calib_path = Path(args.calib_data)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration data not found: {calib_path}")

    tflite_path = Path(args.output)

    # Load calibration data
    print(f"Loading calibration data from: {calib_path}")
    npz_data = np.load(calib_path)
    if args.calib_key not in npz_data:
        available = list(npz_data.keys())
        raise KeyError(
            f"Key '{args.calib_key}' not found in NPZ file. "
            f"Available keys: {available}"
        )

    calib_data = npz_data[args.calib_key]
    if args.num_samples is not None:
        calib_data = calib_data[: args.num_samples]
    print(f"  Shape: {calib_data.shape}")
    print(f"  Dtype: {calib_data.dtype}")
    print(f"  Range: [{calib_data.min():.3f}, {calib_data.max():.3f}]")

    # Verify calibration data is NHWC
    if len(calib_data.shape) == 4 and calib_data.shape[1] <= 4:
        print(
            f"  WARNING: Calibration data appears to be NCHW "
            f"(shape {calib_data.shape}). Expected NHWC. "
            f"Use the *_nhwc.npz file."
        )

    # Set up SavedModel directory
    use_temp = args.saved_model_dir is None
    if use_temp:
        saved_model_dir = Path(tempfile.mkdtemp(prefix="onnx2tf_"))
    else:
        saved_model_dir = Path(args.saved_model_dir)

    try:
        # Step 1: ONNX -> SavedModel
        convert_onnx_to_saved_model(onnx_path, saved_model_dir)

        # Step 2: SavedModel -> Quantized int8 TFLite
        convert_saved_model_to_tflite(saved_model_dir, tflite_path, calib_data)

        # Step 3: Verify
        verify_tflite_model(tflite_path)

    finally:
        # Clean up temp SavedModel
        if use_temp and not args.keep_saved_model:
            shutil.rmtree(saved_model_dir, ignore_errors=True)
            print(f"\nCleaned up intermediate SavedModel")
        elif not use_temp:
            print(f"\nSavedModel kept at: {saved_model_dir}")

    print(f"\nDone! Quantized TFLite model: {tflite_path}")

if __name__ == "__main__":
    main()
