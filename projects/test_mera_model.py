"""
Test MERA quantized model with a specific input image.

This script runs the quantized .mera model through the MERA interpreter
and compares outputs with PyTorch reference outputs.

Usage:
    python test_mera_model.py \
        --mera_model /path/to/model.mera \
        --test_image /path/to/test_image_nchw.npy \
        --pytorch_outputs /path/to/pytorch_outputs.npz

Author: Claude (Anthropic)
"""

import argparse
import numpy as np
import mera
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Test MERA quantized model')
    parser.add_argument('--mera_model', type=str, required=True,
                        help='Path to quantized .mera model')
    parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image .npy file (NCHW format, float32, [0,1] range)')
    parser.add_argument('--pytorch_outputs', type=str, default=None,
                        help='Path to PyTorch outputs .npz file for comparison')
    parser.add_argument('--output_dir', type=str, default='/tmp/mera_test',
                        help='Directory for MERA deployer output')
    args = parser.parse_args()

    # Load test image
    print("=" * 60)
    print("Loading test image...")
    test_img = np.load(args.test_image)
    print(f"  Shape: {test_img.shape}")
    print(f"  Dtype: {test_img.dtype}")
    print(f"  Range: [{test_img.min():.4f}, {test_img.max():.4f}]")
    print(f"  Sum: {test_img.sum():.4f}")
    print(f"  First 5 values: {test_img.flatten()[:5]}")

    # Load PyTorch outputs if provided
    pytorch_outputs = None
    if args.pytorch_outputs:
        print("\nLoading PyTorch reference outputs...")
        pytorch_data = np.load(args.pytorch_outputs)
        pytorch_outputs = {
            'output0': pytorch_data['output0'],
            'output1': pytorch_data['output1']
        }
        print(f"  output0 shape: {pytorch_outputs['output0'].shape}")
        print(f"  output1 shape: {pytorch_outputs['output1'].shape}")

    # Load and run MERA model
    print("\n" + "=" * 60)
    print("Loading MERA quantized model...")
    print(f"  Model path: {args.mera_model}")

    with mera.Deployer(args.output_dir, overwrite=True) as deployer:
        mera_model = mera.ModelLoader(deployer).from_quantized_mera(args.mera_model)

        # Print model info
        print("\nModel inputs:")
        for name, inp in mera_model.input_desc.all_inputs.items():
            print(f"  {name}: shape={inp.input_shape}, dtype={inp.input_type}")

        # Deploy to MERA interpreter
        print("\nDeploying to MERA Interpreter...")
        deploy = deployer.deploy(
            mera_model,
            mera_platform=mera.Platform.MCU_CPU,
            target=mera.Target.MERAInterpreter
        )
        runner = deploy.get_runner()

        # Get input name
        input_name = list(mera_model.input_desc.all_inputs.keys())[0]
        print(f"  Input name: {input_name}")

        # Run inference
        print("\nRunning inference...")
        runner.set_input({input_name: test_img}).run()
        mera_outputs = runner.get_outputs()

        print(f"\nMERA outputs: {len(mera_outputs)} tensors")
        for i, out in enumerate(mera_outputs):
            print(f"  output[{i}]: shape={out.shape}, dtype={out.dtype}")

    # Print raw MERA output values
    print("\n" + "=" * 60)
    print("MERA Output Values")
    print("=" * 60)

    # Output 0: 6x6 grid (should be shape [1, 18, 6, 6] or [1, 6, 6, 18])
    mera_out0 = mera_outputs[0]
    print(f"\nOutput 0 shape: {mera_out0.shape}")
    mera_out0_flat = mera_out0.flatten()

    print("\nOutput0 (6x6 grid) - first 72 values:")
    for i in range(min(72, len(mera_out0_flat))):
        if i % 12 == 0:
            print(f"[{i:3d}]: ", end="")
        print(f"{mera_out0_flat[i]:7.3f} ", end="")
        if i % 12 == 11:
            print()

    print(f"\nOutput0 key indices:")
    print(f"  [0]   = {mera_out0_flat[0]:.4f}")
    print(f"  [21]  = {mera_out0_flat[21]:.4f}")
    print(f"  [36]  = {mera_out0_flat[36]:.4f}")
    print(f"  [180] = {mera_out0_flat[180]:.4f}")
    print(f"  [216] = {mera_out0_flat[216]:.4f}")

    # Output 1: 12x12 grid
    if len(mera_outputs) > 1:
        mera_out1 = mera_outputs[1]
        print(f"\nOutput 1 shape: {mera_out1.shape}")
        mera_out1_flat = mera_out1.flatten()

        print("\nOutput1 (12x12 grid) - first 48 values:")
        for i in range(min(48, len(mera_out1_flat))):
            if i % 12 == 0:
                print(f"[{i:3d}]: ", end="")
            print(f"{mera_out1_flat[i]:7.3f} ", end="")
            if i % 12 == 11:
                print()

    # Compare with PyTorch if available
    if pytorch_outputs is not None:
        print("\n" + "=" * 60)
        print("Comparison: MERA Interpreter vs PyTorch")

        pytorch_out0 = pytorch_outputs['output0'].flatten()
        pytorch_out1 = pytorch_outputs['output1'].flatten()
        mera_out1 = mera_outputs[1].flatten() if len(mera_outputs) > 1 else None

        # Output 0 comparison
        print("\nOutput 0 (6x6 grid):")
        print("Index      MERA     PyTorch      Diff")
        print("-" * 45)
        for i in range(min(12, len(mera_out0))):
            diff = mera_out0[i] - pytorch_out0[i]
            print(f"[{i:3d}]  {mera_out0[i]:8.4f}  {pytorch_out0[i]:8.4f}  {diff:+8.4f}")

        max_diff0 = np.abs(mera_out0 - pytorch_out0[:len(mera_out0)]).max()
        mean_diff0 = np.abs(mera_out0 - pytorch_out0[:len(mera_out0)]).mean()
        print(f"\nOutput 0 statistics:")
        print(f"  Max absolute difference:  {max_diff0:.4f}")
        print(f"  Mean absolute difference: {mean_diff0:.4f}")

        # Output 1 comparison
        if mera_out1 is not None:
            print("\nOutput 1 (12x12 grid):")
            print("Index      MERA     PyTorch      Diff")
            print("-" * 45)
            for i in range(min(12, len(mera_out1))):
                diff = mera_out1[i] - pytorch_out1[i]
                print(f"[{i:3d}]  {mera_out1[i]:8.4f}  {pytorch_out1[i]:8.4f}  {diff:+8.4f}")

            max_diff1 = np.abs(mera_out1 - pytorch_out1[:len(mera_out1)]).max()
            mean_diff1 = np.abs(mera_out1 - pytorch_out1[:len(mera_out1)]).mean()
            print(f"\nOutput 1 statistics:")
            print(f"  Max absolute difference:  {max_diff1:.4f}")
            print(f"  Mean absolute difference: {mean_diff1:.4f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if max_diff0 < 1.0 and mean_diff0 < 0.5:
            print("MERA Interpreter outputs MATCH PyTorch (within quantization tolerance)")
            print("  -> Problem is likely in Ethos-U code generation")
        else:
            print("MERA Interpreter outputs DO NOT MATCH PyTorch")
            print("  -> Problem is in quantization for this specific input")
            print("  -> Consider adding this test image to calibration data")

    # Save MERA outputs for comparison with firmware
    mera_output_path = Path(args.output_dir) / "mera_outputs.npz"
    np.savez(
        mera_output_path,
        output0=mera_outputs[0],
        output1=mera_outputs[1] if len(mera_outputs) > 1 else np.array([])
    )
    print(f"\nSaved MERA outputs to: {mera_output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
