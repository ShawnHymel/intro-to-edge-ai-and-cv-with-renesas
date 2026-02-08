"""
YOLO-Fastest to Quantized TFLite Converter
===========================================

Converts a YOLO-Fastest PyTorch model (.pt) to a quantized INT8 TFLite model
suitable for deployment on Renesas RA8 (Ethos-U55 NPU) via MERA/RUHMI.

The conversion rebuilds the model in tf.keras to produce a clean NHWC graph
with zero transposes, then applies INT8 post-training quantization.

Usage:
    from yolo_fastest_tflite_converter import (
        yolo_fastest_to_quantized_tflite,
        run_tflite_inference,
        print_raw_outputs,
    )

    # Convert PyTorch model to INT8 TFLite
    info = yolo_fastest_to_quantized_tflite(
        pt_model_path="model.pt",
        calibration_data_path="calibration_data.npz",
        output_path="model_int8.tflite",
    )

    # Run test inference through TFLite for comparison with PyTorch
    results = run_tflite_inference(
        tflite_path="model_int8.tflite",
        test_npz_path="test_image.npz",
    )
"""

import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch


###############################################################################
# Private functions

def _get_fused_weights(pt_model, idx):
    """
    Extract conv weights from module_list[idx], fusing BatchNorm if present.

    For ConvBN layers, folds BN into conv weights:
        fused_w = w * (gamma / sqrt(var + eps))
        fused_b = beta - mean * gamma / sqrt(var + eps)

    For ConvNoBN layers, uses the conv bias directly.

    Returns (kernel_tf, bias_tf, is_depthwise) with weights in TF format:
        Regular conv:   [H, W, in_ch, out_ch]
        Depthwise conv: [H, W, ch, 1]
    """
    m = pt_model.module_list[idx]
    conv = m.conv
    w = conv.weight.detach().numpy()  # [out, in, H, W]

    if hasattr(m, "bn") and m.bn is not None:
        bn = m.bn
        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        mean = bn.running_mean.detach().numpy()
        var = bn.running_var.detach().numpy()
        eps = bn.eps

        scale = gamma / np.sqrt(var + eps)
        w = w * scale[:, None, None, None]
        b = beta - mean * scale
    else:
        b = conv.bias.detach().numpy()

    # Transpose to TF layout
    is_depthwise = conv.groups == conv.in_channels and conv.in_channels > 1
    if is_depthwise:
        w = w.transpose(2, 3, 0, 1)  # [ch, 1, H, W] -> [H, W, ch, 1]
    else:
        w = w.transpose(2, 3, 1, 0)  # [out, in, H, W] -> [H, W, in, out]

    return w, b, is_depthwise

def _build_keras_model(pt_model, input_shape=(192, 192, 3)):
    """
    Rebuild the YOLO-Fastest architecture in tf.keras with weights from the
    PyTorch model. Produces a pure NHWC graph with zero transposes.

    Architecture: MobileNetV2-style backbone + SPP + FPN + dual YOLO heads.
    """
    def add_conv(x, idx, strides=1, padding="same"):
        w, b, is_dw = _get_fused_weights(pt_model, idx)
        m = pt_model.module_list[idx]
        has_act = isinstance(m.act, torch.nn.ReLU6)

        if is_dw:
            layer = tf.keras.layers.DepthwiseConv2D(
                kernel_size=w.shape[:2],
                strides=strides,
                padding=padding,
                use_bias=True,
                name=f"dw_{idx}",
            )
        else:
            layer = tf.keras.layers.Conv2D(
                w.shape[-1],
                kernel_size=w.shape[:2],
                strides=strides,
                padding=padding,
                use_bias=True,
                name=f"conv_{idx}",
            )

        x = layer(x)
        layer.set_weights([w, b])

        if has_act:
            x = tf.keras.layers.ReLU(max_value=6.0, name=f"relu6_{idx}")(x)
        return x

    # === Build graph ===
    inp = tf.keras.layers.Input(shape=input_shape, name="image_input")

    # Stem (stride-2 with symmetric padding to match PyTorch)
    x = tf.keras.layers.ZeroPadding2D(1, name="pad_0")(inp)
    x = add_conv(x, 0, strides=2, padding="valid")

    # 96x96 blocks
    x = add_conv(x, 1); x = add_conv(x, 2); s = add_conv(x, 3)
    x = add_conv(s, 4); x = add_conv(x, 5); x = add_conv(x, 6)
    x = tf.keras.layers.Add(name="add_0")([x, s])

    # Stride 2 -> 48x48
    x = add_conv(x, 9)
    x = tf.keras.layers.ZeroPadding2D(1, name="pad_10")(x)
    x = add_conv(x, 10, strides=2, padding="valid")
    s = add_conv(x, 11)

    x = add_conv(s, 12); x = add_conv(x, 13); x = add_conv(x, 14)
    x = tf.keras.layers.Add(name="add_1")([x, s]); s = x
    x = add_conv(x, 17); x = add_conv(x, 18); x = add_conv(x, 19)
    x = tf.keras.layers.Add(name="add_2")([x, s])

    # Stride 2 -> 24x24
    x = add_conv(x, 22)
    x = tf.keras.layers.ZeroPadding2D(1, name="pad_23")(x)
    x = add_conv(x, 23, strides=2, padding="valid")
    s = add_conv(x, 24)

    x = add_conv(s, 25); x = add_conv(x, 26); x = add_conv(x, 27)
    x = tf.keras.layers.Add(name="add_3")([x, s]); s = x
    x = add_conv(x, 30); x = add_conv(x, 31); x = add_conv(x, 32)
    x = tf.keras.layers.Add(name="add_4")([x, s])

    # Channel expansion 8->16 (no residual)
    x = add_conv(x, 35); x = add_conv(x, 36); s = add_conv(x, 37)

    # Four residual blocks at 24x24, 16ch
    x = add_conv(s, 38); x = add_conv(x, 39); x = add_conv(x, 40)
    x = tf.keras.layers.Add(name="add_5")([x, s]); s = x
    x = add_conv(x, 43); x = add_conv(x, 44); x = add_conv(x, 45)
    x = tf.keras.layers.Add(name="add_6")([x, s]); s = x
    x = add_conv(x, 48); x = add_conv(x, 49); x = add_conv(x, 50)
    x = tf.keras.layers.Add(name="add_7")([x, s]); s = x
    x = add_conv(x, 53); x = add_conv(x, 54); x = add_conv(x, 55)
    x = tf.keras.layers.Add(name="add_8")([x, s])

    # Stride 2 -> 12x12
    x = add_conv(x, 58)
    x = tf.keras.layers.ZeroPadding2D(1, name="pad_59")(x)
    x = add_conv(x, 59, strides=2, padding="valid")
    s = add_conv(x, 60)

    # Four residual blocks at 12x12, 24ch
    x = add_conv(s, 61); x = add_conv(x, 62); x = add_conv(x, 63)
    x = tf.keras.layers.Add(name="add_9")([x, s]); s = x
    x = add_conv(x, 66); x = add_conv(x, 67); x = add_conv(x, 68)
    x = tf.keras.layers.Add(name="add_10")([x, s]); s = x
    x = add_conv(x, 71); x = add_conv(x, 72); x = add_conv(x, 73)
    x = tf.keras.layers.Add(name="add_11")([x, s]); s = x
    x = add_conv(x, 76); x = add_conv(x, 77); x = add_conv(x, 78)
    x = tf.keras.layers.Add(name="add_12")([x, s])

    route_12x12 = x  # Save for FPN

    # Stride 2 -> 6x6
    x = add_conv(x, 81)
    x = tf.keras.layers.ZeroPadding2D(1, name="pad_82")(x)
    x = add_conv(x, 82, strides=2, padding="valid")
    s = add_conv(x, 83)

    # Five residual blocks at 6x6, 48ch
    x = add_conv(s, 84); x = add_conv(x, 85); x = add_conv(x, 86)
    x = tf.keras.layers.Add(name="add_13")([x, s]); s = x
    x = add_conv(x, 89); x = add_conv(x, 90); x = add_conv(x, 91)
    x = tf.keras.layers.Add(name="add_14")([x, s]); s = x
    x = add_conv(x, 94); x = add_conv(x, 95); x = add_conv(x, 96)
    x = tf.keras.layers.Add(name="add_15")([x, s]); s = x
    x = add_conv(x, 99); x = add_conv(x, 100); x = add_conv(x, 101)
    x = tf.keras.layers.Add(name="add_16")([x, s]); s = x
    x = add_conv(x, 104); x = add_conv(x, 105); x = add_conv(x, 106)
    x = tf.keras.layers.Add(name="add_17")([x, s])

    # SPP
    grid_size = x.shape[1]  # Should be 6
    p1 = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same", name="spp_pool3")(x)
    p2 = tf.keras.layers.MaxPooling2D(5, strides=1, padding="same", name="spp_pool5")(x)
    p3 = tf.keras.layers.MaxPooling2D(9, strides=1, padding="same", name="spp_pool9")(x)
    x = tf.keras.layers.Concatenate(axis=3, name="spp_concat")([p3, p2, p1, x])

    # Head 1 (6x6)
    head1 = add_conv(x, 115)
    x = add_conv(head1, 116); x = add_conv(x, 117)
    x = add_conv(x, 118); x = add_conv(x, 119)
    out_6x6 = add_conv(x, 120)

    # FPN: upsample head1 + concat with 12x12 route
    target_size = route_12x12.shape[1]  # Should be 12
    up = tf.keras.layers.Lambda(
        lambda x: tf.raw_ops.ResizeNearestNeighbor(
            images=x, size=[target_size, target_size], half_pixel_centers=True
        ),
        name="upsample",
    )(head1)
    fpn = tf.keras.layers.Concatenate(axis=3, name="fpn_concat")([up, route_12x12])

    # Head 2 (12x12)
    x = add_conv(fpn, 125); x = add_conv(x, 126)
    x = add_conv(x, 127); x = add_conv(x, 128)
    out_12x12 = add_conv(x, 129)

    model = tf.keras.Model(
        inputs=inp,
        outputs=[out_6x6, out_12x12],
        name="yolo_fastest_person",
    )
    return model

def _patch_tflite_flatbuffer(tflite_bytes):
    """
    Patch the TFLite flatbuffer to make it compatible with MERA/RUHMI:
      1. Strip SignatureDef (MERA doesn't expect it)
      2. Rename input tensor from 'serving_default_image_input:0' to 'image_input'
    """
    buf = bytearray(tflite_bytes)

    # 1. Strip SignatureDef by zeroing its vtable entry
    root_offset = struct.unpack_from("<I", buf, 0)[0]
    vtable_soffset = struct.unpack_from("<i", buf, root_offset)[0]
    vtable_offset = root_offset - vtable_soffset
    vtable_size = struct.unpack_from("<H", buf, vtable_offset)[0]
    # SignatureDefs is field 7 in the Model table
    sig_field_pos = vtable_offset + 4 + 7 * 2
    if sig_field_pos < vtable_offset + vtable_size:
        struct.pack_into("<H", buf, sig_field_pos, 0)

    # 2. Fix input tensor name
    old_name = b"serving_default_image_input:0"
    idx = buf.find(old_name)
    if idx >= 0:
        new_name = b"image_input"
        buf[idx : idx + len(old_name)] = new_name + b"\x00" * (
            len(old_name) - len(new_name)
        )
        struct.pack_into("<I", buf, idx - 4, len(new_name))

    return bytes(buf)

###############################################################################
# Public functions

def yolo_fastest_to_quantized_tflite(
    pt_model_path,
    calibration_data_path,
    output_path="yolo_fastest_int8.tflite",
    input_shape=(192, 192, 3),
    test_npz_path=None,
    verbose=True,
):
    """
    Convert a YOLO-Fastest PyTorch model to a quantized INT8 TFLite model.

    This rebuilds the model in tf.keras to produce a clean NHWC graph (no
    transposes), transfers weights with fused BatchNorm, applies INT8
    post-training quantization, and patches the flatbuffer for MERA
    compatibility.

    Args:
        pt_model_path: Path to the PyTorch model file (.pt).
        calibration_data_path: Path to calibration data (.npz) containing
            NCHW float32 images with shape (N, C, H, W), values in [0, 1].
        output_path: Where to save the output .tflite file.
        input_shape: Model input shape as (H, W, C). Default (192, 192, 3).
        test_npz_path: Optional path to test_image.npz for accuracy
            verification. If provided, checks Keras vs PyTorch MSE. The
            .npz should contain keys: 'image' (NCHW), 'pt_out0', 'pt_out1'.
        verbose: Print progress messages.

    Returns:
        dict with keys:
            'output_path': Path to saved .tflite file
            'size_bytes': File size in bytes
            'input_quant': (scale, zero_point) for input tensor
            'output_quant': dict mapping grid name to (scale, zero_point)
            'hal_defines': String of C #define lines for hal_entry.c
    """
    if verbose:
        print(f"Loading PyTorch model from {pt_model_path}...")

    # Load PyTorch model
    pt_model = torch.load(pt_model_path, map_location="cpu", weights_only=False)
    pt_model.eval()

    # Build Keras model with transferred weights
    if verbose:
        print("Building Keras model (pure NHWC, zero transposes)...")
    keras_model = _build_keras_model(pt_model, input_shape)

    if verbose:
        total_params = keras_model.count_params()
        print(f"  Parameters: {total_params:,}")

    # Optional: verify accuracy against PyTorch reference
    if test_npz_path is not None:
        if verbose:
            print("Verifying Keras vs PyTorch accuracy...")
        test_data = np.load(test_npz_path)
        img_nchw = test_data["image"][np.newaxis]
        img_nhwc = img_nchw.transpose(0, 2, 3, 1)

        keras_out = keras_model.predict(img_nhwc, verbose=0)
        ref6 = test_data["pt_out0"][0].transpose(1, 2, 0)
        ref12 = test_data["pt_out1"][0].transpose(1, 2, 0)

        mse_0 = np.mean((keras_out[0][0] - ref6) ** 2)
        mse_1 = np.mean((keras_out[1][0] - ref12) ** 2)
        max_diff = max(
            np.max(np.abs(keras_out[0][0] - ref6)),
            np.max(np.abs(keras_out[1][0] - ref12)),
        )

        if verbose:
            grid0 = keras_out[0].shape[1]
            grid1 = keras_out[1].shape[1]
            print(f"  {grid0}x{grid0} MSE:  {mse_0:.8f}")
            print(f"  {grid1}x{grid1} MSE: {mse_1:.8f}")
            print(f"  Max diff:  {max_diff:.8f}")

        if max_diff > 0.001:
            print(f"  WARNING: Max diff {max_diff:.6f} is larger than expected!")

    # Load calibration data
    if verbose:
        print(f"Loading calibration data from {calibration_data_path}...")
    cal_data = np.load(calibration_data_path)
    cal_nchw = cal_data[cal_data.files[0]]
    cal_nhwc = cal_nchw.transpose(0, 2, 3, 1).astype(np.float32)

    if verbose:
        print(f"  Calibration samples: {cal_nhwc.shape[0]}")

    def representative_dataset():
        for i in range(cal_nhwc.shape[0]):
            yield [cal_nhwc[i : i + 1]]

    # Convert to INT8 TFLite using concrete function for clean tensor names
    if verbose:
        print("Quantizing to INT8 TFLite...")

    run_fn = tf.function(lambda x: keras_model(x))
    concrete = run_fn.get_concrete_function(
        tf.TensorSpec([1] + list(input_shape), tf.float32, name="image_input")
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Patch flatbuffer for MERA compatibility
    if verbose:
        print("Patching flatbuffer (strip SignatureDef, fix tensor names)...")
    tflite_model = _patch_tflite_flatbuffer(tflite_model)

    # Save
    output_path = Path(output_path)
    output_path.write_bytes(tflite_model)

    if verbose:
        print(f"Saved: {output_path} ({len(tflite_model):,} bytes)")

    # Extract quantization parameters
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    inp_det = interpreter.get_input_details()[0]
    input_scale, input_zp = inp_det["quantization"]

    result = {
        "output_path": str(output_path),
        "size_bytes": len(tflite_model),
        "input_scale": float(input_scale),
        "input_zp": int(input_zp),
    }

    for od in interpreter.get_output_details():
        grid_h = od["shape"][1]
        s, z = od["quantization"]
        result[f"output_{grid_h}x{grid_h}_scale"] = float(s)
        result[f"output_{grid_h}x{grid_h}_zp"] = int(z)

    if verbose:
        print(f"\nQuantization parameters:")
        print(f"  input_scale={result['input_scale']:.10f}, input_zp={result['input_zp']}")
        for key in sorted(k for k in result if k.startswith("output_")):
            print(f"  {key}={result[key]}")

    return result

def print_raw_outputs(
    output_0,
    output_1,
    format="nhwc",
    input_image=None,
    predictions=None,
    ground_truth=None,
):
    """
    Print raw model outputs in a standardized format for comparison between
    PyTorch FP32 and TFLite INT8 outputs.

    Args:
        output_0: Raw coarse-grid output (smaller spatial dim, larger objects).
            Shape (C, H, W) if CHW or (H, W, C) if HWC.
        output_1: Raw fine-grid output (larger spatial dim, smaller objects).
            Shape (C, H, W) if CHW or (H, W, C) if HWC.
        label: Label for the printout (e.g. "PyTorch FP32", "TFLite INT8").
        format: 'nchw' or 'chw' for PyTorch-style, 'nhwc' or 'hwc' for TFLite.
        input_image: Optional input image array. If CHW (3, H, W), prints
            first 12 R-channel values. If HWC (H, W, 3), prints first 12
            interleaved RGB values.
        predictions: Optional list of prediction boxes
            [(x1, y1, x2, y2, conf, cls), ...] to print.
        ground_truth: Optional ground truth labels to print count.
    """

    # Input verification
    if input_image is not None:
        img_flat = np.asarray(input_image).flatten()
        is_chw = (len(input_image.shape) == 3 and input_image.shape[0] in (1, 3))
        fmt = "R channel, first row" if is_chw else "R,G,B,R,G,B,..."
        print(f"First 12 input values ({fmt}):")
        for i in range(min(12, len(img_flat))):
            print(f"  [{i}] = {img_flat[i]:.4f}")

    # Print in HWC order: each row = 1 anchor at 1 grid cell (6 values: tx,ty,tw,th,obj,cls)
    num_attrs = 6  # tx, ty, tw, th, obj, cls
    num_anchors = 3

    print(f"\nOutput format: [y, x][anchor] tx ty tw th obj cls")

    for out_idx, (out_arr, fmt_flag) in enumerate([
        (output_0, format), (output_1, format)
    ]):
        arr = np.asarray(out_arr)
        if fmt_flag in ("nhwc", "hwc"):
            grid_size = arr.shape[0]
            hwc = arr.reshape(-1)
        else:
            grid_size = arr.shape[1]  # CHW: (18, H, W)
            hwc = np.transpose(arr, (1, 2, 0)).flatten()

        n_cells = min(4, grid_size * grid_size)
        print(f"\nOutput{out_idx} ({grid_size}x{grid_size} grid) - first {n_cells} cells:")
        for cell in range(n_cells):
            y = cell // grid_size
            x = cell % grid_size
            for a in range(num_anchors):
                base = cell * num_anchors * num_attrs + a * num_attrs
                vals = " ".join(f"{hwc[base + c]:7.3f}" for c in range(num_attrs))
                print(f"  [{y},{x}][a={a}] {vals}")

    # Ground truth and predictions
    if ground_truth is not None:
        gt_len = len(ground_truth) if hasattr(ground_truth, '__len__') else ground_truth
        print(f"\nGround truth boxes: {gt_len}")

    if predictions is not None:
        print(f"\nPredicted boxes: {len(predictions)}")
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2, conf = pred[0], pred[1], pred[2], pred[3], pred[4]
            print(f"  Box {i}: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f}), conf={conf:.4f}")

def run_tflite_inference(tflite_path, test_npz_path):
    """
    Run a test image through the INT8 TFLite model and return dequantized
    outputs for comparison with PyTorch float32 outputs.

    Args:
        tflite_path: Path to the INT8 .tflite file.
        test_npz_path: Path to test_image.npz containing:
            'image': float32 NCHW image (C, H, W), values in [0, 1]
            'pt_out0': PyTorch coarse-grid output (optional, for comparison)
            'pt_out1': PyTorch fine-grid output (optional, for comparison)

    Returns:
        dict with keys:
            'outputs': dict mapping grid name (e.g. '10x10') to dequantized
                float32 NHWC output array, shape (H, W, C)
            'raw_outputs': dict mapping grid name to raw int8 arrays
            'quant_params': dict mapping grid name to (scale, zero_point)
            'comparison': dict with MSE/max_diff vs PyTorch (if ref available)
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp_det = interpreter.get_input_details()[0]
    out_dets = interpreter.get_output_details()

    # Load and quantize input
    test_data = np.load(test_npz_path)
    img_nchw = test_data["image"][np.newaxis]  # (1, C, H, W)
    img_nhwc = img_nchw.transpose(0, 2, 3, 1)  # (1, H, W, C)

    scale_in, zp_in = inp_det["quantization"]
    img_q = np.clip(np.round(img_nhwc / scale_in + zp_in), -128, 127).astype(
        np.int8
    )

    interpreter.set_tensor(inp_det["index"], img_q)
    interpreter.invoke()

    outputs = {}
    raw_outputs = {}
    quant_params = {}
    comparison = {}

    for od in out_dets:
        raw = interpreter.get_tensor(od["index"])
        s, z = od["quantization"]
        deq = (raw.astype(np.float32) - z) * s

        grid_h = raw.shape[1]
        grid_name = f"{grid_h}x{grid_h}"

        outputs[grid_name] = deq[0]       # (H, W, 18)
        raw_outputs[grid_name] = raw[0]    # (H, W, 18) int8
        quant_params[grid_name] = (s, z)

    # Compare with PyTorch reference if available
    # Map grid sizes to reference keys (sorted: smaller grid first)
    sorted_grids = sorted(outputs.keys(), key=lambda g: int(g.split("x")[0]))
    pt_keys_ordered = ["pt_out0", "pt_out1"]
    for i, grid_name in enumerate(sorted_grids):
        if i < len(pt_keys_ordered) and pt_keys_ordered[i] in test_data:
            ref = test_data[pt_keys_ordered[i]][0].transpose(1, 2, 0)  # NCHW -> HWC
            deq = outputs[grid_name]
            mse = float(np.mean((deq - ref) ** 2))
            max_diff = float(np.max(np.abs(deq - ref)))
            comparison[grid_name] = {"mse": mse, "max_diff": max_diff}

    return {
        "outputs": outputs,
        "raw_outputs": raw_outputs,
        "quant_params": quant_params,
        "comparison": comparison,
    }