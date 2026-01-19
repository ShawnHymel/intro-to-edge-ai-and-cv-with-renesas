# Introduction to Edge AI and Computer Vision with Renesas RA8P1

TODO: welcome message, CTA: course link

## How to Deploy a Model Using RUHMI

[RUHMI](https://github.com/renesas/ruhmi-framework-mcu) is a framework for quantizing and converting machine learning models into accelerator binaries and C source code for various Renesas microcontrollers equipped with an Arm Ethos-U NPU. It can ingest a number of model formats (PyTorch, TensorFlow Lite, ONNX). The graph compiler will automatically determine which operations can execute on the NPU while falling back to the CPU (i.e. hybrid executionn).

To handle INT8 quantization, the included RUHMI script can either ingest real calibration samples or generate synthetic samples. I recommend using real calibration samples when possible, so you will want to export some of your validation samples in .npz format in your training script.

### Configure and Run Docker Image

Note that RUHMI is designed for either Ubuntu 22.04 or Windows. As a result, I've created a Docker image so that you can run it on (almost) any host operating system. That being said, if you want to install RUHMI locally, follow the instructions [here](https://github.com/renesas/ruhmi-framework-mcu/tree/main/install). 

Navigate to this directory and build the image:

```sh
docker build -t ruhmi .
```

Then, run the container in interactive mode. Note that the container will automatically delete whenever you exit. You also need to mount the *projects/* directory to read in your ONNX models and export RUHMI files.

```sh
docker run --rm -it -v "$PWD/projects:/projects" ruhmi
```

This will give you an interactive shell inside the Docker container and automatically initialize the required virtual environment (e.g. you should see something like `(ruhmi-venv) root@0df5dffe264c:/scripts#`). If you see this, you are ready to run the RUHMI conversion script.

### Run the Deployment Script

Run the deployment script and pass in representative samples (for the *02_gesture_classification* project):

```sh
python /scripts/mcu_quantize.py \
  -d /projects/02_gesture_classification/model/calibration_data.npz \
  -c 50 \
  --ethos \
  /projects/02_gesture_classification/model/ \
  /projects/02_gesture_classification/model_deploy
```

A few notes about output:

```sh
Successfully quantized model with quality: 44.945552825927734 psnr, 99.99679782865822 score
```

This shows how closely the quantized model matches the original 32-bit float model. 

*Peak Signal-to-Noise Ratio (PSNR)* measures how much "noise" (error) was introduced during the quantization process. Higher is better. Anything over 28 dB is acceptable, over 35 dB is good, and over 40 dB is excellent.

The *score* measures how similar (in percentage) the quantized outputs are to the original 32-bit float outputs. Higher is better. Anything over 98% is acceptable, and over 99.5% is excellent.

```sh
CPU operators = 0 (0.0%)
NPU operators = 5 (100.0%)
```

This shows what percentage of operations will run on the CPU versus the Ethos-U NPU. Maximizing NPU operators gives the best performance and energy efficiency. 100% NPU utilization is ideal.

### Copy Converted Model to e<sup>2</sup> Studio

The deployment script will output a bunch of files to *projects\02_gesture_classification\model_deploy/model_000_model/*. Here is an overview of the subfolders you'll find there:

 * *deploy_mcu/* - C source code for MCU + Ethos-U deployment (this is what you want!)
 * *deploy_qtz/* - Reference data used internally for validation (you don't need this)
 * *quantization/* - The saved .mera quantized model file (backup/debugging)
 * *reference_data/* - Only present if you used --ref_data flag (not needed for basic deployment)

You do not need all of these files! Many of them are for intermediate compilation steps or for advanced debugging. Copy the following files (if present) from *projects/02_gesture_classification/model_deploy/model_000_model/deploy_mcu/build/MCU/compilation/src* to a *src/inference/* directory in your e<sup>2</sup> Studio project:

```sh
my_project/
├── src/
│   ├── inference/
│   │   ├── compute_sub_*.c/h           # Generated code for CPU operations
│   │   ├── ethosu_common.h             # Ethos-U interface
│   │   ├── kernel_library_fp32.c/h     # Float32 math/matrix operations for CPU
│   │   ├── kernel_library_int.c/h      # INT8 math/matrix operations for CPU
│   │   ├── kernel_library_utils.c/h    # Helper functions for kernel libraries
│   │   ├── model.c/h                   # Main API interface
│   │   ├── model_io_data.c/h           # Only present if you used the --ref_data flag
│   │   ├── sub_*.c/h                   # Generated binary instructions for NPU
│   ├── hal_entry.c                     # Your application code
|   ├── hal_warmstart.c                 # Initialization code
...
```

In your application code, you'll need to include `model.h` (and `model_io_data.h` if you generated it). You'll then need to initialize the NPU before performing inference.

## License

**TODO**