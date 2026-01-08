# Introduction to Edge AI and Computer Vision with Renesas RA8P1

TODO: welcome message, CTA: course link

## How to Run RUHMI

This course repository contains a Dockerfile for building a Docker image to run the [RUHMI framework](https://github.com/renesas/ruhmi-framework-mcu) on (almost) any operating system. Navigate to this directory and build the image:

```sh
docker build -t ruhmi .
```

Then, run the container in interactive mode. Note that the container will automatically delete whenever you exit. You also need to mount the *models/* directory to read in your ONNX models and export RUHMI files.

```sh
docker run --rm -it -v "$PWD/models:/models" -v "$PWD/scripts:/scripts" ruhmi
```

