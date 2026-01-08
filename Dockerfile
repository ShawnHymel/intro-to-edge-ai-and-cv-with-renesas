# Docker image for the Renesas RUHMI framework
#
# RUHMI is tightly coupled to the OS. The official install doc (as of Jan 4, 
# 2026) recommends Ubuntu 22.04 or Windows 10/11. This Docker image uses Ubuntu 
# 22.04 and pinned versions of libraries so that you can (ideally) run RUHMI on
# any host OS.
#
# See https://github.com/renesas/ruhmi-framework-mcu for more information.

# Use the official Ubuntu image
FROM ubuntu:22.04

# Settings
ARG RUHMI_REPO="https://github.com/renesas/ruhmi-framework-mcu"
ARG RUHMI_TAG="Release-2025-11-28"
ARG MERA_WHL="mera-2.5.0+pkg.3019-cp310-cp310-manylinux_2_27_x86_64.whl"
ARG MERA_WHL_URL="${RUHMI_REPO}/raw/refs/tags/${RUHMI_TAG}/install/${MERA_WHL}"

#-------------------------------------------------------------------------------
# System dependencies

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Base utilities and PPA tooling (gpg is needed to import PPA signing keys)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    software-properties-common \
    gnupg \
    gpg-agent

# Add additional package repositories
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

# Toolchain and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-13 \
    g++-13 \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# More toolchain dependencies
RUN apt-get install -y --only-upgrade libstdc++6 libgcc-s1

# Clean up
RUN rm -rf /var/lib/apt/lists/*

#-------------------------------------------------------------------------------
# Install RUHMI and MERA

# Create a virtual environment for RUHMI/MERA
ENV VENV_PATH=/opt/ruhmi-venv
RUN python3.10 -m venv ${VENV_PATH}

# Install Python packages
RUN ${VENV_PATH}/bin/pip install --upgrade pip setuptools wheel
RUN ${VENV_PATH}/bin/pip install \
    decorator \
    typing_extensions \
    psutil \
    attrs \
    pybind11 \
    cmake \
    junitparser

# Install MERA
RUN test -n "${MERA_WHL_URL}" \
    && "${VENV_PATH}/bin/pip" install "${MERA_WHL_URL}"

# Copy conversion script
RUN mkdir /scripts
COPY scripts/mcu_quantize.py /scripts/

# Set working directory
WORKDIR /scripts

# Automatically source the RUHMI virtual environment
RUN echo 'source /opt/ruhmi-venv/bin/activate' >> /root/.bashrc

# Give the user an interactive bash shell
CMD ["bash"]
