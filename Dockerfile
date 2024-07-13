# Use the official PyTorch image as a parent image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Confirm Python version
RUN python --version

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install specific versions of NumPy, scikit-image, and easyocr
RUN pip install --no-cache-dir numpy==1.24.4 scikit-image==0.21.0 easyocr==1.7.1

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    torch==1.8.1 \
    torchvision==0.9.1 \
    pandas \
    opencv-python-headless \
    Pillow \
    tqdm \
    pyyaml \
    natsort \
    albumentations \
    nltk

# Copy the rest of the application code to the container
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Set the entrypoint to bash
ENTRYPOINT ["bash"]
