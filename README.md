# GPU-Cuda

CUDA Programming Examples

## Overview

A collection of GPU-accelerated examples demonstrating key CUDA techniques:
- **Vector Addition** (vecAdd)
- **Histogram Computation** (histogramKernel)
- **2D Convolution** (convolutionalLayerKernelWithSharedMemory)
- **Median Filtering** (medianFilterKernel)

## Features

- Compare CPU vs. GPU performance
- Use of shared memory and atomic operations
- Timing utilities for performance measurement
- Jupyter Notebook examples for quick experimentation

## Prerequisites

- NVIDIA GPU with CUDA Toolkit installed
- C++ compiler supporting C++11 or higher
- [CUDA Toolkit 11+](https://developer.nvidia.com/cuda-toolkit)

## Installation

```bash
git clone https://github.com/bahademircioglu/GPU-Cuda.git
cd GPU-Cuda
```

## Build

Compile individual examples:

```bash
nvcc vector_addition.cu -o vector_add
nvcc histogram_computation.cu -o histogram
nvcc convolution.cu -o convolution
nvcc median_filter.cu -o median_filter
```

## Usage

```bash
# Vector addition
./vector_add

# Histogram (with dataset size)
./histogram 1048576

# Convolution (with image and filter sizes)
./convolution 512 3

# Median filter
./median_filter 512
```

## Project Structure

```
GPU-Cuda/
├── histogram_computation.cu
├── histogramCPU.cpp
├── histogramKernel.cu
├── vector_addition.cu
├── convolutional_operation.ipynb
├── median_filter.cu
├── LICENSE (GPL-3.0)
└── README.md
```

## Contributing

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-name`)  
3. Commit your changes (`git commit -m 'Add feature'`)  
4. Push to the branch (`git push origin feature-name`)  
5. Open a Pull Request

## License

This project is licensed under the GPL‑3.0 License.
