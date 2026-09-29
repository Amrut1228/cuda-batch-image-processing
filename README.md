# CUDA Batch Signal Processing

This project demonstrates batch signal processing using custom CUDA kernels on an NVIDIA GPU.

## What the program does

The program generates hundreds of synthetic signals and processes them on the GPU using three CUDA kernels:

- Moving-average smoothing
- Maximum absolute value calculation
- Signal normalization to the range [-1, 1]

The GPU result is compared with a CPU implementation for validation.

## Default workload

- Signals: 512
- Samples per signal: 4096
- Total samples: 2,097,152
- GPU: NVIDIA Tesla T4

## Build

```bash
make
