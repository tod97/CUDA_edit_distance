# Final Term Parallel Computing

This project aims to compare the performance of sequential and parallel Levenshtein distance algorithm using the CUDA parallel library.

## Installation

To be able to run this code, you need to install nvcc as CUDA compiler following their dedicated [guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

### Things to consider

- Before running the main.cu, be sure to have a proper GPU with CUDA support and the CUDA toolkit installed
- To compile, use the nvcc compiler with the following command: `nvcc -o main.out main.cu`
