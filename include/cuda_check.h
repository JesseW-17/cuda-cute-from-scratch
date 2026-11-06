#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#pragma once

inline void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}