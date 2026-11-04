#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__global__ void hello() {
    printf("Hello From GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    return 0;
}