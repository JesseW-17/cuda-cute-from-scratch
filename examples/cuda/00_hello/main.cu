#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "cuda_check.h"

using namespace std;

__global__ void hello() {
    printf("Hello From GPU!\n");
}

int main() {
    hello<<<1, 1>>>();
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    return 0;
}