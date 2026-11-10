#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "cuda_check.h"
#include "benchmark.h"

__global__ void reduce_sum_v0(const float* X, float* Sum, int N) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        sum = X[idx];

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    int warps = blockDim.x / 32;
    if (warp_id == 0) {
        float warp_sum = lane_id < warps ? warp_sums[lane_id] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }

        if (tid == 0) {
            float block_sum = warp_sum;
            atomicAdd(Sum, block_sum);
        }
    }
}

__global__ void reduce_sum_v1(const float* X, float* Sum, int N) {
    // 向量化访存
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * 4 + 3 < N) {
        float4 x = reinterpret_cast<const float4*>(X)[idx];
        sum += x.x;
        sum += x.y;
        sum += x.z;
        sum += x.w;
    }
    else {
        for (int i = idx * 4; i < N; ++i) 
            sum += X[i];
    }
    
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    int warp_id = tid / 32, lane_id = tid % 32;
    if (lane_id == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    int warps = blockDim.x / 32;
    if (warp_id == 0) {
        float warp_sum = lane_id < warps ? warp_sums[lane_id] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }

        if (tid == 0) {
            float block_sum = warp_sum;
            atomicAdd(Sum, block_sum);
        }
    }
}

int main() {
    // int N = 17;
    // int N = 1025;
    int N = 1 << 24;
    std::vector<float> X(N);
    float Sum;
    double Ref = 0.0;
    
    for (int i = 0; i < N; ++i) {
        X[i] = i;
        Ref += i;
    }

    float *dX, *dSum;
    cuda_check(cudaMalloc(&dX, N * sizeof(float)));
    cuda_check(cudaMalloc(&dSum, sizeof(float)));
    cuda_check(cudaMemcpy(dX, X.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    auto run_case = [&](char* name, auto launch) {
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(&Sum, dSum, sizeof(float), cudaMemcpyDeviceToHost));

        double tol = 1e-5 + 1e-4 * std::abs(Ref);
        if (std::abs(Sum - Ref) > tol) {
            std::printf("gpu: %.9g, cpu: %.9g\n", Sum, Ref);
            std::fprintf(stderr, "%s FAIL!\n", name);
            std::exit(EXIT_FAILURE);
        }

        float ms = benchmark(launch);
        std::printf("%s PASS! %.2f us / run\n", name, ms * 1000);
    };

    int block = 256;
    int grid = (N + block - 1) / block;
    run_case("v0", [&]{
        cuda_check(cudaMemset(dSum, 0, sizeof(float)));
        reduce_sum_v0<<<grid, block>>>(dX, dSum, N);
    });
    grid = (N + block * 4 - 1) / (block * 4);
    run_case("v1", [&]{
        cuda_check(cudaMemset(dSum, 0, sizeof(float)));
        reduce_sum_v1<<<grid, block>>>(dX, dSum, N);
    });

    cuda_check(cudaFree(dX));
    cuda_check(cudaFree(dSum));

    return 0;
}