#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cuda_check.h"
#include "benchmark.h"

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_residual_v0(const float* X, const float* Res, float* Y, int N) {
    // 基础版
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        Y[idx] = silu(X[idx]) + Res[idx];
}

__global__ void silu_residual_v1(const float* X, const float* Res, float* Y, int N) {
    // 向量化访存
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx * 4 + 3 >= N) {
        for (int i = idx * 4; i < N; ++i) {
            Y[i] = silu(X[i]) + Res[i];
        }
        return;
    }

    float4 x = reinterpret_cast<const float4*>(X)[idx];
    float4 res = reinterpret_cast<const float4*>(Res)[idx];
    float4 y;
    y.x = silu(x.x) + res.x;
    y.y = silu(x.y) + res.y;
    y.z = silu(x.z) + res.z;
    y.w = silu(x.w) + res.w;

    reinterpret_cast<float4*>(Y)[idx] = y;
}

int main() {
    constexpr int N = 1 << 24;
    constexpr int threads = 256;
    const size_t bytes = N * sizeof(float);

    std::vector<float> X(N), Res(N), Y(N), ref(N);
    for (int i = 0; i < N; ++i) {
        X[i] = (i % 201 - 100) * 0.1f;  // 包含正数、负数和零
        Res[i] = (i % 17 - 8) * 0.1f;
        ref[i] = X[i] / (1.0f + std::exp(-X[i])) + Res[i];
    }

    float *dX, *dRes, *dY;
    cuda_check(cudaMalloc(&dX, bytes));
    cuda_check(cudaMalloc(&dRes, bytes));
    cuda_check(cudaMalloc(&dY, bytes));
    cuda_check(cudaMemcpy(dX, X.data(), bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dRes, Res.data(), bytes, cudaMemcpyHostToDevice));

    auto run_case = [&](const char* name, auto launch) {
        cuda_check(cudaMemset(dY, 0xff, bytes));

        // 检查 kernel 正确性
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(Y.data(), dY, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            float tolerance = 1e-5f + 1e-5f * std::abs(ref[i]);
            if (!std::isfinite(Y[i]) ||
                std::abs(Y[i] - ref[i]) > tolerance) {
                std::fprintf(stderr, "%s: FAIL at %d\n", name, i);
                std::exit(EXIT_FAILURE);
            }
        }

        // 耗时测试
        float ms = benchmark(launch);
        std::printf("%s PASS, %.2f us / run\n", name, ms * 1000);
    };

    int blocks = (N + threads - 1) / threads;
    run_case("v0", [&]{silu_residual_v0<<<blocks, threads>>>(dX, dRes, dY, N);});
    
    int nums_per_block = 4 * threads;
    blocks = (N + nums_per_block - 1) / nums_per_block;
    run_case("v1", [&]{silu_residual_v1<<<blocks, threads>>>(dX, dRes, dY, N);});

    cuda_check(cudaFree(dX));
    cuda_check(cudaFree(dRes));
    cuda_check(cudaFree(dY));

    return 0;
}