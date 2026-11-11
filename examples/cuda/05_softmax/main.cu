#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "cuda_check.h"
#include "benchmark.h"

__global__ void softmax_v0(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x;
    
    float max_x = -CUDART_INF_F;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        max_x = fmaxf(max_x, X[row * N + col]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        max_x = fmaxf(max_x, __shfl_xor_sync(0xffffffff, max_x, offset));
    }
    
    __shared__ float warp_max_x[32];
    int warp_idx = threadIdx.x / 32, lane_idx = threadIdx.x % 32;
    if (lane_idx == 0)
        warp_max_x[warp_idx] = max_x;
    __syncthreads();

    if (warp_idx == 0) {
        if (lane_idx < blockDim.x / 32) {
            max_x = warp_max_x[lane_idx];
        }
        else {
            max_x = -CUDART_INF_F;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            max_x = fmaxf(max_x, __shfl_xor_sync(0xffffffff, max_x, offset));
        }
        if (lane_idx == 0)
            warp_max_x[0] = max_x;  
    }

    __syncthreads();
    max_x = warp_max_x[0];

    float sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        sum += expf(X[row * N + col] - max_x);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum +=  __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float sums[32];
    if (lane_idx == 0)
        sums[warp_idx] = sum;
    __syncthreads();

    if (warp_idx == 0) {
        if (lane_idx < blockDim.x / 32) {
            sum = sums[lane_idx];
        }
        else {
            sum = 0.0f;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (lane_idx == 0)
            sums[0] = sum;  
    }

    __syncthreads();
    sum = sums[0];

    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        Y[row * N + col] = expf(X[row * N + col] - max_x) / sum;
    }
}

template <int K = 4>
__global__ void softmax_v1(const float* X, float* Y, int M, int N) {
    int row = blockIdx.x;

    float x[K];
    #pragma unroll
    for (int i = 0; threadIdx.x + i * blockDim.x  < N; ++i) {
        int col = threadIdx.x + i * blockDim.x;
        x[i] = X[row * N + col];
    }
    
    float max_x = -CUDART_INF_F;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        max_x = fmaxf(max_x, x[i]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        max_x = fmaxf(max_x, __shfl_xor_sync(0xffffffff, max_x, offset));
    }
    
    __shared__ float warp_max_x[32];
    int warp_idx = threadIdx.x / 32, lane_idx = threadIdx.x % 32;
    if (lane_idx == 0)
        warp_max_x[warp_idx] = max_x;
    __syncthreads();

    if (warp_idx == 0) {
        if (lane_idx < blockDim.x / 32) {
            max_x = warp_max_x[lane_idx];
        }
        else {
            max_x = -CUDART_INF_F;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            max_x = fmaxf(max_x, __shfl_xor_sync(0xffffffff, max_x, offset));
        }
        if (lane_idx == 0)
            warp_max_x[0] = max_x;  
    }

    __syncthreads();
    max_x = warp_max_x[0];

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        x[i] = expf(x[i] - max_x);
        sum += x[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum +=  __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    __shared__ float sums[32];
    if (lane_idx == 0)
        sums[warp_idx] = sum;
    __syncthreads();

    if (warp_idx == 0) {
        if (lane_idx < blockDim.x / 32) {
            sum = sums[lane_idx];
        }
        else {
            sum = 0.0f;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (lane_idx == 0)
            sums[0] = sum;  
    }

    __syncthreads();
    sum = sums[0];

    #pragma unroll
    for (int i = 0; i < K; ++i) {
        int col = threadIdx.x + i * blockDim.x;
        Y[row * N + col] = x[i] / sum;
    }
}

int main() {
    int M = 8192, N = 1024;
    size_t count = size_t(M) * N;
    size_t bytes = count * sizeof(float);
    std::vector<float> X(count), Y(count);
    std::vector<double> Ref(count);

    for (size_t i = 0; i < count; ++i)
        X[i] = (int(i % 101) - 50) * 0.1f;

    // CPU 参考：逐行计算数值稳定的 softmax。
    for (int row = 0; row < M; ++row) {
        size_t base = size_t(row) * N;
        float max_x = X[base];
        for (int col = 0; col < N; ++col)
            max_x = fmaxf(max_x, X[base + col]);
        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            Ref[base + col] = std::exp(double(X[base + col]) - max_x);
            sum += Ref[base + col];
        }
        for (int col = 0; col < N; ++col)
            Ref[base + col] /= sum;
    }

    float *dX, *dY;
    cuda_check(cudaMalloc(&dX, bytes));
    cuda_check(cudaMalloc(&dY, bytes));
    cuda_check(cudaMemcpy(dX, X.data(), bytes, cudaMemcpyHostToDevice));

    auto run_case = [&](const char* name, auto launch) {
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(Y.data(), dY, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            double tol = 1e-6 + 1e-4 * std::abs(Ref[i]);
            if (!std::isfinite(Y[i]) || std::abs(Y[i] - Ref[i]) > tol) {
                std::fprintf(stderr, "%s FAIL at %zu: gpu=%.9g, cpu=%.9g\n", name, i, Y[i], Ref[i]);
                std::exit(EXIT_FAILURE);
            }
        }
        float ms = benchmark(launch);
        std::printf("%s PASS! %.2f us / run\n", name, ms * 1000);
    };

    run_case("v0", [&] { softmax_v0<<<M, 256>>>(dX, dY, M, N); });
    run_case("v1", [&] { softmax_v1<<<M, 256>>>(dX, dY, M, N); });

    cuda_check(cudaFree(dX));
    cuda_check(cudaFree(dY));
    return 0;
}
