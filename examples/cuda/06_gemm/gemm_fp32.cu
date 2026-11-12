#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "cuda_check.h"
#include "benchmark.h"

__global__ void gemm_fp32_v0(const float* A, const float* B, float* C, int M, int N, int K) {
    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;
    __shared__ float tile_A[32][33];
    __shared__ float tile_B[32][33];
    int C_row = blockIdx.y * blockDim.y + tile_row;
    int C_col = blockIdx.x * blockDim.x + tile_col;
    float c = 0.0f;

    for (int offset = 0; offset < K; offset += 32) {
        int A_row = C_row, A_col = tile_col + offset;

        __syncthreads();
        if (A_row < M && A_col < K) {
            tile_A[tile_row][tile_col] = A[A_row * K + A_col];
        }
        else {
            tile_A[tile_row][tile_col] = 0.0f;
        }

        int B_row = tile_row + offset, B_col = C_col;
        if (B_row < K && B_col < N) {
            tile_B[tile_row][tile_col] = B[B_row * N + B_col];
        } 
        else {
            tile_B[tile_row][tile_col] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            c += tile_A[tile_row][i] * tile_B[i][tile_col];
        }
    }

    if (C_row < M && C_col < N)
        C[C_row * N + C_col] = c;
}

__global__ void gemm_fp32_v1(const float* A, const float* B, float* C, int M, int N, int K) {
    // 减少 SMEM 读次数
    int tile_col = threadIdx.x;
    __shared__ float tile_A[32][33];
    __shared__ float tile_B[32][33];
    int C_col = blockIdx.x * blockDim.x + tile_col;
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int offset = 0; offset < K; offset += 32) {
        int A_col = tile_col + offset;

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int tile_row = threadIdx.y + i * 8;
            int A_row = tile_row + blockIdx.y * 32;
            if (A_row < M && A_col < K) {
                tile_A[tile_row][tile_col] = A[A_row * K + A_col];
            }
            else {
                tile_A[tile_row][tile_col] = 0.0f;
            }
        }
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int tile_row = threadIdx.y + i * 8;
            int B_row = tile_row + offset, B_col = C_col;
            if (B_row < K && B_col < N) {
                tile_B[tile_row][tile_col] = B[B_row * N + B_col];
            } 
            else {
                tile_B[tile_row][tile_col] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            float b = tile_B[i][tile_col];
            for (int j = 0; j < 4; ++j) {
                int tile_row = threadIdx.y + j * 8;
                c[j] += tile_A[tile_row][i] * b;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int C_row = threadIdx.y + i * 8 + blockIdx.y * 32;
        if (C_row < M && C_col < N)
            C[C_row * N + C_col] = c[i];
    }
}

int main() {
    int M = 2048, N = 2048, K = 512;
    // int M = 512, N = 512, K = 512;
    // int M = 257, N = 259, K = 263;
    std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
    std::vector<double> Ref(C.size());
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (int(i % 101) - 50) * 0.01f;
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = (int(i % 67) - 33) * 0.02f;

    // CPU 参考：A[M, K] * B[K, N] = C[M, N]，double 累加。
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += double(A[m * K + k]) * B[k * N + n];
            Ref[m * N + n] = sum;
        }
    }

    size_t bytes_A = A.size() * sizeof(float);
    size_t bytes_B = B.size() * sizeof(float);
    size_t bytes_C = C.size() * sizeof(float);
    float *dA, *dB, *dC;
    cuda_check(cudaMalloc(&dA, bytes_A));
    cuda_check(cudaMalloc(&dB, bytes_B));
    cuda_check(cudaMalloc(&dC, bytes_C));
    cuda_check(cudaMemcpy(dA, A.data(), bytes_A, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dB, B.data(), bytes_B, cudaMemcpyHostToDevice));

    auto run_case = [&](const char* name, auto launch) {
        cuda_check(cudaMemset(dC, 0xff, bytes_C)); 
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(C.data(), dC, bytes_C, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < C.size(); ++i) {
            double tol = 1e-4 + 1e-4 * std::abs(Ref[i]);
            if (!std::isfinite(C[i]) || std::abs(C[i] - Ref[i]) > tol) {
                std::fprintf(stderr, "%s FAIL at %zu: gpu=%.9g, cpu=%.9g\n",
                             name, i, C[i], Ref[i]);
                std::exit(EXIT_FAILURE);
            }
        }
        float ms = benchmark(launch);
        std::printf("%s PASS! %.2f us / run\n", name, ms * 1000);
    };

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    run_case("v0", [&] { gemm_fp32_v0<<<grid, block>>>(dA, dB, dC, M, N, K); });
    block = dim3(32, 8);
    run_case("v1", [&] { gemm_fp32_v1<<<grid, block>>>(dA, dB, dC, M, N, K); });
    

    cuda_check(cudaFree(dA));
    cuda_check(cudaFree(dB));
    cuda_check(cudaFree(dC));
    return 0;
}
