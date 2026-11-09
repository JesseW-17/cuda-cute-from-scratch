#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "cuda_check.h"
#include "benchmark.h"
#include <vector>

__global__ void transpose_v0(const float* X, float* Y, int M, int N) {
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (r < M && c < N) {
        Y[c * M + r] = X[r * N + c];
    }
}

template <int TILE_SIZE = 32>
__global__ void transpose_v1(const float* X, float* Y, int M, int N) {
    // coalescing
    int tile_r = threadIdx.y, tile_c = threadIdx.x;
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x_r = blockIdx.y * blockDim.y + tile_r;
    int x_c = blockIdx.x * blockDim.x + tile_c;
    int y_r = blockIdx.x * blockDim.x + tile_r;
    int y_c = blockIdx.y * blockDim.y + tile_c;

    if (x_r < M && x_c < N) {
        tile[tile_r][tile_c] = X[x_r * N + x_c];
    }

    __syncthreads();

    if (y_r < N && y_c < M) {
        Y[y_r * M + y_c] = tile[tile_c][tile_r];
    }
}

template <int TILE_SIZE = 32>
__global__ void transpose_v2(const float* X, float* Y, int M, int N) {
    // bank conflict, 这里 tile size 为 16，32 的时候都会出现读 tile 的 bank conflict。
    int tile_r = threadIdx.y, tile_c = threadIdx.x;
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x_r = blockIdx.y * blockDim.y + tile_r;
    int x_c = blockIdx.x * blockDim.x + tile_c;
    int y_r = blockIdx.x * blockDim.x + tile_r;
    int y_c = blockIdx.y * blockDim.y + tile_c;

    if (x_r < M && x_c < N) {
        tile[tile_r][tile_c] = X[x_r * N + x_c];
    }

    __syncthreads();

    if (y_r < N && y_c < M) {
        Y[y_r * M + y_c] = tile[tile_c][tile_r];
    }
}

int main() {
    // constexpr int M = 257, N = 513;
    constexpr int M = 4096, N = 4096;

    std::vector<float> X(M * N), Y(M * N), Ref(M * N);
    for (int i = 0; i < M * N; ++i) {
        X[i] = static_cast<float>(i);
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Ref[j * M + i] = X[i * N + j];
        }
    }

    float *dX, *dY;
    const size_t bytes = M * N * sizeof(float);
    cuda_check(cudaMalloc(&dX, bytes));
    cuda_check(cudaMalloc(&dY, bytes));
    cuda_check(cudaMemcpy(dX, X.data(), bytes, cudaMemcpyHostToDevice));
    
    auto run_case = [&] (const char* name, auto launch) {
        cuda_check(cudaMemset(dY, 0xff, bytes));
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(Y.data(), dY, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (Y[i * N + j] != Ref[i * N + j]) {
                    std::fprintf(stderr, "%s FAIL at %d, %d\n", name, i, j);
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        float ms = benchmark(launch);
        std::printf("%s PASS, %.2f us / run\n", name, ms * 1000);
    };

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    run_case("v0", [&]{transpose_v0<<<grid, block>>>(dX, dY, M, N);});
    run_case("v1", [&]{transpose_v1<<<grid, block>>>(dX, dY, M, N);});
    run_case("v2", [&]{transpose_v2<<<grid, block>>>(dX, dY, M, N);});

    cuda_check(cudaFree(dX));
    cuda_check(cudaFree(dY));

    return 0;
}
