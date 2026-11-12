#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_bf16.h>
#include <mma.h>
#include "cuda_check.h"
#include "benchmark.h"

namespace wmma = nvcuda::wmma;

template<typename T, int ROWS, int COLS>
__device__ void load_tile(T (&tile)[ROWS][COLS], const T* src, int stride, int invalid_rows, int invalid_cols) {
    for (int t = threadIdx.x; t < ROWS * COLS; t += blockDim.x) {
        int row = t / COLS;
        int col = t % COLS;
        tile[row][col] = (row < invalid_rows && col < invalid_cols) ? src[row * stride + col] : T{};
    }
}

__global__ void gemm_bf16_v0(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    // warp coarsening
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    __shared__ __align__(32) __nv_bfloat16 tile_a[32][16];
    __shared__ __align__(32) __nv_bfloat16 tile_b[16][32];
    for (int offset = 0; offset < K; offset += 16) {
        __syncthreads();
        load_tile(tile_a, A + block_row * K + offset, K, M - block_row, K - offset);
        load_tile(tile_b, B + offset * N + block_col, N, K - offset, N - block_col);
        __syncthreads();
        wmma::load_matrix_sync(a, &tile_a[warp_row * 16][0], 16);
        wmma::load_matrix_sync(b, &tile_b[0][warp_col * 16], 32);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + (block_row + warp_row * 16) * N + block_col + warp_col * 16, c, N, wmma::mem_row_major);
}

template<int BK>
__global__ void gemm_bf16_v1(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    int block_row = blockIdx.y * 32;
    int block_col = blockIdx.x * 32;
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    __shared__ __align__(32) __nv_bfloat16 tile_a[32][BK];
    __shared__ __align__(32) __nv_bfloat16 tile_b[BK][32];
    for (int offset = 0; offset < K; offset += BK) {
        __syncthreads();
        load_tile(tile_a, A + block_row * K + offset, K, M - block_row, K - offset);
        load_tile(tile_b, B + offset * N + block_col, N, K - offset, N - block_col);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 16){
            wmma::load_matrix_sync(a, &tile_a[warp_row * 16][k], BK);
            wmma::load_matrix_sync(b, &tile_b[k][warp_col * 16], 32);
            wmma::mma_sync(c, a, b, c);
        }
    }
    wmma::store_matrix_sync(C + (block_row + warp_row * 16) * N + block_col + warp_col * 16, c, N, wmma::mem_row_major);
}

template<int BK>
__global__ void gemm_bf16_v2(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];

    #pragma unroll 
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(c[i][j], 0.0f);   

    int block_row = blockIdx.y * 64;
    int block_col = blockIdx.x * 64;
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    __shared__ __align__(64) __nv_bfloat16 tile_a[64][BK];
    __shared__ __align__(64) __nv_bfloat16 tile_b[BK][64];
    for (int offset = 0; offset < K; offset += BK) {
        __syncthreads();
        load_tile(tile_a, A + block_row * K + offset, K, M - block_row, K - offset);
        load_tile(tile_b, B + offset * N + block_col, N, K - offset, N - block_col);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], &tile_a[(warp_row + 2 * i) * 16][k], BK);

            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], &tile_b[k][(warp_col + 2 * j) * 16], 64);

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int row = block_row + (warp_row + 2 * i) * 16;
            int col = block_col + (warp_col + 2 * j) * 16;
            wmma::store_matrix_sync(C + row * N + col, c[i][j], N, wmma::mem_row_major);
        }
    }
}

int main() {
    int M = 2048, N = 2048, K = 512;
    // 当前 WMMA 直接整块写回 C，M/N 需为 32 的倍数；K 可有尾块。
    
    std::vector<__nv_bfloat16> A(size_t(M) * K), B(size_t(K) * N);
    std::vector<float> C(size_t(M) * N);
    std::vector<double> Ref(C.size());
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = __float2bfloat16((int(i % 101) - 50) * 0.01f);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = __float2bfloat16((int(i % 67) - 33) * 0.02f);

    // CPU 参考使用量化后的 BF16 输入，double 累加。
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += double(__bfloat162float(A[m * K + k])) * __bfloat162float(B[k * N + n]);
            Ref[m * N + n] = sum;
        }
    }

    size_t bytes_A = A.size() * sizeof(__nv_bfloat16);
    size_t bytes_B = B.size() * sizeof(__nv_bfloat16);
    size_t bytes_C = C.size() * sizeof(float);
    __nv_bfloat16 *dA, *dB;
    float* dC;
    cuda_check(cudaMalloc(&dA, bytes_A));
    cuda_check(cudaMalloc(&dB, bytes_B));
    cuda_check(cudaMalloc(&dC, bytes_C));
    cuda_check(cudaMemcpy(dA, A.data(), bytes_A, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dB, B.data(), bytes_B, cudaMemcpyHostToDevice));

    auto run_case = [&](const char* name, auto launch) {
        cuda_check(cudaMemset(dC, 0xff, bytes_C));  // NaN，检查是否漏写输出。
        launch();
        cuda_check(cudaGetLastError());
        cuda_check(cudaMemcpy(C.data(), dC, bytes_C, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < C.size(); ++i) {
            double tol = 1e-4 + 1e-4 * std::abs(Ref[i]);
            if (!std::isfinite(C[i]) || std::abs(C[i] - Ref[i]) > tol) {
                std::fprintf(stderr, "%s FAIL at %zu: gpu=%.9g, cpu=%.9g\n", name, i, C[i], Ref[i]);
                std::exit(EXIT_FAILURE);
            }
        }
        float ms = benchmark(launch);
        std::printf("%s PASS! %.2f us / run\n", name, ms * 1000);
    };

    dim3 grid(N / 32, M / 32);
    run_case("v0", [&] { gemm_bf16_v0<<<grid, 128>>>(dA, dB, dC, M, N, K); });
    run_case("v1", [&] { gemm_bf16_v1<32><<<grid, 128>>>(dA, dB, dC, M, N, K); });
    grid = dim3(N / 64, M / 64);
    run_case("v2", [&] { gemm_bf16_v2<32><<<grid, 128>>>(dA, dB, dC, M, N, K); });

    cuda_check(cudaFree(dA));
    cuda_check(cudaFree(dB));
    cuda_check(cudaFree(dC));
    return 0;
}
