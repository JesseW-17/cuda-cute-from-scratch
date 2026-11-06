#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cuda_check.h"

using namespace std;

__global__ void vector_add_v0(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

__global__ void vector_add_v1(const float* A, const float* B, float* C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int base = idx * 4;
    if (base >= N) return;

    if (base + 4 > N) {
        for (int i = base; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
        return;
    }

    float4 a = reinterpret_cast<const float4*>(A)[idx];
    float4 b = reinterpret_cast<const float4*>(B)[idx];

    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    reinterpret_cast<float4*>(C)[idx] = c;
}

int main() {
    constexpr int N = 1 << 24;
    constexpr int threads = 256;
    vector<float> A(N), B(N), C(N);
    const size_t bytes = N * sizeof(float);

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i % 7;
    }

    float *da, *db, *dc;
    cuda_check(cudaMalloc(&da, bytes));
    cuda_check(cudaMalloc(&db, bytes));
    cuda_check(cudaMalloc(&dc, bytes));
    cuda_check(cudaMemcpy(da, A.data(), bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(db, B.data(), bytes, cudaMemcpyHostToDevice));

    auto run_case = [&](const char* name, auto launch) {        
        constexpr int repeats = 100;
        cuda_check(cudaMemset(dc, 0, bytes));

        for (int i = 0; i < 10; ++i) {
            launch();
        }

        cudaEvent_t start, stop;
        cuda_check(cudaEventCreate(&start));
        cuda_check(cudaEventCreate(&stop));

        cuda_check(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) {
            launch();
        }
        cuda_check(cudaEventRecord(stop));

        cuda_check(cudaGetLastError());
        cuda_check(cudaEventSynchronize(stop));

        float total_ms;
        cuda_check(cudaEventElapsedTime(&total_ms, start, stop));
        printf("%s: %.2f us / run\n", name, total_ms / repeats * 1000);

        cuda_check(cudaEventDestroy(start));
        cuda_check(cudaEventDestroy(stop));

        cuda_check(cudaMemcpy(C.data(), dc, bytes, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i) {
            if (C[i] != A[i] + B[i]) {
                fprintf(stderr, "FAIL at %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        printf("%s PASS\n", name);
    };

    run_case("v0", [&] {vector_add_v0<<<(N + threads - 1) / threads, threads>>>(da, db, dc, N);});
    int nums_per_block = threads * 4;
    run_case("v1", [&] {vector_add_v1<<<(N + nums_per_block - 1) / nums_per_block, threads>>>(da, db, dc, N);});
    
    cuda_check(cudaFree(da));
    cuda_check(cudaFree(db));
    cuda_check(cudaFree(dc));

    return 0;
}