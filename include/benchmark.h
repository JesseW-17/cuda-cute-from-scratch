#pragma once
#include "cuda_check.h"

template <class Launch>
float benchmark(Launch launch, int repeats = 100) {
    // 测试 kernel 重复 repeats 次的耗时。
    
    // 预热
    for (int i = 0; i < 10; ++i) {
        launch();        
    } 

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));

    cuda_check(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    cuda_check(cudaEventRecord(stop));

    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    float total_ms;
    cuda_check(cudaEventElapsedTime(&total_ms, start, stop));
    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));

    return total_ms / repeats;
}