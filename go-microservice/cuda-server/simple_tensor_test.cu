// Simplified CUDA Tensor Core Test
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void simple_tensor_multiply(
    float* a, float* b, float* c, 
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate host memory
    std::vector<float> h_a(M * K, 1.0f);
    std::vector<float> h_b(K * N, 2.0f);
    std::vector<float> h_c(M * N, 0.0f);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // Time the kernel execution
    auto start = std::chrono::high_resolution_clock::now();
    
    simple_tensor_multiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy result back to host
    cudaMemcpy(h_c.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result
    bool correct = true;
    for (int i = 0; i < std::min(100, M * N); i++) {
        if (abs(h_c[i] - (K * 2.0f)) > 1e-3) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Matrix multiplication " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << "Expected: " << K * 2.0f << ", Got: " << h_c[0] << std::endl;
    
    // GPU Info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}