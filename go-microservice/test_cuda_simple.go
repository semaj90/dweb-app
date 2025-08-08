package main

/*
#cgo CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include"
#cgo LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64" -lcudart

#include <stdio.h>
#include <cuda_runtime.h>

void test_cuda_simple() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return;
    }
    
    printf("✅ CUDA 13.0 devices found: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("✅ Device 0: %s\n", prop.name);
        printf("✅ Compute capability: %d.%d\n", prop.major, prop.minor);
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("✅ Memory: %.2f GB free / %.2f GB total\n", 
            (float)free_mem/(1<<30), 
            (float)total_mem/(1<<30));
    }
}
*/
import "C"
import "fmt"

func main() {
    fmt.Println("🔥 Testing CUDA 13.0 GPU Support...")
    C.test_cuda_simple()
    fmt.Println("🚀 CUDA test completed!")
}