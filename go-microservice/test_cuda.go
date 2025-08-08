package main

/*
#cgo CFLAGS: -I"C:/Progra~1/NVIDIA~2/CUDA/v12.8/include"
#cgo LDFLAGS: -L"C:/Progra~1/NVIDIA~2/CUDA/v12.8/lib/x64" -lcudart_static -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

void test_cuda() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error == cudaSuccess && deviceCount > 0) {
        printf("CUDA devices: %d\n", deviceCount);
        
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status == CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS initialized\n");
            cublasDestroy(handle);
        }
    } else {
        printf("No CUDA devices\n");
    }
}
*/
import "C"

func main() {
    C.test_cuda()
}
