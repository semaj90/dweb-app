// +build cuda

package main

/*
#cgo CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
#cgo LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64" -lcudart -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>

int checkCUDA() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}
*/
import "C"

func isCUDAAvailable() bool {
	return C.checkCUDA() > 0
}

func getCUDADeviceCount() int {
	return int(C.checkCUDA())
}
