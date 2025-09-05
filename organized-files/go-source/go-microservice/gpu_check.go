//go:build legacy
// +build legacy

package main

/*
#cgo CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
#cgo LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lcudart -lcublas
#include <cuda_runtime.h>
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
