package main

/*
#cgo CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
#cgo LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" -lcudart_static -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>

int init_cuda() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return -1;
    cudaSetDevice(0);
    return 0;
}

void* create_cublas() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}
*/
import "C"

import (
	"fmt"
	"unsafe"
	"github.com/gin-gonic/gin"
)

var cublasHandle unsafe.Pointer

func init() {
	if C.init_cuda() == 0 {
		cublasHandle = C.create_cublas()
		fmt.Println("CUDA initialized")
	}
}

func main() {
	r := gin.Default()
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"cuda": cublasHandle != nil,
			"service": "operational",
		})
	})
	r.Run(":8080")
}
