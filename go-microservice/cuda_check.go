package main

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lcudart -lcublas

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct {
    int device_count;
    int cuda_version;
    int driver_version;
    size_t free_mem;
    size_t total_mem;
    char device_name[256];
    int compute_major;
    int compute_minor;
} cuda_info_t;

cuda_info_t get_cuda_info() {
    cuda_info_t info = {0};
    
    cudaGetDeviceCount(&info.device_count);
    if (info.device_count == 0) return info;
    
    cudaRuntimeGetVersion(&info.cuda_version);
    cudaDriverGetVersion(&info.driver_version);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(info.device_name, 256, "%s", prop.name);
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    
    cudaMemGetInfo(&info.free_mem, &info.total_mem);
    
    return info;
}

cublasHandle_t init_cublas() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}
*/
import "C"
import (
	"fmt"
	"github.com/gin-gonic/gin"
)

func main() {
	info := C.get_cuda_info()
	
	if info.device_count == 0 {
		panic("No CUDA devices found")
	}
	
	fmt.Printf("CUDA Device: %s\n", C.GoString(&info.device_name[0]))
	fmt.Printf("Compute Capability: %d.%d\n", info.compute_major, info.compute_minor)
	fmt.Printf("VRAM: %.2f GB free / %.2f GB total\n", 
		float64(info.free_mem)/(1<<30), 
		float64(info.total_mem)/(1<<30))
	fmt.Printf("CUDA Runtime: %d.%d\n", info.cuda_version/1000, (info.cuda_version%100)/10)
	
	handle := C.init_cublas()
	if handle == nil {
		panic("cuBLAS initialization failed")
	}
	
	r := gin.Default()
	r.GET("/cuda", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"device": C.GoString(&info.device_name[0]),
			"compute": fmt.Sprintf("%d.%d", info.compute_major, info.compute_minor),
			"vram_gb": float64(info.total_mem) / (1<<30),
			"cuda_version": info.cuda_version,
		})
	})
	
	r.Run(":8080")
}
