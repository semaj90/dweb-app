# Makefile for Enhanced CUDA Worker v2.0
# Enterprise-grade build configuration for Windows/MSVC

# CUDA toolkit configuration
CUDA_PATH ?= C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
CUDA_BIN_PATH ?= $(CUDA_PATH)\bin
CUDA_LIB_PATH ?= $(CUDA_PATH)\lib\x64
CUDA_INC_PATH ?= $(CUDA_PATH)\include

# Compiler configuration
NVCC = "$(CUDA_BIN_PATH)\nvcc.exe"
CL = cl.exe

# Target executable
TARGET = cuda-worker-enhanced.exe
SOURCE = cuda-worker-enhanced.cu

# JSON library (nlohmann/json - header-only)
JSON_INCLUDE = -I.

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_86 --compiler-options "/W3 /EHsc /MD"
CUDA_LIBS = -lcublas -lcurand -lcufft -lcudart

# Performance optimization flags
NVCC_OPT_FLAGS = --use_fast_math --prec-div=false --prec-sqrt=false --ftz=true

# Debug build configuration
DEBUG_FLAGS = -g -G -DDEBUG
RELEASE_FLAGS = -DNDEBUG

# Build targets
all: release

release: 
	@echo Building Enhanced CUDA Worker v2.0 (Release)...
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) $(RELEASE_FLAGS) $(JSON_INCLUDE) \
		-L"$(CUDA_LIB_PATH)" $(CUDA_LIBS) \
		$(SOURCE) -o $(TARGET)
	@echo Build complete: $(TARGET)

debug:
	@echo Building Enhanced CUDA Worker v2.0 (Debug)...
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(JSON_INCLUDE) \
		-L"$(CUDA_LIB_PATH)" $(CUDA_LIBS) \
		$(SOURCE) -o cuda-worker-debug.exe
	@echo Debug build complete: cuda-worker-debug.exe

profile:
	@echo Building Enhanced CUDA Worker v2.0 (Profile)...
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) $(RELEASE_FLAGS) --profile $(JSON_INCLUDE) \
		-L"$(CUDA_LIB_PATH)" $(CUDA_LIBS) \
		$(SOURCE) -o cuda-worker-profile.exe
	@echo Profile build complete: cuda-worker-profile.exe

clean:
	@echo Cleaning build artifacts...
	@if exist $(TARGET) del $(TARGET)
	@if exist cuda-worker-debug.exe del cuda-worker-debug.exe
	@if exist cuda-worker-profile.exe del cuda-worker-profile.exe
	@if exist *.obj del *.obj
	@if exist *.exp del *.exp
	@if exist *.lib del *.lib
	@if exist *.pdb del *.pdb
	@echo Clean complete

test: release
	@echo Running enhanced CUDA worker tests...
	@echo {"type":"cosine_similarity","vectorA":[1,0,0],"vectorB":[0,1,0],"jobId":"test1"} | $(TARGET)
	@echo {"type":"normalize_vector","vector":[3,4,0],"jobId":"test2"} | $(TARGET)
	@echo {"type":"batch_similarity","vectorsA":[[1,0],[0,1]],"vectorsB":[[1,0],[1,0]],"jobId":"test3"} | $(TARGET)

benchmark: release
	@echo Running performance benchmark...
	@echo Starting CUDA Worker benchmark suite...
	@echo {"type":"cosine_similarity","vectorA":[1,2,3,4,5],"vectorB":[5,4,3,2,1],"jobId":"bench1"} | $(TARGET)
	@echo Benchmark complete

install: release
	@echo Installing Enhanced CUDA Worker to ../bin/...
	@if not exist ..\bin mkdir ..\bin
	copy $(TARGET) ..\bin\$(TARGET)
	@echo Installation complete

# Development helpers
format:
	@echo Formatting code with clang-format...
	clang-format -i $(SOURCE)

lint:
	@echo Running static analysis...
	@echo Static analysis tools would run here

# GPU information
gpu-info:
	@echo Detecting GPU information...
	"$(CUDA_BIN_PATH)\nvcc.exe" --version
	"$(CUDA_BIN_PATH)\nvidia-smi.exe"

# Dependencies check
deps:
	@echo Checking dependencies...
	@echo CUDA Toolkit: $(CUDA_PATH)
	@if exist "$(CUDA_PATH)" (echo ✓ CUDA found) else (echo ✗ CUDA not found)
	@if exist "$(CUDA_LIB_PATH)\cublas.lib" (echo ✓ cuBLAS found) else (echo ✗ cuBLAS not found)
	@if exist "$(CUDA_LIB_PATH)\curand.lib" (echo ✓ cuRAND found) else (echo ✗ cuRAND not found)
	@if exist "$(CUDA_LIB_PATH)\cufft.lib" (echo ✓ cuFFT found) else (echo ✗ cuFFT not found)

# Help target
help:
	@echo Enhanced CUDA Worker v2.0 Build System
	@echo =====================================
	@echo Available targets:
	@echo   all       - Build release version (default)
	@echo   release   - Build optimized release version
	@echo   debug     - Build debug version with symbols
	@echo   profile   - Build profiling-enabled version
	@echo   test      - Run basic functionality tests
	@echo   benchmark - Run performance benchmarks
	@echo   install   - Install to ../bin/ directory
	@echo   clean     - Remove build artifacts
	@echo   format    - Format source code
	@echo   lint      - Run static analysis
	@echo   gpu-info  - Display GPU and CUDA information
	@echo   deps      - Check build dependencies
	@echo   help      - Display this help message
	@echo.
	@echo Build Configuration:
	@echo   CUDA Path: $(CUDA_PATH)
	@echo   Architecture: SM 8.6 (RTX 3060 Ti optimized)
	@echo   Optimization: Fast math, precision optimizations
	@echo   Libraries: cuBLAS, cuRAND, cuFFT

.PHONY: all release debug profile clean test benchmark install format lint gpu-info deps help