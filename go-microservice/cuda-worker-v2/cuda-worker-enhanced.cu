// cuda-worker-enhanced.cu - Version 2.0 Enterprise CUDA Worker
// Enhanced with cuBLAS for mathematically precise vector operations
// Optimized for RTX 3060 Ti with 8GB VRAM management

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <mutex>
#include <unordered_map>
#include "json.hpp" // nlohmann/json

using json = nlohmann::json;

// Enhanced error checking macros
#define CUDA_CHECK(err)                                                     \
  do {                                                                      \
    cudaError_t err_ = (err);                                               \
    if (err_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n",                      \
              err_, __FILE__, __LINE__, cudaGetErrorString(err_));          \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(err)                                                   \
  do {                                                                      \
    cublasStatus_t err_ = (err);                                            \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n",                        \
              (int)err_, __FILE__, __LINE__);                               \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define CURAND_CHECK(err)                                                   \
  do {                                                                      \
    curandStatus_t err_ = (err);                                            \
    if (err_ != CURAND_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuRAND error %d at %s:%d\n",                        \
              (int)err_, __FILE__, __LINE__);                               \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// GPU Memory Pool for efficient allocation management
class GPUMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };

    std::vector<MemoryBlock> memory_blocks;
    std::mutex memory_mutex;
    size_t total_allocated = 0;
    size_t max_memory = 6ULL * 1024 * 1024 * 1024; // 6GB of 8GB VRAM

public:
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(memory_mutex);

        // Try to find an existing unused block of sufficient size
        for (auto& block : memory_blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                block.last_used = std::chrono::steady_clock::now();
                return block.ptr;
            }
        }

        // Check if we have enough memory left
        if (total_allocated + size > max_memory) {
            cleanup_unused_blocks();
            if (total_allocated + size > max_memory) {
                throw std::runtime_error("GPU memory pool exhausted");
            }
        }

        // Allocate new block
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));

        memory_blocks.push_back({
            ptr, size, true, std::chrono::steady_clock::now()
        });
        total_allocated += size;

        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        for (auto& block : memory_blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                block.last_used = std::chrono::steady_clock::now();
                return;
            }
        }
    }

    void cleanup_unused_blocks() {
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - std::chrono::minutes(5); // Clean blocks unused for 5 minutes

        for (auto it = memory_blocks.begin(); it != memory_blocks.end();) {
            if (!it->in_use && it->last_used < cutoff) {
                CUDA_CHECK(cudaFree(it->ptr));
                total_allocated -= it->size;
                it = memory_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t get_total_allocated() const { return total_allocated; }
    size_t get_available_memory() const { return max_memory - total_allocated; }

    ~GPUMemoryPool() {
        for (const auto& block : memory_blocks) {
            cudaFree(block.ptr);
        }
    }
};

// Enhanced CUDA Worker with enterprise-grade capabilities
class EnhancedCudaWorker {
private:
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_generator;
    cudaStream_t compute_stream;
    cudaStream_t memory_stream;

    GPUMemoryPool memory_pool;

    // Performance monitoring
    struct PerformanceMetrics {
        std::chrono::steady_clock::time_point start_time;
        size_t operations_processed = 0;
        float total_processing_time = 0.0f;
        float max_processing_time = 0.0f;
        float min_processing_time = std::numeric_limits<float>::max();
    } metrics;

    // GPU device properties
    cudaDeviceProp device_props;
    int device_id = 0;

    // Thread-safe result cache
    std::unordered_map<std::string, json> result_cache;
    std::mutex cache_mutex;

public:
    EnhancedCudaWorker() {
        initialize_cuda_context();
        initialize_performance_monitoring();
    }

    ~EnhancedCudaWorker() {
        cleanup_resources();
    }

private:
    void initialize_cuda_context() {
        // Set device and get properties
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

        std::cout << "Initialized CUDA Worker on: " << device_props.name
                  << " with " << (device_props.totalGlobalMem / (1024 * 1024))
                  << " MB VRAM" << std::endl;

        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        // Create cuRAND generator
        CURAND_CHECK(curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator, 12345ULL));

        // Create CUDA streams for overlapped execution
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&memory_stream));

        // Set cuBLAS stream for async operations
        CUBLAS_CHECK(cublasSetStream(cublas_handle, compute_stream));
    }

    void initialize_performance_monitoring() {
        metrics.start_time = std::chrono::steady_clock::now();
    }

    void cleanup_resources() {
        if (curand_generator) curandDestroyGenerator(curand_generator);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (compute_stream) cudaStreamDestroy(compute_stream);
        if (memory_stream) cudaStreamDestroy(memory_stream);

        cudaDeviceReset();
    }

public:
    // Enhanced cosine similarity with cuBLAS - mathematically precise
    float compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) {
            throw std::runtime_error("Vectors must be non-empty and of equal size");
        }

        auto start = std::chrono::high_resolution_clock::now();

        int n = static_cast<int>(a.size());
        size_t bytes = n * sizeof(float);

        // Allocate GPU memory using our pool
        float* d_a = static_cast<float*>(memory_pool.allocate(bytes));
        float* d_b = static_cast<float*>(memory_pool.allocate(bytes));

        // Asynchronous memory transfers
        CUDA_CHECK(cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, memory_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, memory_stream));

        // Synchronize streams
        CUDA_CHECK(cudaStreamSynchronize(memory_stream));

        float dot_product, norm_a, norm_b;

        // Use cuBLAS for mathematically precise operations
        CUBLAS_CHECK(cublasSdot(cublas_handle, n, d_a, 1, d_b, 1, &dot_product));
        CUBLAS_CHECK(cublasSnrm2(cublas_handle, n, d_a, 1, &norm_a));
        CUBLAS_CHECK(cublasSnrm2(cublas_handle, n, d_b, 1, &norm_b));

        // Synchronize computation stream
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Return memory to pool
        memory_pool.deallocate(d_a);
        memory_pool.deallocate(d_b);

        // Calculate cosine similarity
        float similarity = (norm_a == 0.0f || norm_b == 0.0f) ? 0.0f : dot_product / (norm_a * norm_b);

        // Update metrics
        auto end = std::chrono::high_resolution_clock::now();
        float processing_time = std::chrono::duration<float, std::milli>(end - start).count();
        update_performance_metrics(processing_time);

        return similarity;
    }

    // Batch cosine similarity computation for high throughput
    std::vector<float> compute_batch_cosine_similarity(
        const std::vector<std::vector<float>>& vectors_a,
        const std::vector<std::vector<float>>& vectors_b) {

        if (vectors_a.size() != vectors_b.size()) {
            throw std::runtime_error("Vector batch sizes must match");
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> results;
        results.reserve(vectors_a.size());

        // Process in batches to optimize GPU memory usage
        const size_t batch_size = 32; // Optimize based on available memory

        for (size_t i = 0; i < vectors_a.size(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, vectors_a.size() - i);

            // Process current batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                results.push_back(compute_cosine_similarity(vectors_a[i + j], vectors_b[i + j]));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        float total_time = std::chrono::duration<float, std::milli>(end - start).count();

        std::cout << "Batch similarity computation: " << vectors_a.size()
                  << " pairs in " << total_time << "ms"
                  << " (" << (total_time / vectors_a.size()) << "ms per pair)"
                  << std::endl;

        return results;
    }

    // Enhanced vector normalization using cuBLAS
    std::vector<float> normalize_vector(const std::vector<float>& input) {
        if (input.empty()) return {};

        int n = static_cast<int>(input.size());
        size_t bytes = n * sizeof(float);

        float* d_input = static_cast<float*>(memory_pool.allocate(bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, memory_stream));
        CUDA_CHECK(cudaStreamSynchronize(memory_stream));

        // Compute L2 norm using cuBLAS
        float norm;
        CUBLAS_CHECK(cublasSnrm2(cublas_handle, n, d_input, 1, &norm));

        if (norm > 0.0f) {
            // Normalize: d_input = d_input / norm
            const float inv_norm = 1.0f / norm;
            CUBLAS_CHECK(cublasSscal(cublas_handle, n, &inv_norm, d_input, 1));
        }

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        std::vector<float> result(n);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), d_input, bytes, cudaMemcpyDeviceToHost, memory_stream));
        CUDA_CHECK(cudaStreamSynchronize(memory_stream));

        memory_pool.deallocate(d_input);
        return result;
    }

    // Matrix-vector multiplication using cuBLAS
    std::vector<float> matrix_vector_multiply(
        const std::vector<float>& matrix, int rows, int cols,
        const std::vector<float>& vector) {

      if (matrix.size() != static_cast<size_t>(rows * cols) ||
          vector.size() != static_cast<size_t>(cols)) {
        throw std::runtime_error("Matrix-vector dimensions mismatch");
      }

        size_t matrix_bytes = rows * cols * sizeof(float);
        size_t vector_bytes = cols * sizeof(float);
        size_t result_bytes = rows * sizeof(float);

        float* d_matrix = static_cast<float*>(memory_pool.allocate(matrix_bytes));
        float* d_vector = static_cast<float*>(memory_pool.allocate(vector_bytes));
        float* d_result = static_cast<float*>(memory_pool.allocate(result_bytes));

        // Transfer data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_matrix, matrix.data(), matrix_bytes, cudaMemcpyHostToDevice, memory_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vector, vector.data(), vector_bytes, cudaMemcpyHostToDevice, memory_stream));
        CUDA_CHECK(cudaStreamSynchronize(memory_stream));

        // Perform matrix-vector multiplication: y = alpha * A * x + beta * y
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, rows, cols,
                                &alpha, d_matrix, rows, d_vector, 1, &beta, d_result, 1));

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        std::vector<float> result(rows);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), d_result, result_bytes, cudaMemcpyDeviceToHost, memory_stream));
        CUDA_CHECK(cudaStreamSynchronize(memory_stream));

        memory_pool.deallocate(d_matrix);
        memory_pool.deallocate(d_vector);
        memory_pool.deallocate(d_result);

        return result;
    }

    // Process JSON request with enhanced capabilities
    json process_json_request(const json& request) {
        auto start = std::chrono::high_resolution_clock::now();

        json response;
        response["jobId"] = request.value("jobId", "unknown");
        response["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        try {
            std::string type = request.value("type", "unknown");

            if (type == "cosine_similarity") {
                process_cosine_similarity_request(request, response);
            } else if (type == "batch_similarity") {
                process_batch_similarity_request(request, response);
            } else if (type == "normalize_vector") {
                process_normalize_vector_request(request, response);
            } else if (type == "matrix_multiply") {
                process_matrix_multiply_request(request, response);
            } else {
                // Legacy vector processing
                process_legacy_vector_request(request, response);
            }

            response["status"] = "success";

        } catch (const std::exception& e) {
            response["status"] = "error";
            response["error"] = e.what();
        }

        // Add performance and system information
        auto end = std::chrono::high_resolution_clock::now();
        float processing_time = std::chrono::duration<float, std::milli>(end - start).count();

        response["processingTimeMs"] = processing_time;
        response["gpu"] = device_props.name;
        response["gpuMemoryTotalMB"] = device_props.totalGlobalMem / (1024 * 1024);
        response["gpuMemoryUsedMB"] = memory_pool.get_total_allocated() / (1024 * 1024);
        response["gpuMemoryAvailableMB"] = memory_pool.get_available_memory() / (1024 * 1024);

        // Add performance metrics
        response["performanceMetrics"] = get_performance_metrics();

        return response;
    }

private:
    void process_cosine_similarity_request(const json& request, json& response) {
        auto vector_a = request["vectorA"].get<std::vector<float>>();
        auto vector_b = request["vectorB"].get<std::vector<float>>();

        float similarity = compute_cosine_similarity(vector_a, vector_b);

        response["cosineSimilarity"] = similarity;
        response["vectorDimensions"] = vector_a.size();
    }

    void process_batch_similarity_request(const json& request, json& response) {
        auto vectors_a = request["vectorsA"].get<std::vector<std::vector<float>>>();
        auto vectors_b = request["vectorsB"].get<std::vector<std::vector<float>>>();

        auto similarities = compute_batch_cosine_similarity(vectors_a, vectors_b);

        response["similarities"] = similarities;
        response["batchSize"] = similarities.size();
        response["averageSimilarity"] = std::accumulate(similarities.begin(), similarities.end(), 0.0f) / similarities.size();
    }

    void process_normalize_vector_request(const json& request, json& response) {
        auto input_vector = request["vector"].get<std::vector<float>>();
        auto normalized = normalize_vector(input_vector);

        response["normalizedVector"] = normalized;
        response["originalDimensions"] = input_vector.size();
    }

    void process_matrix_multiply_request(const json& request, json& response) {
        auto matrix = request["matrix"].get<std::vector<float>>();
        int rows = request["rows"];
        int cols = request["cols"];
        auto vector = request["vector"].get<std::vector<float>>();

        auto result = matrix_vector_multiply(matrix, rows, cols, vector);

        response["result"] = result;
        response["matrixRows"] = rows;
        response["matrixCols"] = cols;
    }

    void process_legacy_vector_request(const json& request, json& response) {
        // Handle legacy requests for backward compatibility
        auto data = request.value("data", std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

        // Apply simple transformation
        std::vector<float> result(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * 1.25f + static_cast<float>(i) * 0.1f;
        }

        response["vector"] = result;
        response["dimensions"] = data.size();
        response["sum"] = std::accumulate(result.begin(), result.end(), 0.0f);
        response["mean"] = response["sum"].get<float>() / result.size();
        response["nonzeros"] = std::count_if(result.begin(), result.end(), [](float f) { return f != 0.0f; });
    }

    void update_performance_metrics(float processing_time) {
        metrics.operations_processed++;
        metrics.total_processing_time += processing_time;
        metrics.max_processing_time = std::max(metrics.max_processing_time, processing_time);
        metrics.min_processing_time = std::min(metrics.min_processing_time, processing_time);
    }

    json get_performance_metrics() {
        json perf;
        perf["operationsProcessed"] = metrics.operations_processed;
        perf["totalProcessingTimeMs"] = metrics.total_processing_time;
        perf["averageProcessingTimeMs"] =
            metrics.operations_processed > 0
                ? metrics.total_processing_time / metrics.operations_processed
                : 0.0f;
        perf["maxProcessingTimeMs"] = metrics.max_processing_time;
        perf["minProcessingTimeMs"] =
            metrics.min_processing_time == std::numeric_limits<float>::max()
                ? 0.0f
                : metrics.min_processing_time;

        auto uptime = std::chrono::steady_clock::now() - metrics.start_time;
        perf["uptimeSeconds"] = std::chrono::duration<float>(uptime).count();

        return perf;
    }
};

// Global instance
std::unique_ptr<EnhancedCudaWorker> g_worker;

// Main entry point
int main() {
    try {
        // Initialize enhanced CUDA worker
        g_worker = std::make_unique<EnhancedCudaWorker>();

        std::string line;
        while (std::getline(std::cin, line) && !line.empty()) {
            try {
                json request = json::parse(line);
                json response = g_worker->process_json_request(request);
                std::cout << response.dump() << std::endl;
                std::cout.flush();
            } catch (const json::exception& e) {
                json error_response;
                error_response["status"] = "error";
                error_response["error"] = "Invalid JSON: " + std::string(e.what());
                error_response["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                std::cout << error_response.dump() << std::endl;
                std::cout.flush();
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}