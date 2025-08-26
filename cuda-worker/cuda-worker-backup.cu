// cuda-worker.cu
// Refactored CUDA worker that reads JSON from stdin, executes GPU kernels,
// and outputs JSON to stdout.
//
// Compile options:
// 1. NVCC (recommended): nvcc -std=c++14 -O3 cuda-worker.cu -o cuda-worker.exe
// 2. Clang (alternative): See build scripts for proper setup.

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include <cfloat>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio> // For snprintf

// PRODUCTION FIX: Named constants instead of magic numbers
namespace CudaConstants {
    constexpr float EMBEDDING_SCALE = 1.2345f;    // Previously magic number in embedding kernel
    constexpr float SOM_LEARNING_RATE = 0.5f;     // Previously magic number in SOM training
    constexpr float VERSION_FLAG = 1.0f;          // Version identifier
    constexpr int DEFAULT_BLOCK_SIZE = 256;        // Standard CUDA block size
    constexpr int DEFAULT_SOM_EPOCHS = 5;          // Default SOM training epochs
}

// PRODUCTION FIX: Consistent exception-based error handling instead of exit()
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::string error_msg = "CUDA error at " + std::string(__FILE__) +      \
                              ":" + std::to_string(__LINE__) + ": " +        \
                              std::string(cudaGetErrorString(err_));          \
      throw std::runtime_error(error_msg);                                    \
    }                                                                          \
  } while (0)

// REFACTOR: Encapsulated JSON logic into a cleaner utility namespace.
namespace JsonUtil {
    // A slightly more robust parser, but a real library (nlohmann/json) is recommended for production.
    std::vector<float> parseFloatArray(const std::string& json, const std::string& key) {
        std::vector<float> result;
        std::string key_str = "\"" + key + "\"";
        size_t keyPos = json.find(key_str);
        if (keyPos == std::string::npos) return result;

        size_t arrayStart = json.find('[', keyPos);
        size_t arrayEnd = json.find(']', arrayStart);
        if (arrayStart == std::string::npos || arrayEnd == std::string::npos) return result;

        std::string arrayStr = json.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
        std::stringstream ss(arrayStr);
        std::string item;

        while (std::getline(ss, item, ',')) {
            size_t first = item.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) continue;
            size_t last = item.find_last_not_of(" \t\r\n");
            result.push_back(std::stof(item.substr(first, (last - first + 1))));
        }
        return result;
    }

    std::string parseString(const std::string& json, const std::string& key) {
        std::string key_str = "\"" + key + "\"";
        size_t keyPos = json.find(key_str);
        if (keyPos == std::string::npos) return "";

        size_t colonPos = json.find(':', keyPos);
        size_t quoteStart = json.find('"', colonPos);
        size_t quoteEnd = json.find('"', quoteStart + 1);

        if (quoteStart == std::string::npos || quoteEnd == std::string::npos) return "";
        return json.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
    }

    // REFACTOR: Centralized response generation logic.
    std::string createResponse(const std::string& jobId, const std::string& type, const std::vector<float>& vector, const cudaDeviceProp& prop) {
        double sum = 0.0;
        size_t nonzeros = 0;
        const float eps = 1e-9f;

        for (float v : vector) {
            sum += static_cast<double>(v);
            if (std::fabs(v) > eps) {
                ++nonzeros;
            }
        }
        double mean = vector.empty() ? 0.0 : sum / static_cast<double>(vector.size());

        std::stringstream ss;
        ss.precision(8); // Set precision for floating point numbers
        ss << "{\"jobId\":\"" << jobId << "\",\"type\":\"" << type << "\",\"vector\":[";

        for (size_t i = 0; i < vector.size(); ++i) {
            ss << vector[i] << (i == vector.size() - 1 ? "" : ",");
        }
        
        ss << "],\"status\":\"success\",\"timestamp\":" << time(nullptr)
           << ",\"dimensions\":" << vector.size()
           << ",\"sum\":" << sum
           << ",\"mean\":" << mean
           << ",\"nonzeros\":" << nonzeros
           << ",\"gpu\":\"" << prop.name
           << "\",\"memMB\":" << (long long)(prop.totalGlobalMem / (1024LL * 1024LL))
           << "}";
        return ss.str();
    }

    std::string createErrorResponse(const std::string& jobId, const std::string& errorMsg) {
        std::stringstream ss;
        ss << "{\"jobId\":\"" << jobId << "\",\"error\":\"" << errorMsg << "\",\"status\":\"failed\"}";
        return ss.str();
    }
}

// --- CUDA Kernels (Largely unchanged, but one is removed) ---
__global__ void simple_embedding_kernel(const float* input, float* output, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale + sinf(input[idx] * 0.1f);
    }
}

// CRITICAL FIX: Proper dot product kernel for cosine similarity computation
__global__ void dot_product_kernel(const float* vec1, const float* vec2, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec1[idx] * vec2[idx]; // Element-wise product for dot product
    }
}

// Reduction kernel for computing vector magnitude
__global__ void magnitude_kernel(const float* vec, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec[idx] * vec[idx]; // Square for magnitude calculation
    }
}

// Element-wise product kernel (renamed for clarity)
__global__ void element_wise_product_kernel(const float* vec1, const float* vec2, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = vec1[idx] * vec2[idx];
    }
}

// FIX: Removed the inefficient and incorrect som_update_kernel. The accumulate/finalize pattern is correct.

__global__ void som_assign_kernel(const float* input, const float* centroids, int* assignments, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        const float* p = input + idx * dim;
        float best_dist = FLT_MAX; 
        int best_centroid = 0;
        for (int c = 0; c < k; c++){
            const float* cent = centroids + c * dim;
            float dist = 0.f; 
            for (int d = 0; d < dim; d++){ 
                float diff = p[d] - cent[d]; 
                dist += diff * diff; 
            }
            if (dist < best_dist){ 
                best_dist = dist; 
                best_centroid = c; 
            }
        }
        assignments[idx] = best_centroid;
    }
}

__global__ void som_accumulate_kernel(const float* input, const int* assignments, float* accum, int* counts, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        int c = assignments[idx];
        if (c >= 0 && c < k){
            atomicAdd(&counts[c], 1);
            const float* p = input + idx * dim;
            for (int d = 0; d < dim; d++){
                atomicAdd(&accum[c * dim + d], p[d]);
            }
        }
    }
}

__global__ void som_finalize_kernel(float* centroids, const float* accum, const int* counts, int k, int dim, float lr){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < k){
        int count = counts[c];
        if (count > 0){
            for (int d = 0; d < dim; d++){
                float target = accum[c * dim + d] / count;
                float* cent = &centroids[c * dim + d];
                *cent = *cent + lr * (target - *cent);
            }
        }
    }
}

// --- CudaWorker Class ---
class CudaWorker {
private:
    cudaDeviceProp deviceProp;
    float* d_bufferA = nullptr;
    float* d_bufferB = nullptr;
    size_t bufferCapacity = 0; // in number of floats

    // SOM specific buffers
    float* d_somCentroids = nullptr;
    int* d_somAssignments = nullptr;
    size_t somAssignmentsCapacity = 0;
    int somDim = 0;
    int somK = 0;
    
    // PERFORMANCE FIX: Persistent GPU memory for SOM operations
    float* d_somPoints = nullptr;
    float* d_somAccum = nullptr;
    int* d_somCounts = nullptr;
    size_t somPointsCapacity = 0;
    size_t somAccumCapacity = 0;
    size_t somCountsCapacity = 0;

    void ensureCapacity(size_t n) {
        if (n <= bufferCapacity) return;
        if (d_bufferA) CUDA_CHECK(cudaFree(d_bufferA));
        if (d_bufferB) CUDA_CHECK(cudaFree(d_bufferB));
        
        CUDA_CHECK(cudaMalloc(&d_bufferA, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bufferB, n * sizeof(float)));
        bufferCapacity = n;
    }
    
    // PERFORMANCE FIX: Persistent GPU memory allocation for SOM operations
    void ensureSomCapacity(size_t pointsCount, size_t k, size_t dim) {
        size_t pointsSize = pointsCount * dim;
        size_t accumSize = k * dim;
        size_t countsSize = k;
        
        // Only reallocate if we need more space
        if (pointsSize > somPointsCapacity) {
            if (d_somPoints) CUDA_CHECK(cudaFree(d_somPoints));
            CUDA_CHECK(cudaMalloc(&d_somPoints, pointsSize * sizeof(float)));
            somPointsCapacity = pointsSize;
        }
        
        if (accumSize > somAccumCapacity) {
            if (d_somAccum) CUDA_CHECK(cudaFree(d_somAccum));
            CUDA_CHECK(cudaMalloc(&d_somAccum, accumSize * sizeof(float)));
            somAccumCapacity = accumSize;
        }
        
        if (countsSize > somCountsCapacity) {
            if (d_somCounts) CUDA_CHECK(cudaFree(d_somCounts));
            CUDA_CHECK(cudaMalloc(&d_somCounts, countsSize * sizeof(int)));
            somCountsCapacity = countsSize;
        }
    }

    void initSOM(int k, int dim) {
        if (d_somCentroids) CUDA_CHECK(cudaFree(d_somCentroids));
        CUDA_CHECK(cudaMalloc(&d_somCentroids, k * dim * sizeof(float)));
        somK = k;
        somDim = dim;
        somAssignmentsCapacity = 0;

        std::vector<float> host_centroids(k * dim);
        for (int i = 0; i < k * dim; i++) {
            host_centroids[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
        CUDA_CHECK(cudaMemcpy(d_somCentroids, host_centroids.data(), k * dim * sizeof(float), cudaMemcpyHostToDevice));
    }

public:
    CudaWorker() {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) throw std::runtime_error("No CUDA devices available");
        
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
        srand(time(nullptr)); // Seed random number generator
        std::cerr << "CUDA Worker initialized on: " << deviceProp.name << std::endl;
    }

    ~CudaWorker() {
        // Clean up all GPU memory
        if (d_bufferA) cudaFree(d_bufferA);
        if (d_bufferB) cudaFree(d_bufferB);
        if (d_somCentroids) cudaFree(d_somCentroids);
        if (d_somAssignments) cudaFree(d_somAssignments);
        
        // PERFORMANCE FIX: Clean up persistent SOM buffers
        if (d_somPoints) cudaFree(d_somPoints);
        if (d_somAccum) cudaFree(d_somAccum);
        if (d_somCounts) cudaFree(d_somCounts);
    }

    // REFACTOR: All processing logic is now encapsulated.
    std::string processJob(const std::string& jsonInput) {
        std::string jobId = JsonUtil::parseString(jsonInput, "jobId");
        if (jobId.empty()) jobId = "unknown-" + std::to_string(time(nullptr));
        
        try {
            std::string type = JsonUtil::parseString(jsonInput, "type");
            std::vector<float> data = JsonUtil::parseFloatArray(jsonInput, "data");

            std::cerr << "Processing job " << jobId << " type=" << type << " elements=" << data.size() << std::endl;

            std::vector<float> result;
            if (type == "embedding") {
                result = processEmbedding(data);
            } else if (type == "similarity") {
                size_t mid = data.size() / 2;
                std::vector<float> a(data.begin(), data.begin() + mid);
                std::vector<float> b(data.begin() + mid, data.end());
                result = processSimilarity(a, b);  // Returns single cosine similarity score
            } else if (type == "element_wise_product") {
                size_t mid = data.size() / 2;
                std::vector<float> a(data.begin(), data.begin() + mid);
                std::vector<float> b(data.begin() + mid, data.end());
                result = processElementWiseProduct(a, b);  // Returns vector of products
            } else if (type == "autoindex") {
                result = processAutoIndex(data);
            } else if (type == "som_train") {
                if (data.size() < 3) throw std::runtime_error("som_train requires at least 3 elements in data: [k, dim, ...points]");
                int k = static_cast<int>(data[0]);
                int dim = static_cast<int>(data[1]);
                std::vector<float> points(data.begin() + 2, data.end());
                int n_points = static_cast<int>(points.size() / dim);
                result = trainSOM(points, n_points, dim, k, 5);
            } else {
                result = processEmbedding(data); // Default action
            }
            return JsonUtil::createResponse(jobId, type, result, deviceProp);
        } catch (const std::exception& e) {
            return JsonUtil::createErrorResponse(jobId, e.what());
        }
    }

private:
    std::vector<float> processEmbedding(const std::vector<float>& input) {
        int n = input.size();
        if (n == 0) return {};
        ensureCapacity(n);
        CUDA_CHECK(cudaMemcpy(d_bufferA, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        
        int blockSize = CudaConstants::DEFAULT_BLOCK_SIZE;
        int gridSize = (n + blockSize - 1) / blockSize;
        simple_embedding_kernel<<<gridSize, blockSize>>>(d_bufferA, d_bufferB, n, CudaConstants::EMBEDDING_SCALE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> output(n);
        CUDA_CHECK(cudaMemcpy(output.data(), d_bufferB, n * sizeof(float), cudaMemcpyDeviceToHost));
        return output;
    }

    // CRITICAL FIX: True cosine similarity implementation
    std::vector<float> processSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) throw std::runtime_error("Vectors for similarity must be non-empty and of equal size.");
        int n = a.size();
        ensureCapacity(n * 3); // Need space for 2 vectors + intermediate results
        
        float* d_vec1 = d_bufferA;
        float* d_vec2 = d_bufferA + n;
        float* d_temp = d_bufferB;

        CUDA_CHECK(cudaMemcpy(d_vec1, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec2, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        
        int blockSize = CudaConstants::DEFAULT_BLOCK_SIZE;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        // Step 1: Compute dot product (A · B)
        dot_product_kernel<<<gridSize, blockSize>>>(d_vec1, d_vec2, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: Reduce dot product to single value
        float dot_product = 0.0f;
        std::vector<float> temp_host(n);
        CUDA_CHECK(cudaMemcpy(temp_host.data(), d_temp, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++) {
            dot_product += temp_host[i];
        }
        
        // Step 3: Compute magnitude of vector A
        magnitude_kernel<<<gridSize, blockSize>>>(d_vec1, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(temp_host.data(), d_temp, n * sizeof(float), cudaMemcpyDeviceToHost));
        float mag_a = 0.0f;
        for (int i = 0; i < n; i++) {
            mag_a += temp_host[i];
        }
        mag_a = sqrtf(mag_a);
        
        // Step 4: Compute magnitude of vector B
        magnitude_kernel<<<gridSize, blockSize>>>(d_vec2, d_temp, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(temp_host.data(), d_temp, n * sizeof(float), cudaMemcpyDeviceToHost));
        float mag_b = 0.0f;
        for (int i = 0; i < n; i++) {
            mag_b += temp_host[i];
        }
        mag_b = sqrtf(mag_b);
        
        // Step 5: Compute cosine similarity: (A · B) / (||A|| * ||B||)
        float cosine_similarity = 0.0f;
        if (mag_a > 1e-8f && mag_b > 1e-8f) {
            cosine_similarity = dot_product / (mag_a * mag_b);
        }
        
        // Return single similarity score (not vector of element-wise products)
        return {cosine_similarity};
    }
    
    // RENAMED: Element-wise product for clarity (what the old function actually did)
    std::vector<float> processElementWiseProduct(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) throw std::runtime_error("Vectors for element-wise product must be non-empty and of equal size.");
        int n = a.size();
        ensureCapacity(n * 2);
        
        float* d_vec1 = d_bufferA;
        float* d_vec2 = d_bufferA + n;
        float* d_res = d_bufferB;

        CUDA_CHECK(cudaMemcpy(d_vec1, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vec2, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        
        int blockSize = CudaConstants::DEFAULT_BLOCK_SIZE;
        int gridSize = (n + blockSize - 1) / blockSize;
        element_wise_product_kernel<<<gridSize, blockSize>>>(d_vec1, d_vec2, d_res, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<float> output(n);
        CUDA_CHECK(cudaMemcpy(output.data(), d_res, n * sizeof(float), cudaMemcpyDeviceToHost));
        return output;
    }

    std::vector<float> processAutoIndex(const std::vector<float>& input) {
        auto processed = processEmbedding(input);
        processed.push_back(static_cast<float>(time(nullptr)));
        processed.push_back(static_cast<float>(input.size()));
        processed.push_back(CudaConstants::VERSION_FLAG); // Version or flag
        return processed;
    }

    std::vector<float> trainSOM(const std::vector<float>& flatPoints, int n_points, int dim, int k, int epochs) {
        if (dim != somDim || k != somK || !d_somCentroids) {
            initSOM(k, dim);
        }

        // PERFORMANCE FIX: Use persistent GPU memory instead of malloc/free
        ensureSomCapacity(n_points, k, dim);
        
        // Copy points data to persistent GPU buffer
        CUDA_CHECK(cudaMemcpy(d_somPoints, flatPoints.data(), n_points * dim * sizeof(float), cudaMemcpyHostToDevice));

        // Ensure assignments buffer capacity
        if (somAssignmentsCapacity < static_cast<size_t>(n_points)) {
            if (d_somAssignments) CUDA_CHECK(cudaFree(d_somAssignments));
            CUDA_CHECK(cudaMalloc(&d_somAssignments, n_points * sizeof(int)));
            somAssignmentsCapacity = n_points;
        }

        int block = CudaConstants::DEFAULT_BLOCK_SIZE;
        int gridPoints = (n_points + block - 1) / block;
        int gridCentroids = (k + block - 1) / block;

        for (int e = 0; e < epochs; e++) {
            // Use persistent buffers instead of malloc/free each iteration
            CUDA_CHECK(cudaMemset(d_somAccum, 0, k * dim * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_somCounts, 0, k * sizeof(int)));
            
            som_assign_kernel<<<gridPoints, block>>>(d_somPoints, d_somCentroids, d_somAssignments, n_points, k, dim);
            CUDA_CHECK(cudaGetLastError());
            
            som_accumulate_kernel<<<gridPoints, block>>>(d_somPoints, d_somAssignments, d_somAccum, d_somCounts, n_points, k, dim);
            CUDA_CHECK(cudaGetLastError());

            som_finalize_kernel<<<gridCentroids, block>>>(d_somCentroids, d_somAccum, d_somCounts, k, dim, CudaConstants::SOM_LEARNING_RATE);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> host_centroids(k * dim);
        CUDA_CHECK(cudaMemcpy(host_centroids.data(), d_somCentroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost));

        // PERFORMANCE FIX: No more malloc/free - persistent buffers are reused
        return host_centroids;
    }
};

int main() {
    // REFACTOR: Main function is now clean and simple.
    try {
        CudaWorker worker;
        std::string input;
        for (std::string line; std::getline(std::cin, line);) {
            input += line;
        }
        
        if (input.empty()) {
            std::cerr << "No input received on stdin." << std::endl;
            return 1;
        }

        std::string response = worker.processJob(input);
        std::cout << response << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        std::cout << JsonUtil::createErrorResponse("fatal", e.what()) << std::endl;
        return 1;
    }
}