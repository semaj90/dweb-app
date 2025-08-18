// cuda-worker.cu
// Minimal CUDA worker that reads JSON from stdin, executes GPU kernels, outputs JSON to stdout
// Compile: nvcc -std=c++14 cuda-worker.cu -o cuda-worker.exe

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Simple JSON parser/writer (avoiding external dependencies for minimal setup)
struct JsonParser {
    static std::vector<float> parseFloatArray(const std::string& json, const std::string& key) {
        std::vector<float> result;
        size_t keyPos = json.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return result;

        size_t arrayStart = json.find("[", keyPos);
        size_t arrayEnd = json.find("]", arrayStart);
        if (arrayStart == std::string::npos || arrayEnd == std::string::npos) return result;

        std::string arrayStr = json.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
        std::stringstream ss(arrayStr);
        std::string item;

        while (std::getline(ss, item, ',')) {
            // Remove whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                result.push_back(std::stof(item));
            }
        }
        return result;
    }

    static std::string parseString(const std::string& json, const std::string& key) {
        size_t keyPos = json.find("\"" + key + "\"");
        if (keyPos == std::string::npos) return "";

        size_t colonPos = json.find(":", keyPos);
        size_t quoteStart = json.find("\"", colonPos);
        size_t quoteEnd = json.find("\"", quoteStart + 1);

        if (quoteStart == std::string::npos || quoteEnd == std::string::npos) return "";
        return json.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
    }

    static std::string createResponse(const std::string& jobId, const std::vector<float>& vector, const std::string& type = "embedding") {
        std::stringstream ss;
        ss << "{\"jobId\":\"" << jobId << "\",\"type\":\"" << type << "\",\"vector\":[";
        for (size_t i = 0; i < vector.size(); ++i) {
            if (i > 0) ss << ",";
            ss << vector[i];
        }
        ss << "],\"status\":\"success\",\"timestamp\":" << time(nullptr) << "}";
        return ss.str();
    }
};

// CUDA Kernels
__global__ void simple_embedding_kernel(const float* input, float* output, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple transformation: normalize and scale
        output[idx] = input[idx] * scale + sinf(input[idx] * 0.1f);
    }
}

__global__ void vector_similarity_kernel(const float* vec1, const float* vec2, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compute element-wise similarity (cosine-like)
        result[idx] = vec1[idx] * vec2[idx];
    }
}

__global__ void som_cluster_kernel(const float* input, float* centroids, int* assignments, int n_points, int n_centroids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        float min_dist = FLT_MAX;
        int best_centroid = 0;

        for (int c = 0; c < n_centroids; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = input[idx * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_centroid = c;
            }
        }
        assignments[idx] = best_centroid;
    }
}

#ifdef __CUDACC__
__global__ void som_update_kernel(float* centroids, const float* input, const int* assignments, int n_points, int n_centroids, int dim){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_centroids){
        for (int p=0; p<n_points; ++p){
            if (assignments[p] == c){
                for (int d=0; d<dim; ++d){
                    float* centroidVal = &centroids[c * dim + d];
                    float current = *centroidVal;
                    float target = input[p * dim + d];
                    *centroidVal = current + 0.05f * (target - current);
                }
            }
        }
    }
}
#endif

#ifdef __CUDACC__
__global__ void som_assign_kernel(const float* input, const float* centroids, int* assignments, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        const float* p = input + idx * dim;
        float best = 1e30f; int bestC = 0;
        for (int c=0;c<k;c++){
            const float* cent = centroids + c * dim;
            float dist=0.f; for (int d=0; d<dim; d++){ float diff = p[d]-cent[d]; dist += diff*diff; }
            if (dist < best){ best = dist; bestC = c; }
        }
        assignments[idx] = bestC;
    }
}

__global__ void som_accumulate_kernel(const float* input, const int* assignments, float* accum, int* counts, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        int c = assignments[idx];
        if (c >=0 && c < k){
            atomicAdd(&counts[c], 1);
            const float* p = input + idx * dim;
            for (int d=0; d<dim; d++){
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
            for (int d=0; d<dim; d++){
                float target = accum[c * dim + d] / count;
                float* cent = &centroids[c * dim + d];
                *cent = *cent + lr * (target - *cent);
            }
        }
    }
}
#endif
class CudaWorker {
private:
    cudaDeviceProp deviceProp;
    float* persistentInput = nullptr;
    float* persistentOutput = nullptr;
    size_t persistentCapacity = 0; // number of floats
    // SOM buffers
    float* somCentroids = nullptr;
    int* somAssignments = nullptr;
    size_t somAssignmentsCapacity = 0;
    int somDim = 0;
    int somK = 0;

    void ensureCapacity(size_t n){
        if (n <= persistentCapacity) return;
#ifdef __CUDACC__
        if (persistentInput) cudaFree(persistentInput);
        if (persistentOutput) cudaFree(persistentOutput);
        cudaMalloc((void**)&persistentInput, n * sizeof(float));
        cudaMalloc((void**)&persistentOutput, n * sizeof(float));
#endif
        persistentCapacity = n;
    }
    void initSOM(int k, int dim){
#ifdef __CUDACC__
        if (somCentroids) cudaFree(somCentroids);
        if (somAssignments) cudaFree(somAssignments);
        cudaMalloc((void**)&somCentroids, k*dim*sizeof(float));
#endif
        somK = k; somDim = dim;
        somAssignmentsCapacity = 0;
        // Initialize centroids with small random host values
        std::vector<float> host(k*dim);
        for (int i=0;i<k*dim;i++) host[i] = (float)(rand()%100)/100.f;
#ifdef __CUDACC__
        cudaMemcpy(somCentroids, host.data(), k*dim*sizeof(float), cudaMemcpyHostToDevice);
#endif
    }

public:
    CudaWorker(){
#ifdef __CUDACC__
        int deviceCount; cudaGetDeviceCount(&deviceCount);
        if (deviceCount==0) throw std::runtime_error("No CUDA devices available");
        cudaSetDevice(0);
        cudaGetDeviceProperties(&deviceProp,0);
        std::cerr << "CUDA Worker initialized: " << deviceProp.name << " globalMemMB=" << deviceProp.totalGlobalMem/(1024*1024) << std::endl;
#else
        std::cerr << "CUDA Worker initialized without CUDA support" << std::endl;
#endif
    }

    ~CudaWorker(){
#ifdef __CUDACC__
        if (persistentInput) cudaFree(persistentInput);
        if (persistentOutput) cudaFree(persistentOutput);
        if (somCentroids) cudaFree(somCentroids);
        if (somAssignments) cudaFree(somAssignments);
#endif
    }

    std::vector<float> processEmbedding(const std::vector<float>& input){
        int n = (int)input.size(); if (!n) return {};
        ensureCapacity(n);
#ifdef __CUDACC__
        cudaMemcpy(persistentInput, input.data(), n*sizeof(float), cudaMemcpyHostToDevice);
#endif
        int blockSize = std::min(256, n);
        int gridSize = (n + blockSize -1)/blockSize;
#ifdef __CUDACC__
        simple_embedding_kernel<<<gridSize, blockSize>>>(persistentInput, persistentOutput, n, 1.2345f);
        cudaDeviceSynchronize();
#endif
        std::vector<float> out(n);
#ifdef __CUDACC__
        cudaMemcpy(out.data(), persistentOutput, n*sizeof(float), cudaMemcpyDeviceToHost);
#endif
        return out;
    }

    std::vector<float> processSimilarity(const std::vector<float>& a, const std::vector<float>& b){
        if (a.size()!=b.size() || a.empty()) return {};
        int n=(int)a.size(); ensureCapacity(n*2);
        float* d_vec1 = persistentInput; float* d_vec2 = persistentOutput; float* d_res = nullptr;
#ifdef __CUDACC__
        cudaMemcpy(d_vec1, a.data(), n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vec2, b.data(), n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_res, n*sizeof(float));
#endif
        int blockSize = std::min(256,n); int gridSize=(n+blockSize-1)/blockSize;
#ifdef __CUDACC__
        vector_similarity_kernel<<<gridSize,blockSize>>>(d_vec1,d_vec2,d_res,n);
        cudaDeviceSynchronize();
#endif
        std::vector<float> out(n);
#ifdef __CUDACC__
        cudaMemcpy(out.data(), d_res, n*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_res);
#endif
        return out;
    }

    std::vector<float> processAutoIndex(const std::vector<float>& input){
        auto processed = processEmbedding(input);
        processed.push_back((float)time(nullptr));
        processed.push_back((float)input.size());
        processed.push_back(1.0f);
        return processed;
    }

    std::vector<float> trainSOM(const std::vector<float>& flatPoints, int n_points, int dim, int k, int epochs){
#ifndef __CUDACC__
        if (dim != somDim || k != somK || !somCentroids){ initSOM(k, dim); }
        float* d_points = nullptr; cudaMalloc((void**)&d_points, n_points * dim * sizeof(float));
        cudaMemcpy(d_points, flatPoints.data(), n_points*dim*sizeof(float), cudaMemcpyHostToDevice);
    std::vector<float> trainSOM(const std::vector<float>& flatPoints, int n_points, int dim, int k, int epochs){
#ifdef __CUDACC__
        if (dim != somDim || k != somK || !somCentroids){ initSOM(k, dim); }
        float* d_points = nullptr; cudaMalloc((void**)&d_points, n_points * dim * sizeof(float));
        cudaMemcpy(d_points, flatPoints.data(), n_points*dim*sizeof(float), cudaMemcpyHostToDevice);

        if (somAssignmentsCapacity < (size_t)n_points){
            if (somAssignments) cudaFree(somAssignments);
            cudaMalloc((void**)&somAssignments, n_points * sizeof(int));
            somAssignmentsCapacity = n_points;
        }

        float* d_accum = nullptr; int* d_counts = nullptr;
        cudaMalloc((void**)&d_accum, k*dim*sizeof(float));
        cudaMalloc((void**)&d_counts, k*sizeof(int));
        int block = 256; int gridPoints = (n_points + block -1)/block; int gridC = (k + block -1)/block;
        for (int e=0;e<epochs;e++){
            cudaMemset(d_accum, 0, k*dim*sizeof(float));
            cudaMemset(d_counts, 0, k*sizeof(int));
            som_assign_kernel<<<gridPoints, block>>>(d_points, somCentroids, somAssignments, n_points, k, dim);
            som_accumulate_kernel<<<gridPoints, block>>>(d_points, somAssignments, d_accum, d_counts, n_points, k, dim);
            som_finalize_kernel<<<gridC, block>>>(somCentroids, d_accum, d_counts, k, dim, 0.5f);
            cudaDeviceSynchronize();
        }
        std::vector<float> host(k*dim);
        cudaMemcpy(host.data(), somCentroids, k*dim*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_points); cudaFree(d_accum); cudaFree(d_counts);
        return host;
#else
        (void)flatPoints; (void)n_points; (void)dim; (void)k; (void)epochs;
        return {};
#endif
    }

        // Read entire stdin
        std::string input;
        std::string line;
        while (std::getline(std::cin, line)) {
            input += line;
        }

        if (input.empty()) {
            std::cerr << "No input received" << std::endl;
            return 1;
        }

        // Parse JSON request
        std::string jobId = JsonParser::parseString(input, "jobId");
        std::string type = JsonParser::parseString(input, "type");
        std::vector<float> data = JsonParser::parseFloatArray(input, "data");

        if (jobId.empty()) jobId = "unknown";
        if (type.empty()) type = "embedding";
        if (data.empty()) data = {1.0f, 2.0f, 3.0f, 4.0f}; // default test data

        std::cerr << "Processing job " << jobId << " type " << type << " with " << data.size() << " elements" << std::endl;

        // Process based on type
        std::vector<float> result;
    if (type == "embedding") {
            result = worker.processEmbedding(data);
        } else if (type == "similarity") {
            // For similarity, expect data to be [vec1..., separator, vec2...]
            // Simple split at midpoint for demo
            size_t mid = data.size() / 2;
            std::vector<float> vec1(data.begin(), data.begin() + mid);
            std::vector<float> vec2(data.begin() + mid, data.end());
            result = worker.processSimilarity(vec1, vec2);
        } else if (type == "autoindex") {
            result = worker.processAutoIndex(data);
        } else if (type == "som_train") {
            if (data.size() < 3) {
                result = { -1.f };
            } else {
                int k = (int)data[0];
                int dim = (int)data[1];
                std::vector<float> points(data.begin()+2, data.end());
                int n_points = (int)(points.size()/dim);
                result = worker.trainSOM(points, n_points, dim, k, 5);
            }
        } else {
            result = worker.processEmbedding(data); // fallback
        }

        // Output JSON response
        std::string response = JsonParser::createResponse(jobId, result, type);
        std::cout << response << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << "{\"jobId\":\"error\",\"error\":\"" << e.what() << "\",\"status\":\"failed\"}" << std::endl;
        return 1;
    }
}