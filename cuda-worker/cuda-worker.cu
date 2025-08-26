// cuda-worker.cu
// Minimal CUDA worker that reads JSON from stdin, executes GPU kernels, outputs
// JSON to stdout
//
// Compile options:
// 1. NVCC (recommended): nvcc -std=c++14 cuda-worker.cu -o cuda-worker.exe
// 2. Clang (alternative): Use build-clang.bat script for proper setup
// 3. Manual Clang: clang++ -std=c++14 --cuda-gpu-arch=sm_75
// --cuda-path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
// -I"CUDA_PATH\include" -L"CUDA_PATH\lib\x64" -lcudart cuda-worker.cu -o
// cuda-worker.exe

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__clang__)
// Clang/LLVM with CUDA: include driver and runtime API headers
#include <cuda.h>
#include <cuda_runtime_api.h>
#else
// NVCC and standard CUDA runtime
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <curand_kernel.h>
#include <future>
#include <mutex>
#include <thread>

// Simple JSON parser/writer (avoiding external dependencies for minimal setup)
struct JsonParser {
  static std::vector<float> parseFloatArray(const std::string &json,
                                            const std::string &key) {
    std::vector<float> result;
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos)
      return result;

    size_t arrayStart = json.find("[", keyPos);
    size_t arrayEnd = json.find("]", arrayStart);
    if (arrayStart == std::string::npos || arrayEnd == std::string::npos)
      return result;

    std::string arrayStr =
        json.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
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

  static std::string parseString(const std::string &json,
                                 const std::string &key) {
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos)
      return "";

    size_t colonPos = json.find(":", keyPos);
    size_t quoteStart = json.find("\"", colonPos);
    size_t quoteEnd = json.find("\"", quoteStart + 1);

    if (quoteStart == std::string::npos || quoteEnd == std::string::npos)
      return "";
    return json.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
  }

  static std::string createResponse(const std::string &jobId,
                                    const std::vector<float> &vector,
                                    const std::string &type = "embedding") {
    std::stringstream ss;
    ss << "{\"jobId\":\"" << jobId << "\",\"type\":\"" << type
       << "\",\"vector\":[";
    for (size_t i = 0; i < vector.size(); ++i) {
      if (i > 0)
        ss << ",";
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

__global__ void vector_similarity_kernel(const float *vec1, const float *vec2,
                                         float *result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Compute element-wise similarity (cosine-like)
    result[idx] = vec1[idx] * vec2[idx];
  }
}

__global__ void som_cluster_kernel(const float *input, float *centroids,
                                   int *assignments, int n_points,
                                   int n_centroids, int dim) {
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

__global__ void som_update_kernel(float *centroids, const float *input,
                                  const int *assignments, int n_points,
                                  int n_centroids, int dim) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n_centroids) {
    for (int p = 0; p < n_points; ++p) {
      if (assignments[p] == c) {
        for (int d = 0; d < dim; ++d) {
          float *centroidVal = &centroids[c * dim + d];
          float current = *centroidVal;
          float target = input[p * dim + d];
          *centroidVal = current + 0.05f * (target - current);
        }
      }
    }
  }
}

__global__ void som_assign_kernel(const float* input, const float* centroids, int* assignments, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        const float* p = input + idx * dim;
        float best = 1e30f;
        int bestC = 0;
        for (int c = 0; c < k; c++) {
          const float *cent = centroids + c * dim;
          float dist = 0.f;
          for (int d = 0; d < dim; d++) {
            float diff = p[d] - cent[d];
            dist += diff * diff;
          }
          if (dist < best) {
            best = dist;
            bestC = c;
          }
        }
        assignments[idx] = bestC;
    }
}

__global__ void som_accumulate_kernel(const float* input, const int* assignments, float* accum, int* counts, int n_points, int k, int dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points){
        int c = assignments[idx];
        if (c >= 0 && c < k) {
          atomicAdd(&counts[c], 1);
          const float *p = input + idx * dim;
          for (int d = 0; d < dim; d++) {
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
          for (int d = 0; d < dim; d++) {
            float target = accum[c * dim + d] / count;
            float *cent = &centroids[c * dim + d];
            *cent = *cent + lr * (target - *cent);
          }
        }
    }
}
class CudaWorker {
private:
    cudaDeviceProp deviceProp;
    float *persistentInput = nullptr;
    float *persistentOutput = nullptr;
    size_t persistentCapacity = 0; // number of floats
    // SOM buffers
    float *somCentroids = nullptr;
    int *somAssignments = nullptr;
    size_t somAssignmentsCapacity = 0;
    int somDim = 0;
    int somK = 0;

    void ensureCapacity(size_t n) {
      if (n <= persistentCapacity)
        return;
      if (persistentInput)
        cudaFree(persistentInput);
      if (persistentOutput)
        cudaFree(persistentOutput);
      cudaMalloc((void **)&persistentInput, n * sizeof(float));
      cudaMalloc((void **)&persistentOutput, n * sizeof(float));
      persistentCapacity = n;
    }
    void initSOM(int k, int dim) {
      if (somCentroids)
        cudaFree(somCentroids);
      if (somAssignments)
        cudaFree(somAssignments);
      cudaMalloc((void **)&somCentroids, k * dim * sizeof(float));
      somK = k;
      somDim = dim;
      somAssignmentsCapacity = 0;
      // Initialize centroids with small random host values
      std::vector<float> host(k * dim);
      for (int i = 0; i < k * dim; i++)
        host[i] = (float)(rand() % 100) / 100.f;
      cudaMemcpy(somCentroids, host.data(), k * dim * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

public:
  CudaWorker() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
      throw std::runtime_error("No CUDA devices available");
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cerr << "CUDA Worker initialized: " << deviceProp.name
              << " globalMemMB=" << deviceProp.totalGlobalMem / (1024 * 1024)
              << std::endl;
  }

  ~CudaWorker() {
    if (persistentInput)
      cudaFree(persistentInput);
    if (persistentOutput)
      cudaFree(persistentOutput);
    if (somCentroids)
      cudaFree(somCentroids);
    if (somAssignments)
      cudaFree(somAssignments);
  }

  std::vector<float> processEmbedding(const std::vector<float> &input) {
    int n = (int)input.size();
    if (!n)
      return {};
    ensureCapacity(n);
    cudaMemcpy(persistentInput, input.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    int blockSize = std::min(256, n);
    int gridSize = (n + blockSize - 1) / blockSize;
    simple_embedding_kernel<<<gridSize, blockSize>>>(
        persistentInput, persistentOutput, n, 1.2345f);
    cudaDeviceSynchronize();
    std::vector<float> out(n);
    cudaMemcpy(out.data(), persistentOutput, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    return out;
  }

    std::vector<float> processSimilarity(const std::vector<float> &a,
                                         const std::vector<float> &b) {
      if (a.size() != b.size() || a.empty())
        return {};
      int n = (int)a.size();
      ensureCapacity(n * 2);
      float *d_vec1 = persistentInput;
      float *d_vec2 = persistentOutput;
      float *d_res = nullptr;
      cudaMemcpy(d_vec1, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_vec2, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_res, n * sizeof(float));
      int blockSize = std::min(256, n);
      int gridSize = (n + blockSize - 1) / blockSize;
      vector_similarity_kernel<<<gridSize, blockSize>>>(d_vec1, d_vec2, d_res,
                                                        n);
      cudaDeviceSynchronize();
      std::vector<float> out(n);
      cudaMemcpy(out.data(), d_res, n * sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(d_res);
      return out;
    }

    std::vector<float> processAutoIndex(const std::vector<float> &input) {
      auto processed = processEmbedding(input);
      processed.push_back((float)time(nullptr));
      processed.push_back((float)input.size());
      processed.push_back(1.0f);
      return processed;
    }
    std::vector<float> trainSOM(const std::vector<float> &flatPoints,
                                int n_points, int dim, int k, int epochs) {
      if (dim != somDim || k != somK || !somCentroids) {
        initSOM(k, dim);
      }
      float *d_points = nullptr;
      cudaMalloc((void **)&d_points, n_points * dim * sizeof(float));
      cudaMemcpy(d_points, flatPoints.data(), n_points * dim * sizeof(float),
                 cudaMemcpyHostToDevice);

      if (somAssignmentsCapacity < (size_t)n_points) {
        if (somAssignments)
          cudaFree(somAssignments);
        cudaMalloc((void **)&somAssignments, n_points * sizeof(int));
        somAssignmentsCapacity = n_points;
      }

      float *d_accum = nullptr;
      int *d_counts = nullptr;
      cudaMalloc((void **)&d_accum, k * dim * sizeof(float));
      cudaMalloc((void **)&d_counts, k * sizeof(int));
      int block = 256;
      int gridPoints = (n_points + block - 1) / block;
      int gridC = (k + block - 1) / block;
      for (int e = 0; e < epochs; e++) {
        cudaMemset(d_accum, 0, k * dim * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));
        som_assign_kernel<<<gridPoints, block>>>(
            d_points, somCentroids, somAssignments, n_points, k, dim);
        som_accumulate_kernel<<<gridPoints, block>>>(
            d_points, somAssignments, d_accum, d_counts, n_points, k, dim);
        som_finalize_kernel<<<gridC, block>>>(somCentroids, d_accum, d_counts,
                                              k, dim, 0.5f);
        cudaDeviceSynchronize();
      }
      std::vector<float> host(k * dim);
      cudaMemcpy(host.data(), somCentroids, k * dim * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaFree(d_points);
      cudaFree(d_accum);
      cudaFree(d_counts);
      return host;
    }
}; // end class CudaWorker

int main() {
  try {
    CudaWorker worker;
    // Read entire stdin
    std::string input;
    std::string line;
    while (std::getline(std::cin, line))
      input += line;
    if (input.empty()) {
      std::cerr << "No input received" << std::endl;
      return 1;
    }

    std::string jobId = JsonParser::parseString(input, "jobId");
    std::string type = JsonParser::parseString(input, "type");
    std::vector<float> data = JsonParser::parseFloatArray(input, "data");
    if (jobId.empty())
      jobId = "unknown";
    if (type.empty())
      type = "embedding";
    if (data.empty())
      data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::cerr << "Processing job " << jobId << " type=" << type
              << " elements=" << data.size() << std::endl;

    std::vector<float> result;
    if (type == "embedding")
      result = worker.processEmbedding(data);
    else if (type == "similarity") {
      size_t mid = data.size() / 2;
      std::vector<float> a(data.begin(), data.begin() + mid);
      std::vector<float> b(data.begin() + mid, data.end());
      result = worker.processSimilarity(a, b);
    } else if (type == "autoindex")
      result = worker.processAutoIndex(data);
    else if (type == "som_train") {
      if (data.size() < 3) {
        result = {-1.f};
      } else {
        int k = (int)data[0];
        int dim = (int)data[1];
        std::vector<float> points(data.begin() + 2, data.end());
        int n_points = (int)(points.size() / dim);
        result = worker.trainSOM(points, n_points, dim, k, 5);
      }
    } else
      result = worker.processEmbedding(data);

    {
      // Quantize results, compute simple stats and include GPU info for the
      // Go service. This builds a compact JSON payload (includes
      // dimensions, sum, mean, nonzeros, gpu name and memMB) and writes it
      // atomically to stdout from a background thread, then exits
      // immediately to avoid duplicate output below.
      double sum = 0.0;
      size_t nonzero = 0;
      const float eps = 1e-9f;
      const float quant = 1e6f; // quantize to 6 decimal places

      for (size_t i = 0; i < result.size(); ++i) {
        float v = result[i];
        if (!std::isfinite(v))
          v = 0.0f;
        float q = std::round(v * quant) / quant;
        result[i] = q;
        sum += static_cast<double>(q);
        if (std::fabs(q) > eps)
          ++nonzero;
      }

      double mean =
          result.empty() ? 0.0 : sum / static_cast<double>(result.size());

      // Query device info (best-effort; non-fatal)
      cudaDeviceProp localProp;
      cudaError_t cerr = cudaGetDeviceProperties(&localProp, 0);
      const char *gpuName = (cerr == cudaSuccess) ? localProp.name : "unknown";
      long long memMB =
          (cerr == cudaSuccess)
              ? (long long)(localProp.totalGlobalMem / (1024LL * 1024LL))
              : 0LL;

      // Build JSON response efficiently
      std::string response;
      response.reserve(512 + result.size() * 8);

      response += "{\"jobId\":\"";
      response += jobId;
      response += "\",\"type\":\"";
      response += type;
      response += "\",\"vector\":[";

      char numbuf[64];
      for (size_t i = 0; i < result.size(); ++i) {
        if (i)
          response.push_back(',');
        // Use %g for compact float formatting
        int n = snprintf(numbuf, sizeof(numbuf), "%.6g",
                         static_cast<double>(result[i]));
        response.append(numbuf, n);
      }

      response += "],\"status\":\"success\",\"timestamp\":";
      response += std::to_string(static_cast<long long>(time(nullptr)));
      response += ",\"dimensions\":";
      response += std::to_string(result.size());
      response += ",\"sum\":";
      response += std::to_string(sum);
      response += ",\"mean\":";
      response += std::to_string(mean);
      response += ",\"nonzeros\":";
      response += std::to_string(nonzero);
      response += ",\"gpu\":\"";
      response += gpuName;
      response += "\",\"memMB\":";
      response += std::to_string(memMB);
      response += "}";

      // Thread-safe write to stdout (offload I/O so main GPU thread isn't
      // blocked)
      static std::mutex stdout_mutex;
      auto writer = [&response]() {
        std::lock_guard<std::mutex> lk(stdout_mutex);
        std::cout << response << std::endl;
      };

      auto fut = std::async(std::launch::async, writer);
      fut.get();

      // Exit early to avoid the duplicate responder below
      return 0;
    }
    // Build JSON response efficiently into a single string (reduces
    // concurrent I/O)
    std::string response;
    response.reserve(256 +
                     result.size() * 8); // rough reserve to avoid reallocations

    response += "{\"jobId\":\"";
    response += jobId;
    response += "\",\"type\":\"";
    response += type;
    response += "\",\"vector\":[";

    for (size_t i = 0; i < result.size(); ++i) {
      if (i)
        response.push_back(',');
      response += std::to_string(result[i]);
    }

    response += "],\"status\":\"success\",\"timestamp\":";
    response += std::to_string(static_cast<long long>(time(nullptr)));
    response += "}";

    // Thread-safe write to stdout so multiple workers can output
    // concurrently
    static std::mutex stdout_mutex;
    auto writer = [&response]() {
      std::lock_guard<std::mutex> lk(stdout_mutex);
      std::cout << response << std::endl;
    };

    // Offload actual I/O to a background thread to avoid blocking GPU-bound
    // main thread. .get() waits for completion so the program remains
    // deterministic for single-job runs.
    auto fut = std::async(std::launch::async, writer);
    fut.get();

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    std::cout << "{\"jobId\":\"error\",\"error\":\"" << e.what()
              << "\",\"status\":\"failed\"}" << std::endl;
    return 1;
  }
}