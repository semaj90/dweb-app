// cuda_legal_kernels.cuh - CUDA kernels & host helpers for legal AI processing
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>


#include <cstdio>
#include <cstdlib>

namespace legal_cuda {

// Constants
constexpr int MAX_TOKENS_PER_DOCUMENT = 2048;
constexpr int EMBEDDING_DIM = 768;
constexpr int MAX_BATCH_SIZE = 32;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Error checking macros
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =========================
// Kernel declarations
// =========================

// Text embedding kernels
__global__ void tokenizeText(const char* input_text, int* tokens, int max_tokens);

// Token->embedding (FP32)
__global__ void embedTokens(const int *tokens, int token_count,
                            const float *embedding_table, int embedding_dim,
                            float *output_embeddings);

// FP16 variant (for Tensor Core / memory reduction)
__global__ void embedTokens_fp16(const int *tokens, int token_count,
                                 const __half *embedding_table,
                                 int embedding_dim, __half *output_embeddings);

// Pooling (e.g., mean pooling)
__global__ void poolEmbeddings(const float* token_embeddings, int token_count,
                              int embedding_dim, float* pooled_embedding);

// Similarity kernels (per-element used by host launcher)
// A simple kernel to compute cosine similarity elementwise (fallback)
__global__ void computeCosineSimilarityKernel(const float *query_embedding,
                                              const float *document_embeddings,
                                              int num_documents,
                                              int embedding_dim,
                                              float *similarities);

// findTopK device kernel (naive); prefer thrust/cub in host launcher
__global__ void findTopKNaive(const float *similarities,
                              const int *document_ids, int num_documents, int k,
                              float *top_similarities, int *top_document_ids);

// Clustering kernels
// Assignment: one thread per embedding (parallel across embeddings)
__global__ void kmeansAssignClusters(const float* embeddings, int num_embeddings,
                                    const float* centroids, int num_clusters,
                                    int embedding_dim, int* assignments);

// Update centroids: each thread reduces contribution for one dimension of one
// centroid We'll provide an implementation that uses atomic adds to avoid race
// conditions.
__global__ void kmeansUpdateCentroidsAtomic(const float *embeddings,
                                            int num_embeddings,
                                            const int *assignments,
                                            int num_clusters, int embedding_dim,
                                            float *centroid_sums,
                                            int *cluster_sizes);

// Legal entity extraction
// Block-level parallelism: one block per token (or tile of tokens) for local
// reductions
__global__ void extractLegalEntities(const int* tokens, int token_count,
                                    const float* ner_weights, int* entity_labels);

// Document similarity (Jaccard) - token sets as sorted int arrays
__global__ void computeJaccardSimilarity(const int* doc1_tokens, int doc1_count,
                                        const int* doc2_tokens, int doc2_count,
                                        float* similarity);

// Memory-efficient batch processing (templated)
template <typename T>
__global__ void batchProcess(T *input_data, T *output_data, int batch_size,
                             int data_dim);

// Transformer ops (note: these are signatures; implementations should use
// tensor cores / fmha libs)
__global__ void multiHeadAttention(const float* query, const float* key, const float* value,
                                  int seq_len, int hidden_dim, int num_heads,
                                  float* attention_output);

__global__ void layerNorm(const float* input, const float* gamma, const float* beta,
                         int batch_size, int hidden_dim, float* output);

__global__ void gelu(const float* input, int size, float* output);

// Legal-specific processing
__global__ void detectContractClauses(const int* tokens, int token_count,
                                     const float* clause_patterns,
                                     int* clause_positions, float* confidences);

__global__ void analyzeLegalSentiment(const float* embeddings, int embedding_count,
                                     const float* sentiment_weights,
                                     float* sentiment_scores);

// =========================
// Host launcher declarations / helpers
// =========================

// Memory management utilities
class CudaMemoryManager {
public:
  static void *allocateGPU(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
  }
  static void deallocateGPU(void *ptr) {
    if (ptr)
      CUDA_CHECK(cudaFree(ptr));
  }
  static cudaError_t copyHostToDevice(void *device_ptr, const void *host_ptr,
                                      size_t size) {
    return cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
  }
  static cudaError_t copyDeviceToHost(void *host_ptr, const void *device_ptr,
                                      size_t size) {
    return cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
  }
  static cudaError_t copyDeviceToDeviceAsync(void *dst, const void *src,
                                             size_t size, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
  }
};

// Performance profiling utilities
class CudaProfiler {
public:
  static void startTimer(cudaEvent_t &start_event, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventRecord(start_event, stream));
  }
  static float stopTimer(cudaEvent_t &start_event, cudaEvent_t &stop_event,
                         cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    return ms;
  }
  static void printKernelInfo(const char *kernel_name, float elapsed_ms,
                              size_t memory_bytes) {
    printf("Kernel %s: %.3f ms, %.3f MB transferred\n", kernel_name, elapsed_ms,
           memory_bytes / (1024.0f * 1024.0f));
  }
};

// =========================
// Host launcher prototypes (improved)
// =========================

// Tokenization host launcher
cudaError_t launchTokenization(const char *text, int *tokens, int max_tokens,
                               cudaStream_t stream = 0);

// Embedding generation host launchers
// Choose fp16 flag to enable __half-based embedding table and outputs (saves
// memory / enables tensor cores)
cudaError_t launchEmbeddingGeneration(const int *tokens, int token_count,
                                      const float *embedding_table,
                                      int embedding_dim, float *output,
                                      cudaStream_t stream = 0);

cudaError_t launchEmbeddingGenerationFP16(const int *tokens, int token_count,
                                          const __half *embedding_table,
                                          int embedding_dim, __half *output,
                                          cudaStream_t stream = 0);

// Similarity search host launcher using cuBLAS for dot-products for higher
// throughput.
// - It will compute cosine similarities using cuBLAS dot (or fallback to
// kernel).
// - It will then select top-k results using thrust::sort_by_key on device
// (efficient). Note: findTopK uses thrust::sort_by_key; for large-scale sorts,
// consider cub::DeviceRadixSort.
cudaError_t launchSimilaritySearch(
    const float *query_embedding,
    const float
        *document_embeddings, // flattened [num_documents * embedding_dim]
    int num_documents, int embedding_dim, int k, float *top_similarities,
    int *top_document_ids, bool use_cublas = true, cudaStream_t stream = 0);

// KMeans host launcher: max_iterations parameter belongs here (host), not the
// kernel.
cudaError_t launchKMeansClustering(const float *embeddings, int num_embeddings,
                                   int embedding_dim, int num_clusters,
                                   int max_iterations, float *centroids,
                                   int *assignments, cudaStream_t stream = 0);

// TopK helper using thrust (device-side sort by key)
inline void deviceTopKSort(float *d_similarities, int *d_doc_ids,
                           int num_documents, int k, float *d_top_similarities,
                           int *d_top_doc_ids, cudaStream_t stream = 0) {
  // Wrap raw pointers with thrust device pointers
  thrust::device_ptr<float> sim_ptr(d_similarities);
  thrust::device_ptr<int> id_ptr(d_doc_ids);

  // sort descending by similarities
  thrust::sort_by_key(thrust::cuda::par.on(stream), sim_ptr,
                      sim_ptr + num_documents, id_ptr,
                      thrust::greater<float>());

  // copy top-k to outputs
  CUDA_CHECK(cudaMemcpyAsync(d_top_similarities, d_similarities,
                             sizeof(float) * k, cudaMemcpyDeviceToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_top_doc_ids, d_doc_ids, sizeof(int) * k,
                             cudaMemcpyDeviceToDevice, stream));
}

// Note: implementations (in .cu/.cuh) should:
// - Use atomicAdd in kmeansUpdateCentroidsAtomic when accumulating centroid
// sums.
// - Use block-level parallelism in extractLegalEntities (one block processes a
// tile of tokens).
// - Use cublasSdot (or Strided Batched GEMV) in launchSimilaritySearch to
// compute dot products efficiently.
// - For TensorCore-enabled paths, provide FP16 kernels / use
// cuBLAS/cuDNN/FMHAv2 where appropriate.
// - Consider using cooperative groups and warp-level reductions for better
// performance on large dims.

// Additional utility: compute cosine similarity on host/device using cuBLAS
// (prototype)
cudaError_t computeCosineSimilaritiesWithCublas(
    const float *d_query, const float *d_docs, int num_documents,
    int embedding_dim, float *d_similarities, cublasHandle_t handle,
    cudaStream_t stream = 0);

// =========================
// End namespace
// =========================
} // namespace legal_cuda