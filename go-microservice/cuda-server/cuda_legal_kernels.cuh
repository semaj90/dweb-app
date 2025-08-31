// cuda_legal_kernels.cuh - CUDA kernels for legal AI processing
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <device_launch_parameters.h>

// CUDA kernel declarations for legal AI processing

namespace legal_cuda {

// Text embedding kernels
__global__ void tokenizeText(const char* input_text, int* tokens, int max_tokens);

__global__ void embedTokens(const int* tokens, int token_count, 
                           const float* embedding_table, int embedding_dim,
                           float* output_embeddings);

__global__ void poolEmbeddings(const float* token_embeddings, int token_count,
                              int embedding_dim, float* pooled_embedding);

// Similarity search kernels
__global__ void computeCosineSimilarity(const float* query_embedding,
                                       const float* document_embeddings,
                                       int num_documents, int embedding_dim,
                                       float* similarities);

__global__ void findTopK(const float* similarities, const int* document_ids,
                        int num_documents, int k,
                        float* top_similarities, int* top_document_ids);

// Clustering kernels
__global__ void kmeansAssignClusters(const float* embeddings, int num_embeddings,
                                    const float* centroids, int num_clusters,
                                    int embedding_dim, int* assignments);

__global__ void kmeansUpdateCentroids(const float* embeddings, int num_embeddings,
                                     const int* assignments, int num_clusters,
                                     int embedding_dim, float* new_centroids,
                                     int* cluster_sizes);

// Legal entity extraction kernels
__global__ void extractLegalEntities(const int* tokens, int token_count,
                                    const float* ner_weights, int* entity_labels);

// Document similarity kernels
__global__ void computeJaccardSimilarity(const int* doc1_tokens, int doc1_count,
                                        const int* doc2_tokens, int doc2_count,
                                        float* similarity);

// Memory-efficient batch processing
template<typename T>
__global__ void batchProcess(T* input_data, T* output_data, 
                           int batch_size, int data_dim);

// Tensor operation kernels (for transformer models)
__global__ void multiHeadAttention(const float* query, const float* key, const float* value,
                                  int seq_len, int hidden_dim, int num_heads,
                                  float* attention_output);

__global__ void layerNorm(const float* input, const float* gamma, const float* beta,
                         int batch_size, int hidden_dim, float* output);

__global__ void gelu(const float* input, int size, float* output);

// Legal-specific processing kernels
__global__ void detectContractClauses(const int* tokens, int token_count,
                                     const float* clause_patterns,
                                     int* clause_positions, float* confidences);

__global__ void analyzeLegalSentiment(const float* embeddings, int embedding_count,
                                     const float* sentiment_weights,
                                     float* sentiment_scores);

// Host functions for kernel launches
cudaError_t launchTokenization(const char* text, int* tokens, int max_tokens,
                              cudaStream_t stream = 0);

cudaError_t launchEmbeddingGeneration(const int* tokens, int token_count,
                                    const float* embedding_table,
                                    int embedding_dim, float* output,
                                    cudaStream_t stream = 0);

cudaError_t launchSimilaritySearch(const float* query_embedding,
                                 const float* document_embeddings,
                                 int num_documents, int embedding_dim,
                                 int k, float* top_similarities,
                                 int* top_document_ids,
                                 cudaStream_t stream = 0);

cudaError_t launchKMeansClustering(const float* embeddings, int num_embeddings,
                                 int embedding_dim, int num_clusters,
                                 int max_iterations, float* centroids,
                                 int* assignments,
                                 cudaStream_t stream = 0);

// Memory management utilities
class CudaMemoryManager {
public:
    static void* allocateGPU(size_t size);
    static void deallocateGPU(void* ptr);
    static cudaError_t copyHostToDevice(void* device_ptr, const void* host_ptr, size_t size);
    static cudaError_t copyDeviceToHost(void* host_ptr, const void* device_ptr, size_t size);
    static cudaError_t copyDeviceToDeviceAsync(void* dst, const void* src, size_t size, cudaStream_t stream);
};

// Performance profiling utilities
class CudaProfiler {
public:
    static void startTimer(cudaEvent_t& start_event, cudaStream_t stream = 0);
    static float stopTimer(cudaEvent_t& start_event, cudaEvent_t& stop_event, cudaStream_t stream = 0);
    static void printKernelInfo(const char* kernel_name, float elapsed_ms, size_t memory_bytes);
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
constexpr int MAX_TOKENS_PER_DOCUMENT = 2048;
constexpr int EMBEDDING_DIM = 768;
constexpr int MAX_BATCH_SIZE = 32;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

} // namespace legal_cuda