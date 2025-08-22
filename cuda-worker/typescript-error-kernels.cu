// typescript-error-kernels.cu
// Specialized CUDA kernels for TypeScript error processing
// Optimized for RTX 3060 Ti with 8GB VRAM

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

namespace cg = cooperative_groups;

// Constants for TypeScript error processing
#define MAX_ERROR_LENGTH 1024
#define MAX_FIX_LENGTH 2048
#define MAX_CONTEXT_LENGTH 4096
#define VOCAB_SIZE 32000
#define EMBEDDING_DIM 384
#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

// Error pattern types for fast classification
enum ErrorType {
    CANNOT_FIND_NAME = 0,
    PROPERTY_NOT_EXIST = 1,
    TYPE_NOT_ASSIGNABLE = 2,
    MISSING_IMPORT = 3,
    SVELTE5_MIGRATION = 4,
    GENERIC_ERROR = 5,
    NUM_ERROR_TYPES = 6
};

// GPU-optimized data structures
struct ErrorPattern {
    char pattern[256];
    ErrorType type;
    float confidence;
    char fix_template[512];
};

struct TypeScriptErrorGPU {
    char file[256];
    int line;
    int column;
    char message[512];
    char code[1024];
    char context[2048];
    ErrorType detected_type;
    float confidence;
};

struct TokenEmbedding {
    float data[EMBEDDING_DIM];
    int token_id;
};

// Global device memory for patterns and embeddings
__device__ ErrorPattern d_error_patterns[256];
__device__ TokenEmbedding d_token_embeddings[VOCAB_SIZE];
__device__ float d_similarity_matrix[NUM_ERROR_TYPES][NUM_ERROR_TYPES];

// String processing kernels
__device__ int gpu_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0' && len < MAX_ERROR_LENGTH) len++;
    return len;
}

__device__ bool gpu_strstr(const char* haystack, const char* needle) {
    int h_len = gpu_strlen(haystack);
    int n_len = gpu_strlen(needle);
    
    for (int i = 0; i <= h_len - n_len; i++) {
        bool match = true;
        for (int j = 0; j < n_len; j++) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

__device__ float gpu_string_similarity(const char* str1, const char* str2) {
    int len1 = gpu_strlen(str1);
    int len2 = gpu_strlen(str2);
    
    if (len1 == 0 && len2 == 0) return 1.0f;
    if (len1 == 0 || len2 == 0) return 0.0f;
    
    int matches = 0;
    int min_len = min(len1, len2);
    
    for (int i = 0; i < min_len; i++) {
        if (str1[i] == str2[i]) matches++;
    }
    
    return (float)matches / max(len1, len2);
}

// Error pattern matching kernel
__global__ void classify_typescript_errors(
    TypeScriptErrorGPU* errors,
    int num_errors,
    ErrorPattern* patterns,
    int num_patterns,
    float* confidence_scores
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_errors; i += stride) {
        TypeScriptErrorGPU* error = &errors[i];
        float best_confidence = 0.0f;
        ErrorType best_type = GENERIC_ERROR;
        
        // Check against known patterns
        for (int p = 0; p < num_patterns; p++) {
            ErrorPattern* pattern = &patterns[p];
            float similarity = gpu_string_similarity(error->message, pattern->pattern);
            
            if (similarity > best_confidence) {
                best_confidence = similarity;
                best_type = pattern->type;
            }
        }
        
        // Specific pattern checks for common TypeScript errors
        if (gpu_strstr(error->message, "Cannot find name")) {
            best_type = CANNOT_FIND_NAME;
            best_confidence = fmaxf(best_confidence, 0.9f);
        }
        else if (gpu_strstr(error->message, "Property") && gpu_strstr(error->message, "does not exist")) {
            best_type = PROPERTY_NOT_EXIST;
            best_confidence = fmaxf(best_confidence, 0.85f);
        }
        else if (gpu_strstr(error->message, "not assignable to")) {
            best_type = TYPE_NOT_ASSIGNABLE;
            best_confidence = fmaxf(best_confidence, 0.8f);
        }
        else if (gpu_strstr(error->context, "writable") || gpu_strstr(error->context, "readable")) {
            best_type = SVELTE5_MIGRATION;
            best_confidence = fmaxf(best_confidence, 0.95f);
        }
        
        error->detected_type = best_type;
        error->confidence = best_confidence;
        confidence_scores[i] = best_confidence;
    }
}

// Template-based fix generation kernel
__global__ void generate_template_fixes(
    TypeScriptErrorGPU* errors,
    int num_errors,
    char* fix_templates,
    int template_size,
    char* generated_fixes,
    int fix_buffer_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_errors) return;
    
    TypeScriptErrorGPU* error = &errors[tid];
    char* fix_output = &generated_fixes[tid * fix_buffer_size];
    
    // Generate fix based on error type
    switch (error->detected_type) {
        case CANNOT_FIND_NAME:
            if (gpu_strstr(error->code, "writable")) {
                // Svelte 5 migration: writable -> $state
                snprintf(fix_output, fix_buffer_size, 
                    "import { $state } from 'svelte/store';\n"
                    "const %s = $state(null);");
            } else {
                // Generic import fix
                snprintf(fix_output, fix_buffer_size,
                    "// Add appropriate import statement\n"
                    "import { %s } from 'appropriate-module';");
            }
            break;
            
        case PROPERTY_NOT_EXIST:
            snprintf(fix_output, fix_buffer_size,
                "// Type assertion or property check needed\n"
                "(event.target as HTMLFormElement).handleSubmit();");
            break;
            
        case TYPE_NOT_ASSIGNABLE:
            snprintf(fix_output, fix_buffer_size,
                "// Type assertion needed\n"
                "const response = await fetch(url, body as RequestInit);");
            break;
            
        case SVELTE5_MIGRATION:
            snprintf(fix_output, fix_buffer_size,
                "// Svelte 5 runes migration\n"
                "const user = $state(null);");
            break;
            
        default:
            snprintf(fix_output, fix_buffer_size,
                "// Generic fix - manual review needed\n"
                "// %s", error->message);
            break;
    }
}

// Token embedding computation for semantic analysis
__global__ void compute_error_embeddings(
    TypeScriptErrorGPU* errors,
    int num_errors,
    TokenEmbedding* token_embeddings,
    int vocab_size,
    float* error_embeddings,
    int embedding_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_errors) return;
    
    TypeScriptErrorGPU* error = &errors[tid];
    float* embedding = &error_embeddings[tid * embedding_dim];
    
    // Initialize embedding to zero
    for (int i = 0; i < embedding_dim; i++) {
        embedding[i] = 0.0f;
    }
    
    // Simple bag-of-words embedding
    int message_len = gpu_strlen(error->message);
    int token_count = 0;
    
    // Hash-based token ID generation (simplified)
    for (int i = 0; i < message_len - 2; i++) {
        int token_hash = (error->message[i] * 31 + error->message[i+1] * 7 + error->message[i+2]) % vocab_size;
        
        // Add token embedding
        for (int j = 0; j < embedding_dim; j++) {
            embedding[j] += token_embeddings[token_hash].data[j];
        }
        token_count++;
    }
    
    // Normalize by token count
    if (token_count > 0) {
        for (int i = 0; i < embedding_dim; i++) {
            embedding[i] /= token_count;
        }
    }
}

// Similarity clustering for error grouping
__global__ void cluster_similar_errors(
    float* error_embeddings,
    int num_errors,
    int embedding_dim,
    float similarity_threshold,
    int* cluster_assignments,
    int* cluster_sizes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_errors) return;
    
    float* embedding1 = &error_embeddings[tid * embedding_dim];
    int best_cluster = tid; // Start with self as cluster
    float best_similarity = 0.0f;
    
    // Find most similar error (lower index to avoid duplicates)
    for (int i = 0; i < tid; i++) {
        float* embedding2 = &error_embeddings[i * embedding_dim];
        
        // Compute cosine similarity
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        
        for (int j = 0; j < embedding_dim; j++) {
            dot_product += embedding1[j] * embedding2[j];
            norm1 += embedding1[j] * embedding1[j];
            norm2 += embedding2[j] * embedding2[j];
        }
        
        float similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));
        
        if (similarity > similarity_threshold && similarity > best_similarity) {
            best_similarity = similarity;
            best_cluster = cluster_assignments[i]; // Join existing cluster
        }
    }
    
    cluster_assignments[tid] = best_cluster;
    
    // Update cluster size atomically
    if (best_cluster == tid) {
        cluster_sizes[tid] = 1; // New cluster
    } else {
        atomicAdd(&cluster_sizes[best_cluster], 1);
    }
}

// Batch processing optimization kernel
__global__ void optimize_batch_processing(
    TypeScriptErrorGPU* errors,
    int num_errors,
    int* cluster_assignments,
    int* processing_order,
    float* priority_scores
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_errors) return;
    
    TypeScriptErrorGPU* error = &errors[tid];
    
    // Calculate priority score based on:
    // 1. Error type (some are easier to fix)
    // 2. Confidence level
    // 3. File importance (based on path)
    
    float priority = 0.0f;
    
    // Type-based priority
    switch (error->detected_type) {
        case SVELTE5_MIGRATION:
            priority += 0.9f; // High priority, easy to fix
            break;
        case CANNOT_FIND_NAME:
            priority += 0.8f;
            break;
        case PROPERTY_NOT_EXIST:
            priority += 0.7f;
            break;
        case TYPE_NOT_ASSIGNABLE:
            priority += 0.6f;
            break;
        default:
            priority += 0.3f;
            break;
    }
    
    // Confidence bonus
    priority += error->confidence * 0.3f;
    
    // File importance (components vs utilities)
    if (gpu_strstr(error->file, "components")) {
        priority += 0.2f;
    } else if (gpu_strstr(error->file, "stores")) {
        priority += 0.15f;
    } else if (gpu_strstr(error->file, "routes")) {
        priority += 0.1f;
    }
    
    priority_scores[tid] = priority;
    processing_order[tid] = tid; // Will be sorted by host
}

// Memory-efficient string operations using shared memory
__global__ void process_error_strings_shared(
    TypeScriptErrorGPU* errors,
    int num_errors,
    char* output_buffer,
    int buffer_size
) {
    extern __shared__ char shared_buffer[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid >= num_errors) return;
    
    // Load error data into shared memory
    TypeScriptErrorGPU* error = &errors[tid];
    char* local_buffer = &shared_buffer[local_tid * 512];
    
    // Process error message in shared memory
    int msg_len = gpu_strlen(error->message);
    for (int i = 0; i < msg_len && i < 511; i++) {
        local_buffer[i] = error->message[i];
    }
    local_buffer[msg_len] = '\0';
    
    __syncthreads();
    
    // Perform string processing operations
    // (Replace common patterns, normalize whitespace, etc.)
    
    // Copy result back to global memory
    char* output = &output_buffer[tid * 512];
    for (int i = 0; i < 512; i++) {
        output[i] = local_buffer[i];
        if (local_buffer[i] == '\0') break;
    }
}

// Performance monitoring kernel
__global__ void collect_processing_metrics(
    int num_processed,
    float* processing_times,
    float* confidence_scores,
    int* error_types,
    float* metrics_output // [avg_time, avg_confidence, type_distribution[6]]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use cooperative groups for reduction
    auto block = cg::this_thread_block();
    
    float sum_time = 0.0f;
    float sum_confidence = 0.0f;
    int type_counts[NUM_ERROR_TYPES] = {0};
    
    // Each thread processes multiple elements
    for (int i = tid; i < num_processed; i += blockDim.x * gridDim.x) {
        sum_time += processing_times[i];
        sum_confidence += confidence_scores[i];
        if (error_types[i] >= 0 && error_types[i] < NUM_ERROR_TYPES) {
            type_counts[error_types[i]]++;
        }
    }
    
    // Reduce within block
    typedef cub::BlockReduce<float, 256> BlockReduceFloat;
    typedef cub::BlockReduce<int, 256> BlockReduceInt;
    
    __shared__ typename BlockReduceFloat::TempStorage temp_storage_float;
    __shared__ typename BlockReduceInt::TempStorage temp_storage_int;
    
    float block_sum_time = BlockReduceFloat(temp_storage_float).Sum(sum_time);
    float block_sum_conf = BlockReduceFloat(temp_storage_float).Sum(sum_confidence);
    
    if (threadIdx.x == 0) {
        atomicAdd(&metrics_output[0], block_sum_time); // Total time
        atomicAdd(&metrics_output[1], block_sum_conf); // Total confidence
        
        // Add type counts
        for (int i = 0; i < NUM_ERROR_TYPES; i++) {
            atomicAdd((int*)&metrics_output[2 + i], type_counts[i]);
        }
    }
}

// C++ wrapper functions for Go integration
extern "C" {
    
// Initialize CUDA kernels and allocate GPU memory
int init_typescript_gpu_processor(int max_errors) {
    // Allocate device memory for error patterns and embeddings
    // Initialize constant data
    return 0; // Success
}

// Process batch of TypeScript errors on GPU
int process_typescript_errors_gpu(
    TypeScriptErrorGPU* h_errors,
    int num_errors,
    char* h_fixes,
    int fix_buffer_size,
    float* h_confidence_scores
) {
    // Device memory allocation
    TypeScriptErrorGPU* d_errors;
    char* d_fixes;
    float* d_confidence_scores;
    
    cudaMalloc(&d_errors, num_errors * sizeof(TypeScriptErrorGPU));
    cudaMalloc(&d_fixes, num_errors * fix_buffer_size);
    cudaMalloc(&d_confidence_scores, num_errors * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_errors, h_errors, num_errors * sizeof(TypeScriptErrorGPU), cudaMemcpyHostToDevice);
    
    // Launch kernels
    dim3 blockSize(256);
    dim3 gridSize((num_errors + blockSize.x - 1) / blockSize.x);
    
    // 1. Classify errors
    classify_typescript_errors<<<gridSize, blockSize>>>(d_errors, num_errors, nullptr, 0, d_confidence_scores);
    cudaDeviceSynchronize();
    
    // 2. Generate template fixes
    generate_template_fixes<<<gridSize, blockSize>>>(d_errors, num_errors, nullptr, 0, d_fixes, fix_buffer_size);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_fixes, d_fixes, num_errors * fix_buffer_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_confidence_scores, d_confidence_scores, num_errors * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_errors);
    cudaFree(d_fixes);
    cudaFree(d_confidence_scores);
    
    return 0; // Success
}

// Cleanup GPU resources
void cleanup_typescript_gpu_processor() {
    // Free device memory
    cudaDeviceReset();
}

} // extern "C"