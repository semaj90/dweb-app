// cuda-quaternion-worker.cu
// Production CUDA worker for quaternion point cloud transforms
// Compile: nvcc -O3 -std=c++14 cuda-quaternion-worker.cu -o cuda-quaternion-worker.exe -I./json/include

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <memory>

// Simple JSON parser (minimal implementation for production)
struct JSONValue {
    std::string str_val;
    std::vector<float> array_val;
    std::vector<JSONValue> object_val;
    bool is_string = false;
    bool is_array = false;
    bool is_object = false;
    
    static JSONValue parse_simple(const std::string& json);
    std::string get_string(const std::string& key = "") const;
    std::vector<float> get_array(const std::string& key = "") const;
    JSONValue get_object(const std::string& key) const;
};

struct Quaternion { 
    float w, x, y, z; 
    
    __host__ __device__ Quaternion normalize() const {
        float len = sqrtf(w*w + x*x + y*y + z*z);
        if (len < 1e-7f) return {1.0f, 0.0f, 0.0f, 0.0f};
        return {w/len, x/len, y/len, z/len};
    }
};

// GPU quaternion rotation kernel - highly optimized
__device__ void quaternion_rotate_point(const Quaternion &q, const float* in, float* out, int idx) {
    float vx = in[3*idx+0], vy = in[3*idx+1], vz = in[3*idx+2];
    
    // Optimized quaternion-vector multiplication: q * v * q^(-1)
    // t = 2 * cross(q.xyz, v)
    float tx = 2.0f * (q.y * vz - q.z * vy);
    float ty = 2.0f * (q.z * vx - q.x * vz);
    float tz = 2.0f * (q.x * vy - q.y * vx);
    
    // v' = v + q.w * t + cross(q.xyz, t)
    float rx = vx + q.w * tx + (q.y * tz - q.z * ty);
    float ry = vy + q.w * ty + (q.z * tx - q.x * tz);
    float rz = vz + q.w * tz + (q.x * ty - q.y * tx);
    
    out[3*idx+0] = rx;
    out[3*idx+1] = ry;
    out[3*idx+2] = rz;
}

__global__ void batch_rotate_kernel(const Quaternion q, const float* input_points, 
                                   float* output_points, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        quaternion_rotate_point(q, input_points, output_points, idx);
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::ostringstream ss; \
        ss << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__; \
        throw std::runtime_error(ss.str()); \
    } \
} while(0)

class CUDAQuaternionProcessor {
private:
    float* d_input;
    float* d_output;
    float* h_pinned_input;
    float* h_pinned_output;
    cudaStream_t stream;
    size_t max_points;
    
public:
    CUDAQuaternionProcessor(size_t max_points = 1024*1024) : max_points(max_points) {
        // Allocate pinned host memory for faster transfers
        CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_input, 
                                sizeof(float) * max_points * 3, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_pinned_output, 
                                sizeof(float) * max_points * 3, cudaHostAllocDefault));
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)&d_input, sizeof(float) * max_points * 3));
        CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * max_points * 3));
        
        // Create CUDA stream for async operations
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    
    ~CUDAQuaternionProcessor() {
        if (h_pinned_input) cudaFreeHost(h_pinned_input);
        if (h_pinned_output) cudaFreeHost(h_pinned_output);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
    
    std::vector<float> process_points(const Quaternion& quat, const std::vector<float>& points) {
        if (points.size() % 3 != 0) {
            throw std::runtime_error("Points array size must be multiple of 3");
        }
        
        int num_points = points.size() / 3;
        if (num_points > max_points) {
            throw std::runtime_error("Too many points for current buffer size");
        }
        
        // Normalize quaternion
        Quaternion q = quat.normalize();
        
        // Copy input data to pinned memory
        memcpy(h_pinned_input, points.data(), sizeof(float) * points.size());
        
        // Async memory transfer H2D
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_pinned_input, 
                                  sizeof(float) * points.size(), 
                                  cudaMemcpyHostToDevice, stream));
        
        // Configure kernel launch parameters
        int blockSize = 256;
        int gridSize = (num_points + blockSize - 1) / blockSize;
        
        // Launch kernel
        batch_rotate_kernel<<<gridSize, blockSize, 0, stream>>>(q, d_input, d_output, num_points);
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Async memory transfer D2H
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_output, d_output, 
                                  sizeof(float) * points.size(), 
                                  cudaMemcpyDeviceToHost, stream));
        
        // Synchronize stream
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Return results
        std::vector<float> result(points.size());
        memcpy(result.data(), h_pinned_output, sizeof(float) * points.size());
        
        return result;
    }
};

// Simple JSON implementation for this worker
JSONValue JSONValue::parse_simple(const std::string& json) {
    JSONValue result;
    
    if (json.front() == '{' && json.back() == '}') {
        result.is_object = true;
        // Simple object parsing - extract key-value pairs
        std::string content = json.substr(1, json.length()-2);
        
        // For this worker, we only need to extract specific fields
        size_t job_pos = content.find("\"jobId\"");
        if (job_pos != std::string::npos) {
            size_t colon = content.find(':', job_pos);
            size_t quote1 = content.find('"', colon);
            size_t quote2 = content.find('"', quote1 + 1);
            if (quote1 != std::string::npos && quote2 != std::string::npos) {
                JSONValue job_val;
                job_val.is_string = true;
                job_val.str_val = content.substr(quote1+1, quote2-quote1-1);
                result.object_val.push_back(job_val);
            }
        }
    } else if (json.front() == '[' && json.back() == ']') {
        result.is_array = true;
        std::string content = json.substr(1, json.length()-2);
        
        std::istringstream ss(content);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                float val = std::stof(token);
                result.array_val.push_back(val);
            } catch (...) {}
        }
    } else {
        result.is_string = true;
        if (json.front() == '"' && json.back() == '"') {
            result.str_val = json.substr(1, json.length()-2);
        } else {
            result.str_val = json;
        }
    }
    
    return result;
}

std::string JSONValue::get_string(const std::string& key) const {
    if (is_string) return str_val;
    return "";
}

std::vector<float> JSONValue::get_array(const std::string& key) const {
    if (is_array) return array_val;
    return {};
}

JSONValue JSONValue::get_object(const std::string& key) const {
    if (is_object && !object_val.empty()) {
        return object_val[0]; // Simplified for this worker
    }
    return JSONValue{};
}

// Production JSON output function
std::string create_json_response(const std::string& job_id, const std::vector<float>& rotated_points, 
                                const std::string& status = "success", const std::string& error = "") {
    std::ostringstream json;
    json << "{";
    json << "\"jobId\":\"" << job_id << "\",";
    json << "\"status\":\"" << status << "\"";
    
    if (status == "success") {
        json << ",\"rotated\":[";
        for (size_t i = 0; i < rotated_points.size(); ++i) {
            if (i > 0) json << ",";
            json << rotated_points[i];
        }
        json << "]";
        json << ",\"points_processed\":" << (rotated_points.size() / 3);
    }
    
    if (!error.empty()) {
        json << ",\"error\":\"" << error << "\"";
    }
    
    json << "}";
    return json.str();
}

int main() {
    try {
        // Initialize CUDA processor
        CUDAQuaternionProcessor processor;
        
        // Read JSON input from stdin
        std::string input_line;
        std::string full_input;
        while (std::getline(std::cin, input_line)) {
            full_input += input_line;
        }
        
        if (full_input.empty()) {
            std::cerr << "No input provided" << std::endl;
            std::cout << create_json_response("unknown", {}, "error", "No input provided") << std::endl;
            return 1;
        }
        
        // Parse input JSON (simplified parsing for this worker)
        std::string job_id = "unknown";
        Quaternion quat = {1.0f, 0.0f, 0.0f, 0.0f};
        std::vector<float> points;
        
        // Extract jobId
        size_t job_start = full_input.find("\"jobId\":\"");
        if (job_start != std::string::npos) {
            job_start += 9; // Length of "jobId":""
            size_t job_end = full_input.find('"', job_start);
            if (job_end != std::string::npos) {
                job_id = full_input.substr(job_start, job_end - job_start);
            }
        }
        
        // Extract quaternion
        size_t quat_start = full_input.find("\"quat\":{");
        if (quat_start != std::string::npos) {
            size_t quat_end = full_input.find('}', quat_start);
            std::string quat_json = full_input.substr(quat_start + 8, quat_end - quat_start - 8);
            
            // Parse w, x, y, z values
            auto extract_float = [&](const std::string& key) -> float {
                size_t pos = quat_json.find("\"" + key + "\":");
                if (pos != std::string::npos) {
                    pos = quat_json.find(':', pos) + 1;
                    size_t end = quat_json.find_first_of(",}", pos);
                    std::string val_str = quat_json.substr(pos, end - pos);
                    return std::stof(val_str);
                }
                return key == "w" ? 1.0f : 0.0f;
            };
            
            quat.w = extract_float("w");
            quat.x = extract_float("x");
            quat.y = extract_float("y");
            quat.z = extract_float("z");
        }
        
        // Extract points array
        size_t points_start = full_input.find("\"points\":[");
        if (points_start != std::string::npos) {
            points_start += 10; // Length of "points":["
            size_t points_end = full_input.find(']', points_start);
            std::string points_str = full_input.substr(points_start, points_end - points_start);
            
            std::istringstream ss(points_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    // Remove whitespace
                    token.erase(0, token.find_first_not_of(" \t"));
                    token.erase(token.find_last_not_of(" \t") + 1);
                    float val = std::stof(token);
                    points.push_back(val);
                } catch (...) {
                    // Skip invalid values
                }
            }
        }
        
        // Validate input
        if (points.empty() || points.size() % 3 != 0) {
            std::cout << create_json_response(job_id, {}, "error", "Invalid or empty points array") << std::endl;
            return 1;
        }
        
        // Process points with GPU
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> rotated = processor.process_points(quat, points);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Output success response
        std::cout << create_json_response(job_id, rotated) << std::endl;
        
        // Log performance to stderr
        std::cerr << "Processed " << (points.size()/3) << " points in " 
                  << duration.count() << " microseconds" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cout << create_json_response("unknown", {}, "error", e.what()) << std::endl;
        return 1;
    }
}