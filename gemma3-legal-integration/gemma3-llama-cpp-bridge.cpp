// gemma3-llama-cpp-bridge.cpp
// Native Windows llama.cpp bridge for Gemma 3 legal model
// Compile with: cl /O2 /EHsc /I"llama.cpp" /DGGML_USE_CUDA gemma3-llama-cpp-bridge.cpp /link /LIBPATH:"cuda/lib/x64" cudart.lib cublas.lib

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <chrono>

#include "llama.h"
#include "common.h"
#include "ggml-cuda.h"

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

class Gemma3LegalBridge {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context_params ctx_params;
    llama_model_params model_params;
    
    std::atomic<bool> running{true};
    std::mutex queue_mutex;
    std::queue<std::string> request_queue;
    
    // GPU configuration for RTX 3060 Ti
    struct GPUConfig {
        int n_gpu_layers = 35;  // Optimized for 8GB VRAM
        int main_gpu = 0;
        float tensor_split[LLAMA_MAX_DEVICES] = {0};
        bool use_mmap = true;
        bool use_mlock = false;
    } gpu_config;

    // Legal-specific parameters
    struct LegalParams {
        int n_ctx = 4096;
        int n_batch = 512;
        int n_threads = 8;
        float temperature = 0.1f;
        int top_k = 40;
        float top_p = 0.9f;
        float repeat_penalty = 1.1f;
        int seed = 42;
    } legal_params;

public:
    Gemma3LegalBridge() {
        initialize_cuda();
        setup_model_params();
    }

    ~Gemma3LegalBridge() {
        cleanup();
    }

    void initialize_cuda() {
        #ifdef GGML_USE_CUDA
        ggml_cuda_set_device(gpu_config.main_gpu);
        
        // Check CUDA availability
        int device_count = ggml_cuda_get_device_count();
        if (device_count > 0) {
            size_t free_mem, total_mem;
            ggml_cuda_get_device_memory(0, &free_mem, &total_mem);
            
            std::cout << "CUDA initialized: " << device_count << " device(s) found\n";
            std::cout << "GPU Memory: " << (free_mem / 1024 / 1024) << "MB free / " 
                      << (total_mem / 1024 / 1024) << "MB total\n";
        }
        #endif
    }

    void setup_model_params() {
        model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_config.n_gpu_layers;
        model_params.main_gpu = gpu_config.main_gpu;
        model_params.tensor_split = gpu_config.tensor_split;
        model_params.use_mmap = gpu_config.use_mmap;
        model_params.use_mlock = gpu_config.use_mlock;

        ctx_params = llama_context_default_params();
        ctx_params.n_ctx = legal_params.n_ctx;
        ctx_params.n_batch = legal_params.n_batch;
        ctx_params.n_threads = legal_params.n_threads;
        ctx_params.n_threads_batch = legal_params.n_threads;
        ctx_params.seed = legal_params.seed;
        
        // Enable Flash Attention for better performance
        ctx_params.flash_attn = true;
        
        // Set rope frequency scaling for longer contexts
        ctx_params.rope_freq_scale = 1.0f;
        ctx_params.rope_freq_base = 10000.0f;
    }

    bool load_model(const std::string& model_path) {
        std::cout << "Loading Gemma 3 Legal model from: " << model_path << "\n";
        
        model = llama_load_model_from_file(model_path.c_str(), model_params);
        if (!model) {
            std::cerr << "Failed to load model\n";
            return false;
        }

        ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to create context\n";
            return false;
        }

        std::cout << "Model loaded successfully\n";
        print_model_info();
        return true;
    }

    void print_model_info() {
        const auto n_vocab = llama_n_vocab(model);
        const auto n_ctx_train = llama_n_ctx_train(model);
        const auto n_embd = llama_n_embd(model);
        const auto n_layer = llama_n_layer(model);
        
        std::cout << "Model info:\n";
        std::cout << "  Vocabulary size: " << n_vocab << "\n";
        std::cout << "  Context size (training): " << n_ctx_train << "\n";
        std::cout << "  Embedding dimensions: " << n_embd << "\n";
        std::cout << "  Number of layers: " << n_layer << "\n";
        std::cout << "  GPU layers loaded: " << gpu_config.n_gpu_layers << "\n";
    }

    std::string process_legal_prompt(const std::string& prompt, 
                                     const std::string& system_prompt = "") {
        // Prepare the full prompt with legal context
        std::string full_prompt = build_legal_prompt(prompt, system_prompt);
        
        // Tokenize
        std::vector<llama_token> tokens = tokenize(full_prompt);
        
        // Check if prompt exceeds context
        if (tokens.size() > legal_params.n_ctx - 512) {
            std::cerr << "Warning: Prompt too long, truncating\n";
            tokens.resize(legal_params.n_ctx - 512);
        }

        // Prepare batch for processing
        llama_batch batch = prepare_batch(tokens);
        
        // Process tokens through model
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode\n";
            llama_batch_free(batch);
            return "";
        }

        // Generate response
        std::string response = generate_response(tokens);
        
        llama_batch_free(batch);
        return response;
    }

    std::string generate_response(std::vector<llama_token>& input_tokens) {
        std::string response;
        int n_cur = input_tokens.size();
        int n_len = 2000; // Max response length
        
        llama_token new_token_id;
        
        for (int i = 0; i < n_len; i++) {
            // Sample next token
            auto logits = llama_get_logits_ith(ctx, -1);
            
            // Apply temperature and sampling
            std::vector<llama_token_data> candidates;
            candidates.reserve(llama_n_vocab(model));
            
            for (llama_token token_id = 0; token_id < llama_n_vocab(model); token_id++) {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }
            
            llama_token_data_array candidates_p = {
                candidates.data(), 
                candidates.size(), 
                false
            };
            
            // Apply sampling parameters
            llama_sample_top_k(ctx, &candidates_p, legal_params.top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, legal_params.top_p, 1);
            llama_sample_temp(ctx, &candidates_p, legal_params.temperature);
            
            new_token_id = llama_sample_token(ctx, &candidates_p);
            
            // Check for EOS
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }
            
            // Add to response
            response += llama_token_to_piece(ctx, new_token_id);
            
            // Update context
            input_tokens.push_back(new_token_id);
            
            llama_batch batch = prepare_batch({new_token_id});
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                break;
            }
            llama_batch_free(batch);
        }
        
        return response;
    }

    std::vector<float> generate_embeddings(const std::string& text) {
        std::vector<llama_token> tokens = tokenize(text);
        
        // Process tokens
        llama_batch batch = prepare_batch(tokens);
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return {};
        }
        
        // Extract embeddings from last layer
        const float* embeddings = llama_get_embeddings(ctx);
        const int n_embd = llama_n_embd(model);
        
        std::vector<float> result;
        if (embeddings) {
            result.assign(embeddings, embeddings + n_embd);
        }
        
        llama_batch_free(batch);
        return result;
    }

    // JSON-RPC server for integration
    void start_rpc_server(int port = 8095) {
        #ifdef _WIN32
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            std::cerr << "WSAStartup failed\n";
            return;
        }
        #endif

        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == 0) {
            std::cerr << "Socket creation failed\n";
            return;
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Bind failed\n";
            return;
        }

        if (listen(server_fd, 3) < 0) {
            std::cerr << "Listen failed\n";
            return;
        }

        std::cout << "Gemma3 Legal RPC server listening on port " << port << "\n";

        while (running) {
            int addrlen = sizeof(address);
            int client_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
            
            if (client_socket < 0) {
                continue;
            }

            // Handle request in separate thread
            std::thread([this, client_socket]() {
                handle_rpc_request(client_socket);
            }).detach();
        }

        #ifdef _WIN32
        closesocket(server_fd);
        WSACleanup();
        #else
        close(server_fd);
        #endif
    }

private:
    std::string build_legal_prompt(const std::string& prompt, 
                                   const std::string& system_prompt) {
        std::string full_prompt;
        
        if (!system_prompt.empty()) {
            full_prompt = system_prompt + "\n\n";
        } else {
            full_prompt = "You are a legal AI assistant trained on case law, statutes, and legal documents. "
                         "Provide accurate, detailed legal analysis while noting this is not legal advice.\n\n";
        }
        
        full_prompt += "User Query: " + prompt + "\n\nLegal Analysis:";
        return full_prompt;
    }

    std::vector<llama_token> tokenize(const std::string& text) {
        int n_tokens = text.length() + 100;
        std::vector<llama_token> tokens(n_tokens);
        
        n_tokens = llama_tokenize(model, text.c_str(), text.length(), 
                                  tokens.data(), tokens.size(), true, false);
        
        tokens.resize(n_tokens);
        return tokens;
    }

    llama_batch prepare_batch(const std::vector<llama_token>& tokens) {
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        
        for (size_t i = 0; i < tokens.size(); i++) {
            llama_batch_add(batch, tokens[i], i, {0}, false);
        }
        
        batch.logits[batch.n_tokens - 1] = true;
        return batch;
    }

    void handle_rpc_request(int client_socket) {
        char buffer[8192] = {0};
        int valread = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
        
        if (valread > 0) {
            std::string request(buffer);
            std::string response = process_rpc_call(request);
            send(client_socket, response.c_str(), response.length(), 0);
        }
        
        #ifdef _WIN32
        closesocket(client_socket);
        #else
        close(client_socket);
        #endif
    }

    std::string process_rpc_call(const std::string& request) {
        // Parse JSON-RPC request and route to appropriate method
        // This is a simplified version - you'd want proper JSON parsing
        
        if (request.find("\"method\":\"process\"") != std::string::npos) {
            // Extract prompt from request
            size_t prompt_start = request.find("\"prompt\":\"") + 10;
            size_t prompt_end = request.find("\"", prompt_start);
            std::string prompt = request.substr(prompt_start, prompt_end - prompt_start);
            
            std::string result = process_legal_prompt(prompt);
            
            return "{\"jsonrpc\":\"2.0\",\"result\":\"" + result + "\",\"id\":1}";
        }
        
        return "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32601,\"message\":\"Method not found\"},\"id\":1}";
    }

    void cleanup() {
        running = false;
        
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
        
        llama_backend_free();
    }
};

int main(int argc, char** argv) {
    std::string model_path = "./local-models/gemma3-legal.gguf";
    
    if (argc > 1) {
        model_path = argv[1];
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    Gemma3LegalBridge bridge;
    
    if (!bridge.load_model(model_path)) {
        return 1;
    }

    // Start RPC server for integration
    bridge.start_rpc_server(8095);
    
    return 0;
}
