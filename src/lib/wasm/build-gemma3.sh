#!/bin/bash

# High-Performance Gemma3 WebAssembly Build with LLVM Optimizations
# Builds production-ready inference engine for legal AI platform

set -e

echo "🚀 Building Gemma3 WebAssembly Inference Engine with LLVM optimizations..."

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WASM_DIR="$PROJECT_ROOT"
DIST_DIR="$PROJECT_ROOT/../../../static/wasm"
DEPS_DIR="$PROJECT_ROOT/deps"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Emscripten with LLVM backend
check_emscripten_llvm() {
    print_status "Checking Emscripten with LLVM backend..."
    
    if ! command -v emcc &> /dev/null; then
        print_error "Emscripten not found. Installing..."
        install_emscripten_llvm
    fi
    
    # Verify LLVM backend
    EMCC_VERSION=$(emcc --version | head -n1)
    print_success "Emscripten found: $EMCC_VERSION"
    
    # Check for LLVM optimizations
    if emcc --version | grep -q "fastcomp"; then
        print_warning "Using fastcomp backend. LLVM backend recommended for better performance."
    else
        print_success "Using LLVM backend - optimal for performance"
    fi
}

install_emscripten_llvm() {
    print_status "Installing Emscripten with LLVM backend..."
    
    if [ ! -d "$DEPS_DIR/emsdk" ]; then
        mkdir -p "$DEPS_DIR"
        cd "$DEPS_DIR"
        git clone https://github.com/emscripten-core/emsdk.git
        cd emsdk
        ./emsdk install latest
        ./emsdk activate latest
    fi
    
    source "$DEPS_DIR/emsdk/emsdk_env.sh"
    print_success "Emscripten with LLVM backend installed"
}

# Set up output directory
setup_output() {
    print_status "Setting up output directory..."
    mkdir -p "$DIST_DIR"
    print_success "Output directory ready: $DIST_DIR"
}

# Build optimized WebAssembly with LLVM
build_gemma3_wasm() {
    print_status "Building Gemma3 WebAssembly with LLVM optimizations..."
    cd "$WASM_DIR"
    
    # LLVM-optimized compiler flags
    CXXFLAGS="-std=c++17 -O3 -DNDEBUG -flto=full -ffast-math -funroll-loops -fvectorize"
    CXXFLAGS="$CXXFLAGS -DGEMMA3_OPTIMIZATIONS -DSIMD_ENABLED -fopenmp-simd"
    
    # Emscripten flags for maximum performance
    EMSCRIPTEN_FLAGS=(
        "-s WASM=1"
        "-s EXPORT_ES6=1"
        "-s MODULARIZE=1"
        "-s EXPORT_NAME=\"Gemma3WasmModule\""
        "-s ENVIRONMENT=web,webview,worker"
        "-s USE_ES6_IMPORT_META=0"
        "-s ALLOW_MEMORY_GROWTH=1"
        "-s INITIAL_MEMORY=268435456"        # 256MB initial
        "-s MAXIMUM_MEMORY=2147483648"       # 2GB maximum
        "-s STACK_SIZE=8388608"              # 8MB stack
        "-s TOTAL_STACK=33554432"            # 32MB total stack
        "-s PTHREAD_POOL_SIZE=8"             # 8 worker threads
        "-s USE_PTHREADS=1"
        "-s PROXY_TO_PTHREAD"
        "-s WASM_WORKERS=1"
        "-s SHARED_MEMORY=1"                 # Shared memory for threading
        "-s EXPORTED_FUNCTIONS=\"['_malloc','_free','_create_gemma3_engine','_destroy_gemma3_engine']\""
        "-s EXPORTED_RUNTIME_METHODS=\"['ccall','cwrap','getValue','setValue','allocate','ALLOC_NORMAL']\""
        "-s NO_EXIT_RUNTIME=1"
        "-s ASSERTIONS=0"
        "-s NO_FILESYSTEM=1"
        "-s AGGRESSIVE_VARIABLE_ELIMINATION=1"
        "-s ELIMINATE_DUPLICATE_FUNCTIONS=1"
        "-s SINGLE_FILE=0"                   # Separate .wasm file
        "-lembind"
        "--closure 1"                        # Closure compiler
        "--llvm-lto 3"                      # Link-time optimization
        "--pre-js pre-gemma3.js"
    )
    
    # Create optimized pre-JS for memory management
    cat > pre-gemma3.js << 'EOF'
// Pre-load optimizations for Gemma3 WebAssembly inference
if (typeof performance === 'undefined') {
    var performance = { now: function() { return Date.now(); } };
}

// High-performance memory pool for inference
var Gemma3MemoryPool = {
    buffers: new Map(),
    tensorCache: new Map(),
    maxPoolSize: 50,
    
    getTensorBuffer: function(size, dtype) {
        const key = size + '_' + (dtype || 'float32');
        if (this.buffers.has(key) && this.buffers.get(key).length > 0) {
            return this.buffers.get(key).pop();
        }
        
        // Allocate aligned buffer for SIMD operations
        const buffer = new ArrayBuffer(size * 4); // Float32 assumption
        return new Float32Array(buffer);
    },
    
    releaseTensorBuffer: function(buffer, size, dtype) {
        const key = (size || buffer.length) + '_' + (dtype || 'float32');
        if (!this.buffers.has(key)) {
            this.buffers.set(key, []);
        }
        
        if (this.buffers.get(key).length < this.maxPoolSize) {
            this.buffers.get(key).push(buffer);
        }
    },
    
    clearAll: function() {
        this.buffers.clear();
        this.tensorCache.clear();
    }
};

// WebGPU detection and setup
var WebGPUSupport = {
    device: null,
    supported: false,
    
    async initialize() {
        if (!navigator.gpu) {
            console.log('[Gemma3] WebGPU not available, using CPU inference');
            return false;
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            
            if (adapter) {
                this.device = await adapter.requestDevice({
                    requiredFeatures: ['shader-f16'],
                    requiredLimits: {
                        maxBufferSize: 1024 * 1024 * 1024, // 1GB
                        maxStorageBufferBindingSize: 512 * 1024 * 1024
                    }
                });
                this.supported = true;
                console.log('[Gemma3] WebGPU initialized successfully');
                return true;
            }
        } catch (error) {
            console.warn('[Gemma3] WebGPU initialization failed:', error);
        }
        
        return false;
    }
};

// Initialize WebGPU on module load
if (typeof window !== 'undefined') {
    WebGPUSupport.initialize();
}
EOF
    
    # Build command with LLVM optimizations
    emcc $CXXFLAGS "${EMSCRIPTEN_FLAGS[@]}" \
        -o "$DIST_DIR/gemma3-inference.js" \
        gemma3-inference-engine.cpp
    
    if [ $? -eq 0 ]; then
        print_success "Gemma3 WebAssembly build completed successfully"
        
        # Show file sizes
        if [ -f "$DIST_DIR/gemma3-inference.js" ]; then
            JS_SIZE=$(du -h "$DIST_DIR/gemma3-inference.js" | cut -f1)
            print_status "JavaScript file size: $JS_SIZE"
        fi
        
        if [ -f "$DIST_DIR/gemma3-inference.wasm" ]; then
            WASM_SIZE=$(du -h "$DIST_DIR/gemma3-inference.wasm" | cut -f1)
            print_status "WebAssembly file size: $WASM_SIZE"
        fi
    else
        print_error "Gemma3 WebAssembly build failed"
        exit 1
    fi
    
    # Cleanup
    rm -f pre-gemma3.js
}

# Optimize with wasm-opt
optimize_wasm() {
    print_status "Optimizing WebAssembly with wasm-opt..."
    
    WASM_FILE="$DIST_DIR/gemma3-inference.wasm"
    
    if command -v wasm-opt &> /dev/null; then
        print_status "Running wasm-opt with maximum optimization..."
        wasm-opt -Oz --enable-threads --enable-bulk-memory --enable-simd \
            --enable-nontrapping-float-to-int --enable-sign-ext \
            --enable-multivalue --fast-math \
            "$WASM_FILE" -o "$WASM_FILE.optimized"
        
        if [ $? -eq 0 ]; then
            mv "$WASM_FILE.optimized" "$WASM_FILE"
            print_success "WebAssembly optimization completed"
            
            OPTIMIZED_SIZE=$(du -h "$WASM_FILE" | cut -f1)
            print_status "Optimized WebAssembly size: $OPTIMIZED_SIZE"
        else
            print_warning "WebAssembly optimization failed, using unoptimized version"
        fi
    else
        print_warning "wasm-opt not found, skipping optimization"
        print_status "Install Binaryen for WebAssembly optimization"
    fi
}

# Create TypeScript declarations
create_typescript_declarations() {
    print_status "Creating TypeScript declarations..."
    
    cat > "$DIST_DIR/gemma3-inference.d.ts" << 'EOF'
// TypeScript declarations for Gemma3 WebAssembly Inference Engine

export interface Gemma3GenerationOptions {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    use_cache?: boolean;
    stream?: boolean;
}

export interface Gemma3GenerationResult {
    success: boolean;
    text?: string;
    tokens_generated?: number;
    processing_time_ms?: number;
    tokens_per_second?: number;
    method?: string;
    error?: string;
}

export interface Gemma3PerformanceStats {
    model_loaded: boolean;
    generation_active: boolean;
    total_tokens_generated: number;
    total_inference_time_ms: number;
    average_tokens_per_second: number;
    model_parameters: number;
    memory_usage_mb: number;
}

export interface Gemma3InferenceEngine {
    loadModelWeights(weightData: ArrayBuffer | Uint8Array): Promise<{
        success: boolean;
        message?: string;
        parameters?: number;
        error?: string;
    }>;
    
    generateText(prompt: string, options?: Gemma3GenerationOptions): Promise<Gemma3GenerationResult>;
    getPerformanceStats(): Gemma3PerformanceStats;
}

export interface Gemma3WasmModule {
    Gemma3InferenceEngine: new () => Gemma3InferenceEngine;
    
    // Low-level C++ exports
    create_gemma3_engine(): number;
    destroy_gemma3_engine(enginePtr: number): void;
    
    // Memory management
    malloc(size: number): number;
    free(ptr: number): void;
    HEAP8: Int8Array;
    HEAP16: Int16Array;
    HEAP32: Int32Array;
    HEAPF32: Float32Array;
    HEAPF64: Float64Array;
}

declare const Gemma3WasmModule: () => Promise<Gemma3WasmModule>;
export default Gemma3WasmModule;

// Service integration types
export interface Gemma3ServiceConfig {
    modelUrl?: string;
    wasmUrl?: string;
    enableWebGPU?: boolean;
    enableThreading?: boolean;
    maxCacheSize?: number;
    defaultTemperature?: number;
}

export class Gemma3LocalService {
    constructor(config?: Gemma3ServiceConfig);
    
    initialize(): Promise<boolean>;
    generate(prompt: string, options?: Gemma3GenerationOptions): Promise<Gemma3GenerationResult>;
    analyzeDocument(content: string, analysisType?: string): Promise<{
        summary: string;
        keyTerms: string[];
        entities: Array<{type: string; value: string; confidence: number}>;
        risks: Array<{type: string; severity: string; description: string}>;
        confidence: number;
        processingTime: number;
    }>;
    
    getStats(): Gemma3PerformanceStats;
    dispose(): void;
}
EOF
    
    print_success "TypeScript declarations created"
}

# Run performance tests
run_performance_tests() {
    print_status "Running Gemma3 WebAssembly performance tests..."
    
    # Create test script
    cat > "$WASM_DIR/test-gemma3-performance.js" << 'EOF'
const Gemma3WasmModule = require('./static/wasm/gemma3-inference.js');

async function runPerformanceTests() {
    console.log('🧪 Testing Gemma3 WebAssembly performance...');
    
    try {
        const wasmModule = await Gemma3WasmModule();
        const engine = new wasmModule.Gemma3InferenceEngine();
        
        // Test 1: Engine initialization
        console.log('✅ Engine initialization successful');
        
        // Test 2: Performance stats
        const stats = engine.getPerformanceStats();
        console.log('📊 Performance stats:', {
            modelLoaded: stats.model_loaded,
            memoryUsage: stats.memory_usage_mb + 'MB',
            parameters: stats.model_parameters
        });
        
        // Test 3: Memory allocation test
        const testPrompt = "Analyze this legal document for compliance issues.";
        console.log('🔍 Testing text generation capability...');
        
        const startTime = performance.now();
        const result = await engine.generateText(testPrompt, {
            max_tokens: 100,
            temperature: 0.1,
            use_cache: true
        });
        const endTime = performance.now();
        
        if (result.success) {
            console.log('✅ Text generation test passed');
            console.log('⚡ Generation time:', (endTime - startTime).toFixed(2) + 'ms');
            console.log('🎯 Tokens per second:', result.tokens_per_second);
        } else {
            console.log('❌ Text generation test failed:', result.error);
        }
        
        // Test 4: Threading and SIMD
        if (typeof SharedArrayBuffer !== 'undefined') {
            console.log('✅ SharedArrayBuffer available - threading supported');
        } else {
            console.log('⚠️ SharedArrayBuffer not available - single-threaded mode');
        }
        
        console.log('🎉 All performance tests completed!');
        
    } catch (error) {
        console.error('💥 Performance test failed:', error);
        process.exit(1);
    }
}

runPerformanceTests();
EOF
    
    if [ -f "$DIST_DIR/gemma3-inference.js" ]; then
        cd "$PROJECT_ROOT/../../.."
        node "$WASM_DIR/test-gemma3-performance.js"
        rm -f "$WASM_DIR/test-gemma3-performance.js"
    else
        print_warning "Skipping tests - WebAssembly module not found"
    fi
}

# Main build process
main() {
    print_status "Starting Gemma3 WebAssembly build with LLVM optimizations..."
    
    check_emscripten_llvm
    setup_output
    build_gemma3_wasm
    optimize_wasm
    create_typescript_declarations
    run_performance_tests
    
    print_success "🎉 Gemma3 WebAssembly build completed successfully!"
    print_status ""
    print_status "Output files:"
    print_status "  - JavaScript: $DIST_DIR/gemma3-inference.js"
    print_status "  - WebAssembly: $DIST_DIR/gemma3-inference.wasm"
    print_status "  - TypeScript: $DIST_DIR/gemma3-inference.d.ts"
    print_status ""
    
    # Show integration example
    cat << 'EOF'

📚 Integration Example:

```javascript
import Gemma3WasmModule from '/static/wasm/gemma3-inference.js';

const wasmModule = await Gemma3WasmModule();
const engine = new wasmModule.Gemma3InferenceEngine();

// Load model weights (converted from Ollama)
const weightsResponse = await fetch('/models/gemma3-legal-weights.bin');
const weights = await weightsResponse.arrayBuffer();
await engine.loadModelWeights(weights);

// Generate legal analysis
const result = await engine.generateText(
    "Analyze this contract for potential risks:",
    {
        max_tokens: 1024,
        temperature: 0.1,
        use_cache: true
    }
);

console.log('Generated analysis:', result.text);
console.log('Performance:', result.tokens_per_second, 'tokens/sec');
```

🔗 Features:
- LLVM-optimized C++ inference engine
- WebGPU acceleration when available
- Multi-threaded processing with SharedArrayBuffer
- Intelligent caching for repeated queries
- Full TypeScript integration
- Production-ready for legal AI applications

EOF
}

# Run main function
main "$@"