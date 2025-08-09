#!/bin/bash
# YoRHa Legal AI - WASM Neural Module Build Script
# Advanced C++ to WebAssembly compilation with neural optimizations

echo "════════════════════════════════════════════════════════════════════"
echo "🤖 YoRHa WASM Neural Module Builder v3.0"
echo "════════════════════════════════════════════════════════════════════"
echo

# Check for Emscripten
if ! command -v emcc &> /dev/null; then
    echo "❌ Emscripten not found. Installing Emscripten..."
    
    # Download and install Emscripten
    if [ ! -d "emsdk" ]; then
        git clone https://github.com/emscripten-core/emsdk.git
        cd emsdk
        ./emsdk install latest
        ./emsdk activate latest
        cd ..
    fi
    
    source emsdk/emsdk_env.sh
fi

echo "✅ Emscripten found - building YoRHa neural WASM module..."

# Create output directory
mkdir -p frontend/src/wasm/build

# Build YoRHa Neural WASM module with advanced optimizations
emcc frontend/src/wasm/yorha-neural-processor.cpp \
  -O3 \
  -s WASM=1 \
  -s USE_PTHREADS=1 \
  -s PTHREAD_POOL_SIZE=8 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="YoRHaNeuralWASM" \
  -s EXPORTED_FUNCTIONS="['_malloc', '_free', '_yorha_neural_init', '_yorha_process_neural_array', '_yorha_neural_confidence', '_yorha_benchmark_neural_processing']" \
  -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MAXIMUM_MEMORY=2147483648 \
  -s INITIAL_MEMORY=134217728 \
  -s STACK_SIZE=1048576 \
  -s TOTAL_STACK=1048576 \
  -s ASSERTIONS=0 \
  -s NO_EXIT_RUNTIME=1 \
  -s ENVIRONMENT='web,worker' \
  -s TEXTDECODER=2 \
  -s ABORTING_MALLOC=0 \
  -s ALLOW_TABLE_GROWTH=1 \
  -s DYNAMIC_EXECUTION=0 \
  --bind \
  --pre-js frontend/src/wasm/yorha-pre.js \
  --post-js frontend/src/wasm/yorha-post.js \
  -o frontend/src/wasm/build/yorha-neural-processor.js

if [ $? -eq 0 ]; then
    echo "✅ YoRHa WASM neural module built successfully!"
    echo "   Output: frontend/src/wasm/build/yorha-neural-processor.js"
    echo "   WASM: frontend/src/wasm/build/yorha-neural-processor.wasm"
    echo
    echo "📊 Build Statistics:"
    ls -lh frontend/src/wasm/build/yorha-neural-processor.*
    echo
    echo "🧠 Neural Module Features:"
    echo "   ✓ Multi-threaded processing (8 threads)"
    echo "   ✓ 2GB maximum memory allocation"
    echo "   ✓ Advanced neural network inference"
    echo "   ✓ GPU-accelerated matrix operations (simulated)"
    echo "   ✓ YoRHa document classification"
    echo "   ✓ Real-time performance analytics"
    echo
    echo "🚀 YoRHa WASM neural compilation complete!"
else
    echo "❌ YoRHa WASM neural module build failed"
    echo "   Check Emscripten installation and try again"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════"
