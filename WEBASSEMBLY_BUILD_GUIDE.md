# WebAssembly Build Guide for llama.cpp

## Current Status
- ✅ **Infrastructure**: Complete WebAssembly service layer ready
- ✅ **Services**: `webasm-llama-complete.ts`, `llama-worker.js`, `webasm-ai-adapter.ts`
- ✅ **Demo**: Complete demo page at `/demo/webasm-ai-complete`
- ❌ **Build Artifacts**: Missing `bin/llama.wasm` and `bin/llama.js`

## Option 1: Use Existing Go Services (Recommended)

Your system already has working AI inference via:
- **Ollama**: `gemma3-legal` model (7.3 GB)
- **Go Services**: Enhanced RAG service on port 8094
- **GPU Acceleration**: RTX 3060 Ti support

The WebAssembly layer can delegate to these existing services:

```typescript
// In llama-worker.js - Use Go backend fallback
const goBackendUrl = 'http://localhost:8094/api/rag';

async function generateWithGo(prompt, options) {
  const response = await fetch(goBackendUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      temperature: options.temperature,
      max_tokens: options.maxTokens
    })
  });
  return response.json();
}
```

## Option 2: Build llama.cpp WebAssembly

If you need true client-side inference:

### Prerequisites
- **Emscripten SDK**: For WebAssembly compilation
- **llama.cpp source**: From official repository
- **Model files**: GGUF format models

### Build Commands
```bash
# Install Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build for WebAssembly
emcmake cmake -B build-em -DCMAKE_BUILD_TYPE=Release -DLLAMA_WASM=1
cmake --build build-em --config Release

# Copy artifacts
cp build-em/bin/llama.wasm ../deeds-web-app/bin/
cp build-em/bin/llama.js ../deeds-web-app/bin/
```

## Option 3: Create Placeholder Structure

For immediate development, create placeholder files:

### File Structure
```
bin/
├── llama.wasm          # Compiled WebAssembly module
├── llama.js            # JavaScript glue code
└── README.md          # Build instructions

sveltekit-frontend/static/
├── wasm/
│   ├── llama.wasm     # Copy of WASM module for web serving
│   └── llama.js       # Copy of JS glue code
└── models/
    └── gemma-3-legal-8b-q4_k_m.gguf  # GGUF model file
```

## Integration Points

### 1. Worker Initialization
```javascript
// In llama-worker.js
async function initializeWasm() {
  try {
    const wasmResponse = await fetch('/wasm/llama.wasm');
    const wasmBytes = await wasmResponse.arrayBuffer();
    const wasmModule = await WebAssembly.instantiate(wasmBytes, importObject);
    return wasmModule;
  } catch (error) {
    console.log('WASM loading failed, falling back to Go backend');
    return null;
  }
}
```

### 2. Service Integration
```typescript
// In webasm-llama-complete.ts
async function loadModel(config: LlamaConfig): Promise<boolean> {
  if (wasmModule) {
    // Use WebAssembly for client-side inference
    return await loadWasmModel(config.modelUrl);
  } else {
    // Fallback to existing Go service
    return await loadGoModel(config.modelUrl);
  }
}
```

## Recommended Next Steps

1. **Immediate**: Use Option 1 (Go backend delegation)
2. **Short-term**: Create Option 3 (placeholder structure)
3. **Long-term**: Implement Option 2 (full WASM build) if client-side inference is required

## Current Working Services

Your system is **already functional** without WebAssembly:
- **SvelteKit**: http://localhost:5173
- **Enhanced RAG**: http://localhost:8094
- **Ollama**: http://localhost:11434
- **PostgreSQL**: postgresql://localhost:5432
- **GPU**: RTX 3060 Ti acceleration ready

The WebAssembly layer adds **client-side inference** capability but isn't required for the core functionality.