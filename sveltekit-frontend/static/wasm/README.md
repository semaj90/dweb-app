# WebAssembly Files Directory

This directory should contain the compiled llama.cpp WebAssembly artifacts:

## Required Files
- `llama.wasm` - The compiled WebAssembly module
- `llama.js` - JavaScript glue code for WASM integration

## Current Status
❌ **Missing**: WebAssembly build artifacts not yet compiled

## Build Instructions
See `/WEBASSEMBLY_BUILD_GUIDE.md` in the project root for compilation steps.

## Fallback Behavior
The system automatically falls back to Go backend services when WASM files are unavailable:
- Enhanced RAG Service: http://localhost:8094
- Ollama Service: http://localhost:11434

## Integration
These files are loaded by:
- `/static/workers/llama-worker.js` - Web Worker for WASM inference
- `/src/lib/services/webasm-llama-complete.ts` - Service layer
- `/src/lib/adapters/webasm-ai-adapter.ts` - AI adapter interface