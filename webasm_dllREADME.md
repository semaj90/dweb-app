# deeds-web-app

Windows-native Legal AI web app and services.

## WASM builds (Emscripten)

- Emscripten (emsdk) is a local SDK/toolchain to compile C/C++ to WebAssembly. It is not a runtime DLL.
- Do not commit emsdk. It’s already ignored (see `deeds-web-app/.gitignore`: `src/lib/wasm/deps/emsdk/`).
- What we ship: `.wasm` plus a small JS loader/glue per module (for example, `rapid-json-parser.wasm` and its corresponding `.js`).
- Runtime: Browser and Node can load these `.wasm` modules without any extra DLLs.

### Native DLLs (outside this repo)

- PostgreSQL pgvector on Windows uses `vector.dll` within your Postgres installation. That’s external to this repo and not tracked.
- Optional GPU/CUDA flows (e.g., Go microservice) rely on NVIDIA DLLs such as `cudart64_12.dll`. These binaries are system/installer-provided and are not tracked. Our `.gitignore` includes vendor-specific DLL paths (e.g., under `Ollama/lib/ollama/cuda_v12/`).

## Best practices and architecture

- Context7 MCP integration: `deeds-web-app/best-practices/context7-integration-best-practices.md`
- Enhanced RAG system: `deeds-web-app/best-practices/enhanced-rag-best-practices.md`

These docs cover ports, health checks, error handling, streaming, caching, security, and deployment guidance. Use them with your Context7/MCP setup to keep the system production-ready.
