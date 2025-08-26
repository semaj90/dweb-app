# GPU Leftovers

Summary diagram of readmes + integration diagram for the recent GPU work.

Purpose
- Capture the current state of GPU-related integrations, remaining TODOs, and a high-level integration sketch so reviewers can quickly see "what we worked on".

Included readmes / sources
- `sveltekit-frontend/AI-SYSTEM-README.md` (primary integration notes)
- `sveltekit-frontend/AI-SYNTHESIS-FULL-STACK-GUIDE.md` (pipeline design)
- `mcp-servers/context7-mcp-config.json` (MCP wiring)
- `go-microservice/` (GPU-related Go services)

ASCII integration diagram

  +-----------------+      +----------------+      +-------------+
  |  SvelteKit FE   | <--> |  Node API /    | <--> |  Go RAG &   |
  |  (UI, Worker)   |      |  MCP Context7  |      |  GPU Bridge | 
  +-----------------+      +----------------+      +------+------+ 
         |                           |                     |
         | WebSocket / HTTP          | MCP / gRPC          | QUIC / gRPC
         v                           v                     v
  +-----------------+        +----------------+    +-----------------
  | Service Worker  |        | Ollama / LLM   |    | GPU Runtime /   |
  | (WebGPU + GGUF) |        | endpoints      |    | vLLM / CUDA     |
  +-----------------+        +----------------+    +-----------------

Notes
- Several components are in draft or have experimental WebGPU/FlashAttention hooks.
- Some packages still import CJS-style defaults (e.g. `camelcase`) and require ESM-compatible imports; this causes runtime HMR errors in the browser.

See `NEXT_STEPS.md` for concrete actions.
