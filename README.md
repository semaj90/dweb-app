# Legal GPU Processor v2.0.0

## Quick Start

- Run: `START.bat`
- Access: http://localhost:5173

## WASM builds (Emscripten)

- Emscripten (emsdk) is a local SDK/toolchain to compile C/C++ to WebAssembly. It is not a runtime DLL.
- Do not commit emsdk. It’s already ignored (see `.gitignore`: `src/lib/wasm/deps/emsdk/`).
- What we ship: `.wasm` plus a small JS loader/glue per module (for example, `rapid-json-parser.wasm` and its corresponding `.js`).
- Runtime: Browser and Node can load these `.wasm` modules without any extra DLLs.

### Native DLLs (outside this repo)

- PostgreSQL pgvector on Windows uses `vector.dll` within your Postgres installation. That’s external and not tracked here.
- Optional GPU/CUDA flows (e.g., Go service) rely on NVIDIA DLLs like `cudart64_12.dll`. These are installer/system-provided and ignored in git.

See also: `src/lib/wasm/README.md` for a minimal emsdk build and loader outline.

## Context7 best practices

- Context7 integration: `best-practices/context7-integration-best-practices.md`
- Enhanced RAG best practices: `best-practices/enhanced-rag-best-practices.md`

These cover ports, streaming, caching, error handling, and deployment guidance to keep the system production-ready.

## Changed files reporter

You can generate a concise report of files changed since a date/time expression.

PowerShell examples:

```powershell
# last day
npm run changes:since -- "1 day ago"

# since a specific date
npm run changes:since -- "2025-08-09"

# direct node usage
node scripts/list-changes-since.mjs "1 day ago"
node scripts/list-changes-since.mjs "2025-08-09"
```

The script prints a summary and saves a timestamped report (CHANGES-since-YYYYMMDD-HHMMSS.txt) in this folder.
