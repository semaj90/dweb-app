GPU Leftovers — Next Steps

Goal: produce a minimal, reviewable checklist to finish GPU integration, remove runtime errors, and prepare a small demo.

1) Fix runtime ESM/CJS mismatches
- Replace CJS default imports that cause the browser error (example: `camelcase`) with named or ESM-compatible imports.
- Audit `package.json` exports/`type` fields and ensure Vite resolves ESM entry points.

2) Remove large generated storage from git index
- Exclude `storage/collections/**/payload_index/*` from git (or add to `.gitignore`) and remove the LOCK file from the index.
- Use `git rm --cached` for large generated files and commit the change.

3) Stabilize Service Worker and WebGPU bootstrap
- Ensure service worker imports are ESM-compatible (no CJS default-only modules).
- Limit WebGPU device request options to supported fields on Windows (avoid powerPreference when requestAdapter ignores it).

4) TypeScript / SvelteKit small-batch fixes
- Continue low-risk batches: test vitest imports (replace named imports), add ambient types for common globals, fix Loki simplesort usages.
- After each 3–5 file batch, run `npm run check:typescript` and collect deltas.

5) Create a small GPU demo page
- Add `yorha-demo` route that runs a tiny WebGPU compute shader or wasm-based inference using a tiny GGUF model, with a fallback to Ollama.

6) Documentation + Demo
- Consolidate diagrams into `docs/gpu-leftovers/diagram.svg` and add a short walkthrough.

Commands / quick actions

```powershell
# Remove LOCK from git index and ignore the payload index
git rm --cached "storage/collections/**/payload_index/LOCK" -r
echo "storage/collections/*/payload_index/*" >> .gitignore

# Commit docs only to avoid touching large files
git add docs/gpu-leftovers/* .gitignore
git commit -m "docs(gpu-leftovers): add summary + next steps"
git push origin main
```

If you want, I can run these steps now (I will avoid touching large `storage/` files by only committing docs and .gitignore changes).
