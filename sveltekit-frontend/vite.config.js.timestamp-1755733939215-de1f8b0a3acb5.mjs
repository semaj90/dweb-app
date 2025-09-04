// vite.config.js
import { sveltekit } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/vite/dist/node/index.js";
import UnoCSS from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/unocss/dist/vite.mjs";
import { resolve as resolve2 } from "path";
import { readFileSync as readFileSync2 } from "fs";

// src/lib/vite/vscode-error-logger.ts
import { resolve, dirname } from "path";
import { writeFileSync, existsSync, mkdirSync, readFileSync } from "fs";
function vscodeErrorLogger(options = {}) {
  const config = {
    enabled: true,
    logFile: resolve(process.cwd(), ".vscode/vite-errors.json"),
    maxEntries: 500,
    includeWarnings: true,
    includeSourceMaps: true,
    ...options
  };
  let server = void 0;
  let errorLog = { metadata: { lastUpdated: (/* @__PURE__ */ new Date()).toISOString(), version: 1 }, errors: [] };
  function loadLog() {
    try {
      if (existsSync(config.logFile)) {
        const raw = readFileSync(config.logFile, "utf8");
        const parsed = JSON.parse(raw);
        errorLog = {
          metadata: {
            lastUpdated: parsed?.metadata?.lastUpdated || (/* @__PURE__ */ new Date()).toISOString(),
            version: parsed?.metadata?.version || 1
          },
          errors: Array.isArray(parsed?.errors) ? parsed.errors : []
        };
      }
    } catch (e) {
    }
  }
  function saveLog() {
    try {
      const dir = dirname(config.logFile);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      errorLog.metadata = errorLog.metadata || { version: 1 };
      errorLog.metadata.lastUpdated = (/* @__PURE__ */ new Date()).toISOString();
      errorLog.errors = Array.isArray(errorLog.errors) ? errorLog.errors : [];
      writeFileSync(config.logFile, JSON.stringify(errorLog, null, 2));
    } catch (e) {
    }
  }
  function pushEntry(entry) {
    if (!config.enabled) return;
    errorLog.errors = errorLog.errors || [];
    errorLog.errors.unshift(entry);
    if (errorLog.errors.length > config.maxEntries) errorLog.errors.length = config.maxEntries;
    saveLog();
  }
  function normalizeViteError(err) {
    const entry = {
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      level: "error",
      message: err?.message || err?.text || String(err),
      stack: err?.stack || err?.stackStr,
      file: err?.id || err?.loc?.file || void 0,
      line: err?.loc?.line || err?.loc?.lineNumber || void 0,
      column: err?.loc?.column || void 0,
      frame: err?.frame || void 0,
      plugin: err?.plugin || void 0,
      buildPhase: "vite"
    };
    return entry;
  }
  return {
    name: "vscode-error-logger",
    configureServer(srv) {
      server = srv;
      loadLog();
      try {
        server.ws.on("vite:error", (payload) => {
          const e = normalizeViteError(payload.err || payload);
          pushEntry(e);
        });
        if (config.includeWarnings) {
          server.ws.on("vite:warning", (payload) => {
            const e = normalizeViteError(payload.warn || payload);
            e.level = "warn";
            pushEntry(e);
          });
        }
      } catch (e) {
      }
    },
    buildStart() {
      pushEntry({ timestamp: (/* @__PURE__ */ new Date()).toISOString(), level: "info", message: "Build started", buildPhase: "build" });
    },
    buildEnd(error) {
      if (error) {
        pushEntry({ timestamp: (/* @__PURE__ */ new Date()).toISOString(), level: "error", message: error.message || String(error), stack: error.stack, buildPhase: "build" });
      } else {
        pushEntry({ timestamp: (/* @__PURE__ */ new Date()).toISOString(), level: "info", message: "Build completed", buildPhase: "build" });
      }
    }
  };
}
var defaultVSCodeErrorConfig = {
  enabled: true,
  logFile: resolve(process.cwd(), ".vscode/vite-errors.json"),
  maxEntries: 1e3,
  includeWarnings: true,
  includeSourceMaps: true,
  autoOpenProblems: false,
  notificationLevel: "errors-only",
  integrateTasks: true,
  generateDiagnostics: true
};

// vite.config.js
var __vite_injected_original_import_meta_url = "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/vite.config.js";
async function findAvailablePort(startPort, maxAttempts = 10) {
  const net = await import("net");
  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    try {
      await new Promise((resolvePromise, reject) => {
        const server = net.createServer();
        server.listen(port, (err) => err ? reject(err) : server.close(resolvePromise));
        server.on("error", reject);
      });
      return port;
    } catch {
    }
  }
  return startPort;
}
var vite_config_default = defineConfig(async ({ mode }) => {
  const preferredPort = 5173;
  const availablePort = await findAvailablePort(preferredPort);
  const isProd = mode === "production";
  let pkgVersion = process.env.npm_package_version;
  try {
    if (!pkgVersion) {
      const pkgJson = JSON.parse(readFileSync2(new URL("./package.json", __vite_injected_original_import_meta_url), "utf8"));
      pkgVersion = pkgJson?.version || "1.0.0";
    }
  } catch (e) {
    pkgVersion = pkgVersion || "1.0.0";
  }
  return {
    plugins: [
      UnoCSS(),
      vscodeErrorLogger({
        enabled: !isProd,
        logFile: resolve2(".vscode/vite-errors.json"),
        maxEntries: 500,
        includeWarnings: true,
        includeSourceMaps: true
      }),
      sveltekit(),
      // Custom Go server integration middleware
      {
        name: "vite-plugin-go-integration",
        configureServer(server) {
          server.middlewares.use((req, _res, next) => {
            if (req.url?.startsWith("/api/go/")) {
              req.url = req.url.replace("/api/go", "");
            }
            next();
          });
        }
      }
    ],
    server: {
      port: availablePort,
      host: "0.0.0.0",
      cors: true,
      strictPort: false,
      hmr: { port: 3131, clientPort: 3131 },
      fs: { allow: ["..", "../../"] },
      proxy: {
        // Go microservice proxies
        "/api/go/enhanced-rag": { target: "http://localhost:8094", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/go\/enhanced-rag/, "") },
        "/api/go/upload": { target: "http://localhost:8093", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/go\/upload/, "") },
        "/api/go/cluster": { target: "http://localhost:8213", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/go\/cluster/, "") },
        "/api/go/xstate": { target: "http://localhost:8212", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/go\/xstate/, "") },
        // Direct backend service proxies (for /api/v1/* endpoints)
        "/api/v1/rag": { target: "http://localhost:8094", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/v1\/rag/, "/api/rag") },
        "/api/v1/ai": { target: "http://localhost:8094", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/v1\/ai/, "/api/ai") },
        "/api/v1/upload": { target: "http://localhost:8093", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/v1\/upload/, "/upload") },
        "/api/v1/vector": { target: "http://localhost:8094", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/v1\/vector/, "/api/vector") },
        "/api/v1/cluster": { target: "http://localhost:8213", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/v1\/cluster/, "/api/cluster") },
        // Ollama API proxy - preserve /api path
        "/api/ollama": { target: "http://localhost:11434", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/ollama/, "/api") },
        // Other service proxies
        "/api/nvidia-llama": { target: "http://localhost:8222", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/nvidia-llama/, "") },
        "/api/neo4j": { target: "http://localhost:7474", changeOrigin: true, rewrite: (path) => path.replace(/^\/api\/neo4j/, "") }
      }
    },
    preview: { port: availablePort + 1e3, host: "0.0.0.0", cors: true, strictPort: false },
    build: {
      target: "esnext",
      minify: isProd ? "esbuild" : false,
      sourcemap: !isProd,
      assetsInlineLimit: 4096,
      chunkSizeWarningLimit: 1e3,
      rollupOptions: {
        output: {
          manualChunks: {
            "ui-framework": ["bits-ui", "@melt-ui/svelte"],
            "icons": ["lucide-svelte"],
            "state-management": ["xstate", "@xstate/svelte", "svelte/store"],
            "css-engine": ["unocss", "tailwindcss", "tailwind-merge"],
            "ai-processing": ["bullmq", "ioredis", "socket.io-client"],
            "client-data": ["lokijs", "fuse.js"],
            "validation": ["zod", "sveltekit-superforms"]
          },
          assetFileNames: (assetInfo) => {
            const name = assetInfo.fileName || (Array.isArray(assetInfo.names) ? assetInfo.names[0] : "");
            const ext = (name.split(".").pop() || "").toLowerCase();
            if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) return `assets/images/[name]-[hash][extname]`;
            if (/woff|woff2|eot|ttf|otf/i.test(ext)) return `assets/fonts/[name]-[hash][extname]`;
            return `assets/[name]-[hash][extname]`;
          },
          chunkFileNames: "chunks/[name]-[hash].js",
          entryFileNames: "entries/[name]-[hash].js"
        }
      }
    },
    optimizeDeps: {
      include: ["lucide-svelte", "xstate", "@xstate/svelte", "bullmq", "ioredis", "lokijs", "fuse.js", "bits-ui", "@melt-ui/svelte", "zod", "socket.io-client"]
    },
    resolve: {
      alias: {
        // Local fallback for bits-ui to avoid incompatible compiled runtime internals
        // During migration to Svelte 5 we use a local shim that exports simple
        // fallback components. Remove this alias once a Svelte-5-compatible
        // bits-ui release or fork is available.
        "bits-ui": resolve2("./src/lib/vendor/bits-ui-fallback"),
        $lib: resolve2("./src/lib"),
        $components: resolve2("./src/lib/components"),
        $stores: resolve2("./src/lib/stores"),
        $utils: resolve2("./src/lib/utils"),
        $database: resolve2("./src/lib/database"),
        $agents: resolve2("./src/lib/agents"),
        $legal: resolve2("./src/lib/legal")
      }
    },
    worker: {
      rollupOptions: {
        output: { format: "es", entryFileNames: "workers/[name]-[hash].js", chunkFileNames: "workers/chunks/[name]-[hash].js" }
      }
    },
    define: {
      __DEV__: !isProd,
      __PROD__: isProd,
      __VERSION__: JSON.stringify(pkgVersion || "1.0.0"),
      __BUILD_TIME__: JSON.stringify((/* @__PURE__ */ new Date()).toISOString()),
      __VITE_PORT__: availablePort
    }
  };
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiLCAic3JjL2xpYi92aXRlL3ZzY29kZS1lcnJvci1sb2dnZXIudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHN2ZWx0ZWtpdC1mcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcamFtZXNcXFxcRGVza3RvcFxcXFxkZWVkcy13ZWJcXFxcZGVlZHMtd2ViLWFwcFxcXFxzdmVsdGVraXQtZnJvbnRlbmRcXFxcdml0ZS5jb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzL2phbWVzL0Rlc2t0b3AvZGVlZHMtd2ViL2RlZWRzLXdlYi1hcHAvc3ZlbHRla2l0LWZyb250ZW5kL3ZpdGUuY29uZmlnLmpzXCI7Ly8gQHRzLW5vY2hlY2tcclxuaW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSBcIkBzdmVsdGVqcy9raXQvdml0ZVwiO1xyXG5pbXBvcnQgeyBkZWZpbmVDb25maWcgfSBmcm9tIFwidml0ZVwiO1xyXG5pbXBvcnQgVW5vQ1NTIGZyb20gXCJ1bm9jc3Mvdml0ZVwiO1xyXG5pbXBvcnQgeyByZXNvbHZlIH0gZnJvbSBcInBhdGhcIjtcclxuaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xyXG5pbXBvcnQgeyB2c2NvZGVFcnJvckxvZ2dlciB9IGZyb20gJy4vc3JjL2xpYi92aXRlL3ZzY29kZS1lcnJvci1sb2dnZXIuanMnO1xyXG5cclxuLy8gU21hcnQgcG9ydCBkaXNjb3ZlcnlcclxuYXN5bmMgZnVuY3Rpb24gZmluZEF2YWlsYWJsZVBvcnQoc3RhcnRQb3J0LCBtYXhBdHRlbXB0cyA9IDEwKSB7XHJcbiAgY29uc3QgbmV0ID0gYXdhaXQgaW1wb3J0KCduZXQnKTtcclxuICBmb3IgKGxldCBpID0gMDsgaSA8IG1heEF0dGVtcHRzOyBpKyspIHtcclxuICAgIGNvbnN0IHBvcnQgPSBzdGFydFBvcnQgKyBpO1xyXG4gICAgdHJ5IHtcclxuICAgICAgYXdhaXQgbmV3IFByb21pc2UoKHJlc29sdmVQcm9taXNlLCByZWplY3QpID0+IHtcclxuICAgICAgICBjb25zdCBzZXJ2ZXIgPSBuZXQuY3JlYXRlU2VydmVyKCk7XHJcbiAgICAgICAgc2VydmVyLmxpc3Rlbihwb3J0LCAoZXJyKSA9PiAoZXJyID8gcmVqZWN0KGVycikgOiBzZXJ2ZXIuY2xvc2UocmVzb2x2ZVByb21pc2UpKSk7XHJcbiAgICAgICAgc2VydmVyLm9uKCdlcnJvcicsIHJlamVjdCk7XHJcbiAgICAgIH0pO1xyXG4gICAgICByZXR1cm4gcG9ydDtcclxuICAgIH0gY2F0Y2gge31cclxuICB9XHJcbiAgcmV0dXJuIHN0YXJ0UG9ydDtcclxufVxyXG5cclxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKGFzeW5jICh7IG1vZGUgfSkgPT4ge1xyXG4gIGNvbnN0IHByZWZlcnJlZFBvcnQgPSA1MTczO1xyXG4gIGNvbnN0IGF2YWlsYWJsZVBvcnQgPSBhd2FpdCBmaW5kQXZhaWxhYmxlUG9ydChwcmVmZXJyZWRQb3J0KTtcclxuXHJcbiAgY29uc3QgaXNQcm9kID0gbW9kZSA9PT0gJ3Byb2R1Y3Rpb24nO1xyXG5cclxuICAvLyBSZXNvbHZlIHBhY2thZ2UgdmVyc2lvbiBzYWZlbHkgKHN1cHBvcnRzIGRpcmVjdCBub2RlIGltcG9ydCBhbmQgVml0ZSBlbnYpXHJcbiAgbGV0IHBrZ1ZlcnNpb24gPSBwcm9jZXNzLmVudi5ucG1fcGFja2FnZV92ZXJzaW9uO1xyXG4gIHRyeSB7XHJcbiAgICBpZiAoIXBrZ1ZlcnNpb24pIHtcclxuICAgICAgY29uc3QgcGtnSnNvbiA9IEpTT04ucGFyc2UocmVhZEZpbGVTeW5jKG5ldyBVUkwoJy4vcGFja2FnZS5qc29uJywgaW1wb3J0Lm1ldGEudXJsKSwgJ3V0ZjgnKSk7XHJcbiAgICAgIHBrZ1ZlcnNpb24gPSBwa2dKc29uPy52ZXJzaW9uIHx8ICcxLjAuMCc7XHJcbiAgICB9XHJcbiAgfSBjYXRjaCAoZSkge1xyXG4gICAgcGtnVmVyc2lvbiA9IHBrZ1ZlcnNpb24gfHwgJzEuMC4wJztcclxuICB9XHJcblxyXG4gIHJldHVybiB7XHJcbiAgICBwbHVnaW5zOiBbXHJcbiAgICAgIFVub0NTUygpLFxyXG4gICAgICB2c2NvZGVFcnJvckxvZ2dlcih7XHJcbiAgICAgICAgZW5hYmxlZDogIWlzUHJvZCxcclxuICAgICAgICBsb2dGaWxlOiByZXNvbHZlKCcudnNjb2RlL3ZpdGUtZXJyb3JzLmpzb24nKSxcclxuICAgICAgICBtYXhFbnRyaWVzOiA1MDAsXHJcbiAgICAgICAgaW5jbHVkZVdhcm5pbmdzOiB0cnVlLFxyXG4gICAgICAgIGluY2x1ZGVTb3VyY2VNYXBzOiB0cnVlXHJcbiAgICAgIH0pLFxyXG4gICAgICBzdmVsdGVraXQoKSxcclxuICAgICAgLy8gQ3VzdG9tIEdvIHNlcnZlciBpbnRlZ3JhdGlvbiBtaWRkbGV3YXJlXHJcbiAgICAgIHtcclxuICAgICAgICBuYW1lOiAndml0ZS1wbHVnaW4tZ28taW50ZWdyYXRpb24nLFxyXG4gICAgICAgIGNvbmZpZ3VyZVNlcnZlcihzZXJ2ZXIpIHtcclxuICAgICAgICAgIHNlcnZlci5taWRkbGV3YXJlcy51c2UoKHJlcSwgX3JlcywgbmV4dCkgPT4ge1xyXG4gICAgICAgICAgICBpZiAocmVxLnVybD8uc3RhcnRzV2l0aCgnL2FwaS9nby8nKSkge1xyXG4gICAgICAgICAgICAgIHJlcS51cmwgPSByZXEudXJsLnJlcGxhY2UoJy9hcGkvZ28nLCAnJyk7IC8vIFJld3JpdGUgcGF0aCBkeW5hbWljYWxseVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIG5leHQoKTtcclxuICAgICAgICAgIH0pO1xyXG4gICAgICAgIH1cclxuICAgICAgfVxyXG4gICAgXSxcclxuXHJcbiAgICBzZXJ2ZXI6IHtcclxuICAgICAgcG9ydDogYXZhaWxhYmxlUG9ydCxcclxuICAgICAgaG9zdDogJzAuMC4wLjAnLFxyXG4gICAgICBjb3JzOiB0cnVlLFxyXG4gICAgICBzdHJpY3RQb3J0OiBmYWxzZSxcclxuICAgICAgaG1yOiB7IHBvcnQ6IDMxMzEsIGNsaWVudFBvcnQ6IDMxMzEgfSxcclxuICAgICAgZnM6IHsgYWxsb3c6IFsnLi4nLCAnLi4vLi4vJ10gfSxcclxuICAgICAgcHJveHk6IHtcclxuICAgICAgICAvLyBHbyBtaWNyb3NlcnZpY2UgcHJveGllc1xyXG4gICAgICAgICcvYXBpL2dvL2VuaGFuY2VkLXJhZyc6IHsgdGFyZ2V0OiAnaHR0cDovL2xvY2FsaG9zdDo4MDk0JywgY2hhbmdlT3JpZ2luOiB0cnVlLCByZXdyaXRlOiBwYXRoID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL2dvXFwvZW5oYW5jZWQtcmFnLywgJycpIH0sXHJcbiAgICAgICAgJy9hcGkvZ28vdXBsb2FkJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwOTMnLCBjaGFuZ2VPcmlnaW46IHRydWUsIHJld3JpdGU6IHBhdGggPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvZ29cXC91cGxvYWQvLCAnJykgfSxcclxuICAgICAgICAnL2FwaS9nby9jbHVzdGVyJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgyMTMnLCBjaGFuZ2VPcmlnaW46IHRydWUsIHJld3JpdGU6IHBhdGggPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvZ29cXC9jbHVzdGVyLywgJycpIH0sXHJcbiAgICAgICAgJy9hcGkvZ28veHN0YXRlJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgyMTInLCBjaGFuZ2VPcmlnaW46IHRydWUsIHJld3JpdGU6IHBhdGggPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvZ29cXC94c3RhdGUvLCAnJykgfSxcclxuICAgICAgICBcclxuICAgICAgICAvLyBEaXJlY3QgYmFja2VuZCBzZXJ2aWNlIHByb3hpZXMgKGZvciAvYXBpL3YxLyogZW5kcG9pbnRzKVxyXG4gICAgICAgICcvYXBpL3YxL3JhZyc6IHsgdGFyZ2V0OiAnaHR0cDovL2xvY2FsaG9zdDo4MDk0JywgY2hhbmdlT3JpZ2luOiB0cnVlLCByZXdyaXRlOiBwYXRoID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL3YxXFwvcmFnLywgJy9hcGkvcmFnJykgfSxcclxuICAgICAgICAnL2FwaS92MS9haSc6IHsgdGFyZ2V0OiAnaHR0cDovL2xvY2FsaG9zdDo4MDk0JywgY2hhbmdlT3JpZ2luOiB0cnVlLCByZXdyaXRlOiBwYXRoID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL3YxXFwvYWkvLCAnL2FwaS9haScpIH0sXHJcbiAgICAgICAgJy9hcGkvdjEvdXBsb2FkJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwOTMnLCBjaGFuZ2VPcmlnaW46IHRydWUsIHJld3JpdGU6IHBhdGggPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvdjFcXC91cGxvYWQvLCAnL3VwbG9hZCcpIH0sXHJcbiAgICAgICAgJy9hcGkvdjEvdmVjdG9yJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwOTQnLCBjaGFuZ2VPcmlnaW46IHRydWUsIHJld3JpdGU6IHBhdGggPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvdjFcXC92ZWN0b3IvLCAnL2FwaS92ZWN0b3InKSB9LFxyXG4gICAgICAgICcvYXBpL3YxL2NsdXN0ZXInOiB7IHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODIxMycsIGNoYW5nZU9yaWdpbjogdHJ1ZSwgcmV3cml0ZTogcGF0aCA9PiBwYXRoLnJlcGxhY2UoL15cXC9hcGlcXC92MVxcL2NsdXN0ZXIvLCAnL2FwaS9jbHVzdGVyJykgfSxcclxuICAgICAgICBcclxuICAgICAgICAvLyBPbGxhbWEgQVBJIHByb3h5IC0gcHJlc2VydmUgL2FwaSBwYXRoXHJcbiAgICAgICAgJy9hcGkvb2xsYW1hJzogeyB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjExNDM0JywgY2hhbmdlT3JpZ2luOiB0cnVlLCByZXdyaXRlOiBwYXRoID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL29sbGFtYS8sICcvYXBpJykgfSxcclxuICAgICAgICBcclxuICAgICAgICAvLyBPdGhlciBzZXJ2aWNlIHByb3hpZXNcclxuICAgICAgICAnL2FwaS9udmlkaWEtbGxhbWEnOiB7IHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODIyMicsIGNoYW5nZU9yaWdpbjogdHJ1ZSwgcmV3cml0ZTogcGF0aCA9PiBwYXRoLnJlcGxhY2UoL15cXC9hcGlcXC9udmlkaWEtbGxhbWEvLCAnJykgfSxcclxuICAgICAgICAnL2FwaS9uZW80aic6IHsgdGFyZ2V0OiAnaHR0cDovL2xvY2FsaG9zdDo3NDc0JywgY2hhbmdlT3JpZ2luOiB0cnVlLCByZXdyaXRlOiBwYXRoID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL25lbzRqLywgJycpIH1cclxuICAgICAgfVxyXG4gICAgfSxcclxuXHJcbiAgICBwcmV2aWV3OiB7IHBvcnQ6IGF2YWlsYWJsZVBvcnQgKyAxMDAwLCBob3N0OiAnMC4wLjAuMCcsIGNvcnM6IHRydWUsIHN0cmljdFBvcnQ6IGZhbHNlIH0sXHJcblxyXG4gICAgYnVpbGQ6IHtcclxuICAgICAgdGFyZ2V0OiAnZXNuZXh0JyxcclxuICAgICAgbWluaWZ5OiBpc1Byb2QgPyAnZXNidWlsZCcgOiBmYWxzZSxcclxuICAgICAgc291cmNlbWFwOiAhaXNQcm9kLFxyXG4gICAgICBhc3NldHNJbmxpbmVMaW1pdDogNDA5NixcclxuICAgICAgY2h1bmtTaXplV2FybmluZ0xpbWl0OiAxMDAwLFxyXG4gICAgICByb2xsdXBPcHRpb25zOiB7XHJcbiAgICAgICAgb3V0cHV0OiB7XHJcbiAgICAgICAgICBtYW51YWxDaHVua3M6IHtcclxuICAgICAgICAgICAgXCJ1aS1mcmFtZXdvcmtcIjogW1wiYml0cy11aVwiLCBcIkBtZWx0LXVpL3N2ZWx0ZVwiXSxcclxuICAgICAgICAgICAgXCJpY29uc1wiOiBbXCJsdWNpZGUtc3ZlbHRlXCJdLFxyXG4gICAgICAgICAgICBcInN0YXRlLW1hbmFnZW1lbnRcIjogW1wieHN0YXRlXCIsIFwiQHhzdGF0ZS9zdmVsdGVcIiwgXCJzdmVsdGUvc3RvcmVcIl0sXHJcbiAgICAgICAgICAgIFwiY3NzLWVuZ2luZVwiOiBbXCJ1bm9jc3NcIiwgXCJ0YWlsd2luZGNzc1wiLCBcInRhaWx3aW5kLW1lcmdlXCJdLFxyXG4gICAgICAgICAgICBcImFpLXByb2Nlc3NpbmdcIjogW1wiYnVsbG1xXCIsIFwiaW9yZWRpc1wiLCBcInNvY2tldC5pby1jbGllbnRcIl0sXHJcbiAgICAgICAgICAgIFwiY2xpZW50LWRhdGFcIjogW1wibG9raWpzXCIsIFwiZnVzZS5qc1wiXSxcclxuICAgICAgICAgICAgXCJ2YWxpZGF0aW9uXCI6IFtcInpvZFwiLCBcInN2ZWx0ZWtpdC1zdXBlcmZvcm1zXCJdXHJcbiAgICAgICAgICB9LFxyXG4gICAgICAgICAgYXNzZXRGaWxlTmFtZXM6IGFzc2V0SW5mbyA9PiB7XHJcbiAgICAgICAgICAgIGNvbnN0IG5hbWUgPSBhc3NldEluZm8uZmlsZU5hbWUgfHwgKEFycmF5LmlzQXJyYXkoYXNzZXRJbmZvLm5hbWVzKSA/IGFzc2V0SW5mby5uYW1lc1swXSA6ICcnKTtcclxuICAgICAgICAgICAgY29uc3QgZXh0ID0gKG5hbWUuc3BsaXQoJy4nKS5wb3AoKSB8fCAnJykudG9Mb3dlckNhc2UoKTtcclxuICAgICAgICAgICAgaWYgKC9wbmd8anBlP2d8c3ZnfGdpZnx0aWZmfGJtcHxpY28vaS50ZXN0KGV4dCkpIHJldHVybiBgYXNzZXRzL2ltYWdlcy9bbmFtZV0tW2hhc2hdW2V4dG5hbWVdYDtcclxuICAgICAgICAgICAgaWYgKC93b2ZmfHdvZmYyfGVvdHx0dGZ8b3RmL2kudGVzdChleHQpKSByZXR1cm4gYGFzc2V0cy9mb250cy9bbmFtZV0tW2hhc2hdW2V4dG5hbWVdYDtcclxuICAgICAgICAgICAgcmV0dXJuIGBhc3NldHMvW25hbWVdLVtoYXNoXVtleHRuYW1lXWA7XHJcbiAgICAgICAgICB9LFxyXG4gICAgICAgICAgY2h1bmtGaWxlTmFtZXM6ICdjaHVua3MvW25hbWVdLVtoYXNoXS5qcycsXHJcbiAgICAgICAgICBlbnRyeUZpbGVOYW1lczogJ2VudHJpZXMvW25hbWVdLVtoYXNoXS5qcydcclxuICAgICAgICB9XHJcbiAgICAgIH1cclxuICAgIH0sXHJcblxyXG4gICAgb3B0aW1pemVEZXBzOiB7XHJcbiAgICAgIGluY2x1ZGU6IFtcImx1Y2lkZS1zdmVsdGVcIixcInhzdGF0ZVwiLFwiQHhzdGF0ZS9zdmVsdGVcIixcImJ1bGxtcVwiLFwiaW9yZWRpc1wiLFwibG9raWpzXCIsXCJmdXNlLmpzXCIsXCJiaXRzLXVpXCIsXCJAbWVsdC11aS9zdmVsdGVcIixcInpvZFwiLFwic29ja2V0LmlvLWNsaWVudFwiXVxyXG4gICAgfSxcclxuXHJcbiAgICByZXNvbHZlOiB7XHJcbiAgICAgIGFsaWFzOiB7XHJcbiAgLy8gTG9jYWwgZmFsbGJhY2sgZm9yIGJpdHMtdWkgdG8gYXZvaWQgaW5jb21wYXRpYmxlIGNvbXBpbGVkIHJ1bnRpbWUgaW50ZXJuYWxzXHJcbiAgLy8gRHVyaW5nIG1pZ3JhdGlvbiB0byBTdmVsdGUgNSB3ZSB1c2UgYSBsb2NhbCBzaGltIHRoYXQgZXhwb3J0cyBzaW1wbGVcclxuICAvLyBmYWxsYmFjayBjb21wb25lbnRzLiBSZW1vdmUgdGhpcyBhbGlhcyBvbmNlIGEgU3ZlbHRlLTUtY29tcGF0aWJsZVxyXG4gIC8vIGJpdHMtdWkgcmVsZWFzZSBvciBmb3JrIGlzIGF2YWlsYWJsZS5cclxuICAnYml0cy11aSc6IHJlc29sdmUoJy4vc3JjL2xpYi92ZW5kb3IvYml0cy11aS1mYWxsYmFjaycpLFxyXG4gICAgICAgICRsaWI6IHJlc29sdmUoJy4vc3JjL2xpYicpLFxyXG4gICAgICAgICRjb21wb25lbnRzOiByZXNvbHZlKCcuL3NyYy9saWIvY29tcG9uZW50cycpLFxyXG4gICAgICAgICRzdG9yZXM6IHJlc29sdmUoJy4vc3JjL2xpYi9zdG9yZXMnKSxcclxuICAgICAgICAkdXRpbHM6IHJlc29sdmUoJy4vc3JjL2xpYi91dGlscycpLFxyXG4gICAgICAgICRkYXRhYmFzZTogcmVzb2x2ZSgnLi9zcmMvbGliL2RhdGFiYXNlJyksXHJcbiAgICAgICAgJGFnZW50czogcmVzb2x2ZSgnLi9zcmMvbGliL2FnZW50cycpLFxyXG4gICAgICAgICRsZWdhbDogcmVzb2x2ZSgnLi9zcmMvbGliL2xlZ2FsJylcclxuICAgICAgfVxyXG4gICAgfSxcclxuXHJcbiAgICB3b3JrZXI6IHtcclxuICAgICAgcm9sbHVwT3B0aW9uczoge1xyXG4gICAgICAgIG91dHB1dDogeyBmb3JtYXQ6ICdlcycsIGVudHJ5RmlsZU5hbWVzOiAnd29ya2Vycy9bbmFtZV0tW2hhc2hdLmpzJywgY2h1bmtGaWxlTmFtZXM6ICd3b3JrZXJzL2NodW5rcy9bbmFtZV0tW2hhc2hdLmpzJyB9XHJcbiAgICAgIH1cclxuICAgIH0sXHJcblxyXG4gICAgZGVmaW5lOiB7XHJcbiAgICAgIF9fREVWX186ICFpc1Byb2QsXHJcbiAgICAgIF9fUFJPRF9fOiBpc1Byb2QsXHJcbiAgX19WRVJTSU9OX186IEpTT04uc3RyaW5naWZ5KHBrZ1ZlcnNpb24gfHwgJzEuMC4wJyksXHJcbiAgICAgIF9fQlVJTERfVElNRV9fOiBKU09OLnN0cmluZ2lmeShuZXcgRGF0ZSgpLnRvSVNPU3RyaW5nKCkpLFxyXG4gICAgICBfX1ZJVEVfUE9SVF9fOiBhdmFpbGFibGVQb3J0XHJcbiAgICB9XHJcbiAgfTtcclxufSk7XHJcbiIsICJjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfZGlybmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcamFtZXNcXFxcRGVza3RvcFxcXFxkZWVkcy13ZWJcXFxcZGVlZHMtd2ViLWFwcFxcXFxzdmVsdGVraXQtZnJvbnRlbmRcXFxcc3JjXFxcXGxpYlxcXFx2aXRlXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHN2ZWx0ZWtpdC1mcm9udGVuZFxcXFxzcmNcXFxcbGliXFxcXHZpdGVcXFxcdnNjb2RlLWVycm9yLWxvZ2dlci50c1wiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9pbXBvcnRfbWV0YV91cmwgPSBcImZpbGU6Ly8vQzovVXNlcnMvamFtZXMvRGVza3RvcC9kZWVkcy13ZWIvZGVlZHMtd2ViLWFwcC9zdmVsdGVraXQtZnJvbnRlbmQvc3JjL2xpYi92aXRlL3ZzY29kZS1lcnJvci1sb2dnZXIudHNcIjsvLyBAdHMtbm9jaGVja1xuaW1wb3J0IHsgcmVzb2x2ZSwgZGlybmFtZSB9IGZyb20gJ3BhdGgnO1xuaW1wb3J0IHsgd3JpdGVGaWxlU3luYywgZXhpc3RzU3luYywgbWtkaXJTeW5jLCByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbi8qKlxuICogU2ltcGxlIFZTIENvZGUgRXJyb3IgTG9nZ2VyIHBsdWdpbiBmb3IgVml0ZVxuICogLSBXcml0ZXMgYSBKU09OIGxvZyB0byAudnNjb2RlL3ZpdGUtZXJyb3JzLmpzb25cbiAqIC0gTGlzdGVucyBmb3IgZGV2U2VydmVyIHdzICd2aXRlOmVycm9yJyBhbmQgJ3ZpdGU6d2FybmluZydcbiAqIC0gUmVjb3JkcyBidWlsZFN0YXJ0L2J1aWxkRW5kIGV2ZW50c1xuICovXG5leHBvcnQgZnVuY3Rpb24gdnNjb2RlRXJyb3JMb2dnZXIob3B0aW9uczogYW55ID0ge30pIHtcbiAgY29uc3QgY29uZmlnID0ge1xuICAgIGVuYWJsZWQ6IHRydWUsXG4gICAgbG9nRmlsZTogcmVzb2x2ZShwcm9jZXNzLmN3ZCgpLCAnLnZzY29kZS92aXRlLWVycm9ycy5qc29uJyksXG4gICAgbWF4RW50cmllczogNTAwLFxuICAgIGluY2x1ZGVXYXJuaW5nczogdHJ1ZSxcbiAgICBpbmNsdWRlU291cmNlTWFwczogdHJ1ZSxcbiAgICAuLi5vcHRpb25zXG4gIH07XG5cbiAgbGV0IHNlcnZlcjogYW55ID0gdW5kZWZpbmVkO1xuICBsZXQgZXJyb3JMb2c6IGFueSA9IHsgbWV0YWRhdGE6IHsgbGFzdFVwZGF0ZWQ6IG5ldyBEYXRlKCkudG9JU09TdHJpbmcoKSwgdmVyc2lvbjogMSB9LCBlcnJvcnM6IFtdIH07XG5cbiAgZnVuY3Rpb24gbG9hZExvZygpIHtcbiAgICB0cnkge1xuICAgICAgaWYgKGV4aXN0c1N5bmMoY29uZmlnLmxvZ0ZpbGUpKSB7XG4gICAgICAgIGNvbnN0IHJhdyA9IHJlYWRGaWxlU3luYyhjb25maWcubG9nRmlsZSwgJ3V0ZjgnKTtcbiAgICAgICAgY29uc3QgcGFyc2VkID0gSlNPTi5wYXJzZShyYXcpO1xuICAgICAgICAvLyBOb3JtYWxpemUgc2hhcGUgc28gbGF0ZXIgY29kZSBjYW4gc2FmZWx5IHJlYWQgbWV0YWRhdGEgYW5kIGVycm9yc1xuICAgICAgICBlcnJvckxvZyA9IHtcbiAgICAgICAgICBtZXRhZGF0YToge1xuICAgICAgICAgICAgbGFzdFVwZGF0ZWQ6IHBhcnNlZD8ubWV0YWRhdGE/Lmxhc3RVcGRhdGVkIHx8IG5ldyBEYXRlKCkudG9JU09TdHJpbmcoKSxcbiAgICAgICAgICAgIHZlcnNpb246IHBhcnNlZD8ubWV0YWRhdGE/LnZlcnNpb24gfHwgMVxuICAgICAgICAgIH0sXG4gICAgICAgICAgZXJyb3JzOiBBcnJheS5pc0FycmF5KHBhcnNlZD8uZXJyb3JzKSA/IHBhcnNlZC5lcnJvcnMgOiBbXVxuICAgICAgICB9O1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAvLyBpZ25vcmVcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBzYXZlTG9nKCkge1xuICAgIHRyeSB7XG4gICAgICBjb25zdCBkaXIgPSBkaXJuYW1lKGNvbmZpZy5sb2dGaWxlKTtcbiAgICAgIGlmICghZXhpc3RzU3luYyhkaXIpKSBta2RpclN5bmMoZGlyLCB7IHJlY3Vyc2l2ZTogdHJ1ZSB9KTtcbiAgICAgIC8vIEVuc3VyZSBtZXRhZGF0YSBhbmQgZXJyb3JzIGV4aXN0IGJlZm9yZSBtdXRhdGluZ1xuICAgICAgZXJyb3JMb2cubWV0YWRhdGEgPSBlcnJvckxvZy5tZXRhZGF0YSB8fCB7IHZlcnNpb246IDEgfTtcbiAgICAgIGVycm9yTG9nLm1ldGFkYXRhLmxhc3RVcGRhdGVkID0gbmV3IERhdGUoKS50b0lTT1N0cmluZygpO1xuICAgICAgZXJyb3JMb2cuZXJyb3JzID0gQXJyYXkuaXNBcnJheShlcnJvckxvZy5lcnJvcnMpID8gZXJyb3JMb2cuZXJyb3JzIDogW107XG4gICAgICB3cml0ZUZpbGVTeW5jKGNvbmZpZy5sb2dGaWxlLCBKU09OLnN0cmluZ2lmeShlcnJvckxvZywgbnVsbCwgMikpO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAvLyBpZ25vcmVcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBwdXNoRW50cnkoZW50cnk6IGFueSkge1xuICAgIGlmICghY29uZmlnLmVuYWJsZWQpIHJldHVybjtcbiAgICBlcnJvckxvZy5lcnJvcnMgPSBlcnJvckxvZy5lcnJvcnMgfHwgW107XG4gICAgZXJyb3JMb2cuZXJyb3JzLnVuc2hpZnQoZW50cnkpO1xuICAgIGlmIChlcnJvckxvZy5lcnJvcnMubGVuZ3RoID4gY29uZmlnLm1heEVudHJpZXMpIGVycm9yTG9nLmVycm9ycy5sZW5ndGggPSBjb25maWcubWF4RW50cmllcztcbiAgICBzYXZlTG9nKCk7XG4gIH1cblxuICBmdW5jdGlvbiBub3JtYWxpemVWaXRlRXJyb3IoZXJyOiBhbnkpIHtcbiAgICBjb25zdCBlbnRyeTogYW55ID0ge1xuICAgICAgdGltZXN0YW1wOiBuZXcgRGF0ZSgpLnRvSVNPU3RyaW5nKCksXG4gICAgICBsZXZlbDogJ2Vycm9yJyxcbiAgICAgIG1lc3NhZ2U6IGVycj8ubWVzc2FnZSB8fCBlcnI/LnRleHQgfHwgU3RyaW5nKGVyciksXG4gICAgICBzdGFjazogZXJyPy5zdGFjayB8fCBlcnI/LnN0YWNrU3RyLFxuICAgICAgZmlsZTogZXJyPy5pZCB8fCBlcnI/LmxvYz8uZmlsZSB8fCB1bmRlZmluZWQsXG4gICAgICBsaW5lOiBlcnI/LmxvYz8ubGluZSB8fCBlcnI/LmxvYz8ubGluZU51bWJlciB8fCB1bmRlZmluZWQsXG4gICAgICBjb2x1bW46IGVycj8ubG9jPy5jb2x1bW4gfHwgdW5kZWZpbmVkLFxuICAgICAgZnJhbWU6IGVycj8uZnJhbWUgfHwgdW5kZWZpbmVkLFxuICAgICAgcGx1Z2luOiBlcnI/LnBsdWdpbiB8fCB1bmRlZmluZWQsXG4gICAgICBidWlsZFBoYXNlOiAndml0ZSdcbiAgICB9O1xuICAgIHJldHVybiBlbnRyeTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgbmFtZTogJ3ZzY29kZS1lcnJvci1sb2dnZXInLFxuICAgIGNvbmZpZ3VyZVNlcnZlcihzcnY6IGFueSkge1xuICAgICAgc2VydmVyID0gc3J2O1xuICAgICAgbG9hZExvZygpO1xuXG4gICAgICB0cnkge1xuICAgICAgICBzZXJ2ZXIud3Mub24oJ3ZpdGU6ZXJyb3InLCAocGF5bG9hZDogYW55KSA9PiB7XG4gICAgICAgICAgY29uc3QgZSA9IG5vcm1hbGl6ZVZpdGVFcnJvcihwYXlsb2FkLmVyciB8fCBwYXlsb2FkKTtcbiAgICAgICAgICBwdXNoRW50cnkoZSk7XG4gICAgICAgIH0pO1xuXG4gICAgICAgIGlmIChjb25maWcuaW5jbHVkZVdhcm5pbmdzKSB7XG4gICAgICAgICAgc2VydmVyLndzLm9uKCd2aXRlOndhcm5pbmcnLCAocGF5bG9hZDogYW55KSA9PiB7XG4gICAgICAgICAgICBjb25zdCBlID0gbm9ybWFsaXplVml0ZUVycm9yKHBheWxvYWQud2FybiB8fCBwYXlsb2FkKTtcbiAgICAgICAgICAgIGUubGV2ZWwgPSAnd2Fybic7XG4gICAgICAgICAgICBwdXNoRW50cnkoZSk7XG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgLy8gaWdub3JlIHdlYnNvY2tldCBhdHRhY2ggZXJyb3JzXG4gICAgICB9XG4gICAgfSxcblxuICAgIGJ1aWxkU3RhcnQoKSB7XG4gICAgICBwdXNoRW50cnkoeyB0aW1lc3RhbXA6IG5ldyBEYXRlKCkudG9JU09TdHJpbmcoKSwgbGV2ZWw6ICdpbmZvJywgbWVzc2FnZTogJ0J1aWxkIHN0YXJ0ZWQnLCBidWlsZFBoYXNlOiAnYnVpbGQnIH0pO1xuICAgIH0sXG5cbiAgICBidWlsZEVuZChlcnJvcjogYW55KSB7XG4gICAgICBpZiAoZXJyb3IpIHtcbiAgICAgICAgcHVzaEVudHJ5KHsgdGltZXN0YW1wOiBuZXcgRGF0ZSgpLnRvSVNPU3RyaW5nKCksIGxldmVsOiAnZXJyb3InLCBtZXNzYWdlOiBlcnJvci5tZXNzYWdlIHx8IFN0cmluZyhlcnJvciksIHN0YWNrOiBlcnJvci5zdGFjaywgYnVpbGRQaGFzZTogJ2J1aWxkJyB9KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHB1c2hFbnRyeSh7IHRpbWVzdGFtcDogbmV3IERhdGUoKS50b0lTT1N0cmluZygpLCBsZXZlbDogJ2luZm8nLCBtZXNzYWdlOiAnQnVpbGQgY29tcGxldGVkJywgYnVpbGRQaGFzZTogJ2J1aWxkJyB9KTtcbiAgICAgIH1cbiAgICB9XG4gIH07XG59XG5cbmV4cG9ydCBjb25zdCBkZWZhdWx0VlNDb2RlRXJyb3JDb25maWcgPSB7XG4gIGVuYWJsZWQ6IHRydWUsXG4gIGxvZ0ZpbGU6IHJlc29sdmUocHJvY2Vzcy5jd2QoKSwgJy52c2NvZGUvdml0ZS1lcnJvcnMuanNvbicpLFxuICBtYXhFbnRyaWVzOiAxMDAwLFxuICBpbmNsdWRlV2FybmluZ3M6IHRydWUsXG4gIGluY2x1ZGVTb3VyY2VNYXBzOiB0cnVlLFxuICBhdXRvT3BlblByb2JsZW1zOiBmYWxzZSxcbiAgbm90aWZpY2F0aW9uTGV2ZWw6ICdlcnJvcnMtb25seScsXG4gIGludGVncmF0ZVRhc2tzOiB0cnVlLFxuICBnZW5lcmF0ZURpYWdub3N0aWNzOiB0cnVlXG59OyJdLAogICJtYXBwaW5ncyI6ICI7QUFDQSxTQUFTLGlCQUFpQjtBQUMxQixTQUFTLG9CQUFvQjtBQUM3QixPQUFPLFlBQVk7QUFDbkIsU0FBUyxXQUFBQSxnQkFBZTtBQUN4QixTQUFTLGdCQUFBQyxxQkFBb0I7OztBQ0o3QixTQUFTLFNBQVMsZUFBZTtBQUNqQyxTQUFTLGVBQWUsWUFBWSxXQUFXLG9CQUFvQjtBQVE1RCxTQUFTLGtCQUFrQixVQUFlLENBQUMsR0FBRztBQUNuRCxRQUFNLFNBQVM7QUFBQSxJQUNiLFNBQVM7QUFBQSxJQUNULFNBQVMsUUFBUSxRQUFRLElBQUksR0FBRywwQkFBMEI7QUFBQSxJQUMxRCxZQUFZO0FBQUEsSUFDWixpQkFBaUI7QUFBQSxJQUNqQixtQkFBbUI7QUFBQSxJQUNuQixHQUFHO0FBQUEsRUFDTDtBQUVBLE1BQUksU0FBYztBQUNsQixNQUFJLFdBQWdCLEVBQUUsVUFBVSxFQUFFLGNBQWEsb0JBQUksS0FBSyxHQUFFLFlBQVksR0FBRyxTQUFTLEVBQUUsR0FBRyxRQUFRLENBQUMsRUFBRTtBQUVsRyxXQUFTLFVBQVU7QUFDakIsUUFBSTtBQUNGLFVBQUksV0FBVyxPQUFPLE9BQU8sR0FBRztBQUM5QixjQUFNLE1BQU0sYUFBYSxPQUFPLFNBQVMsTUFBTTtBQUMvQyxjQUFNLFNBQVMsS0FBSyxNQUFNLEdBQUc7QUFFN0IsbUJBQVc7QUFBQSxVQUNULFVBQVU7QUFBQSxZQUNSLGFBQWEsUUFBUSxVQUFVLGdCQUFlLG9CQUFJLEtBQUssR0FBRSxZQUFZO0FBQUEsWUFDckUsU0FBUyxRQUFRLFVBQVUsV0FBVztBQUFBLFVBQ3hDO0FBQUEsVUFDQSxRQUFRLE1BQU0sUUFBUSxRQUFRLE1BQU0sSUFBSSxPQUFPLFNBQVMsQ0FBQztBQUFBLFFBQzNEO0FBQUEsTUFDRjtBQUFBLElBQ0YsU0FBUyxHQUFHO0FBQUEsSUFFWjtBQUFBLEVBQ0Y7QUFFQSxXQUFTLFVBQVU7QUFDakIsUUFBSTtBQUNGLFlBQU0sTUFBTSxRQUFRLE9BQU8sT0FBTztBQUNsQyxVQUFJLENBQUMsV0FBVyxHQUFHLEVBQUcsV0FBVSxLQUFLLEVBQUUsV0FBVyxLQUFLLENBQUM7QUFFeEQsZUFBUyxXQUFXLFNBQVMsWUFBWSxFQUFFLFNBQVMsRUFBRTtBQUN0RCxlQUFTLFNBQVMsZUFBYyxvQkFBSSxLQUFLLEdBQUUsWUFBWTtBQUN2RCxlQUFTLFNBQVMsTUFBTSxRQUFRLFNBQVMsTUFBTSxJQUFJLFNBQVMsU0FBUyxDQUFDO0FBQ3RFLG9CQUFjLE9BQU8sU0FBUyxLQUFLLFVBQVUsVUFBVSxNQUFNLENBQUMsQ0FBQztBQUFBLElBQ2pFLFNBQVMsR0FBRztBQUFBLElBRVo7QUFBQSxFQUNGO0FBRUEsV0FBUyxVQUFVLE9BQVk7QUFDN0IsUUFBSSxDQUFDLE9BQU8sUUFBUztBQUNyQixhQUFTLFNBQVMsU0FBUyxVQUFVLENBQUM7QUFDdEMsYUFBUyxPQUFPLFFBQVEsS0FBSztBQUM3QixRQUFJLFNBQVMsT0FBTyxTQUFTLE9BQU8sV0FBWSxVQUFTLE9BQU8sU0FBUyxPQUFPO0FBQ2hGLFlBQVE7QUFBQSxFQUNWO0FBRUEsV0FBUyxtQkFBbUIsS0FBVTtBQUNwQyxVQUFNLFFBQWE7QUFBQSxNQUNqQixZQUFXLG9CQUFJLEtBQUssR0FBRSxZQUFZO0FBQUEsTUFDbEMsT0FBTztBQUFBLE1BQ1AsU0FBUyxLQUFLLFdBQVcsS0FBSyxRQUFRLE9BQU8sR0FBRztBQUFBLE1BQ2hELE9BQU8sS0FBSyxTQUFTLEtBQUs7QUFBQSxNQUMxQixNQUFNLEtBQUssTUFBTSxLQUFLLEtBQUssUUFBUTtBQUFBLE1BQ25DLE1BQU0sS0FBSyxLQUFLLFFBQVEsS0FBSyxLQUFLLGNBQWM7QUFBQSxNQUNoRCxRQUFRLEtBQUssS0FBSyxVQUFVO0FBQUEsTUFDNUIsT0FBTyxLQUFLLFNBQVM7QUFBQSxNQUNyQixRQUFRLEtBQUssVUFBVTtBQUFBLE1BQ3ZCLFlBQVk7QUFBQSxJQUNkO0FBQ0EsV0FBTztBQUFBLEVBQ1Q7QUFFQSxTQUFPO0FBQUEsSUFDTCxNQUFNO0FBQUEsSUFDTixnQkFBZ0IsS0FBVTtBQUN4QixlQUFTO0FBQ1QsY0FBUTtBQUVSLFVBQUk7QUFDRixlQUFPLEdBQUcsR0FBRyxjQUFjLENBQUMsWUFBaUI7QUFDM0MsZ0JBQU0sSUFBSSxtQkFBbUIsUUFBUSxPQUFPLE9BQU87QUFDbkQsb0JBQVUsQ0FBQztBQUFBLFFBQ2IsQ0FBQztBQUVELFlBQUksT0FBTyxpQkFBaUI7QUFDMUIsaUJBQU8sR0FBRyxHQUFHLGdCQUFnQixDQUFDLFlBQWlCO0FBQzdDLGtCQUFNLElBQUksbUJBQW1CLFFBQVEsUUFBUSxPQUFPO0FBQ3BELGNBQUUsUUFBUTtBQUNWLHNCQUFVLENBQUM7QUFBQSxVQUNiLENBQUM7QUFBQSxRQUNIO0FBQUEsTUFDRixTQUFTLEdBQUc7QUFBQSxNQUVaO0FBQUEsSUFDRjtBQUFBLElBRUEsYUFBYTtBQUNYLGdCQUFVLEVBQUUsWUFBVyxvQkFBSSxLQUFLLEdBQUUsWUFBWSxHQUFHLE9BQU8sUUFBUSxTQUFTLGlCQUFpQixZQUFZLFFBQVEsQ0FBQztBQUFBLElBQ2pIO0FBQUEsSUFFQSxTQUFTLE9BQVk7QUFDbkIsVUFBSSxPQUFPO0FBQ1Qsa0JBQVUsRUFBRSxZQUFXLG9CQUFJLEtBQUssR0FBRSxZQUFZLEdBQUcsT0FBTyxTQUFTLFNBQVMsTUFBTSxXQUFXLE9BQU8sS0FBSyxHQUFHLE9BQU8sTUFBTSxPQUFPLFlBQVksUUFBUSxDQUFDO0FBQUEsTUFDckosT0FBTztBQUNMLGtCQUFVLEVBQUUsWUFBVyxvQkFBSSxLQUFLLEdBQUUsWUFBWSxHQUFHLE9BQU8sUUFBUSxTQUFTLG1CQUFtQixZQUFZLFFBQVEsQ0FBQztBQUFBLE1BQ25IO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7QUFDRjtBQUVPLElBQU0sMkJBQTJCO0FBQUEsRUFDdEMsU0FBUztBQUFBLEVBQ1QsU0FBUyxRQUFRLFFBQVEsSUFBSSxHQUFHLDBCQUEwQjtBQUFBLEVBQzFELFlBQVk7QUFBQSxFQUNaLGlCQUFpQjtBQUFBLEVBQ2pCLG1CQUFtQjtBQUFBLEVBQ25CLGtCQUFrQjtBQUFBLEVBQ2xCLG1CQUFtQjtBQUFBLEVBQ25CLGdCQUFnQjtBQUFBLEVBQ2hCLHFCQUFxQjtBQUN2Qjs7O0FEaEl1UCxJQUFNLDJDQUEyQztBQVN4UyxlQUFlLGtCQUFrQixXQUFXLGNBQWMsSUFBSTtBQUM1RCxRQUFNLE1BQU0sTUFBTSxPQUFPLEtBQUs7QUFDOUIsV0FBUyxJQUFJLEdBQUcsSUFBSSxhQUFhLEtBQUs7QUFDcEMsVUFBTSxPQUFPLFlBQVk7QUFDekIsUUFBSTtBQUNGLFlBQU0sSUFBSSxRQUFRLENBQUMsZ0JBQWdCLFdBQVc7QUFDNUMsY0FBTSxTQUFTLElBQUksYUFBYTtBQUNoQyxlQUFPLE9BQU8sTUFBTSxDQUFDLFFBQVMsTUFBTSxPQUFPLEdBQUcsSUFBSSxPQUFPLE1BQU0sY0FBYyxDQUFFO0FBQy9FLGVBQU8sR0FBRyxTQUFTLE1BQU07QUFBQSxNQUMzQixDQUFDO0FBQ0QsYUFBTztBQUFBLElBQ1QsUUFBUTtBQUFBLElBQUM7QUFBQSxFQUNYO0FBQ0EsU0FBTztBQUNUO0FBRUEsSUFBTyxzQkFBUSxhQUFhLE9BQU8sRUFBRSxLQUFLLE1BQU07QUFDOUMsUUFBTSxnQkFBZ0I7QUFDdEIsUUFBTSxnQkFBZ0IsTUFBTSxrQkFBa0IsYUFBYTtBQUUzRCxRQUFNLFNBQVMsU0FBUztBQUd4QixNQUFJLGFBQWEsUUFBUSxJQUFJO0FBQzdCLE1BQUk7QUFDRixRQUFJLENBQUMsWUFBWTtBQUNmLFlBQU0sVUFBVSxLQUFLLE1BQU1DLGNBQWEsSUFBSSxJQUFJLGtCQUFrQix3Q0FBZSxHQUFHLE1BQU0sQ0FBQztBQUMzRixtQkFBYSxTQUFTLFdBQVc7QUFBQSxJQUNuQztBQUFBLEVBQ0YsU0FBUyxHQUFHO0FBQ1YsaUJBQWEsY0FBYztBQUFBLEVBQzdCO0FBRUEsU0FBTztBQUFBLElBQ0wsU0FBUztBQUFBLE1BQ1AsT0FBTztBQUFBLE1BQ1Asa0JBQWtCO0FBQUEsUUFDaEIsU0FBUyxDQUFDO0FBQUEsUUFDVixTQUFTQyxTQUFRLDBCQUEwQjtBQUFBLFFBQzNDLFlBQVk7QUFBQSxRQUNaLGlCQUFpQjtBQUFBLFFBQ2pCLG1CQUFtQjtBQUFBLE1BQ3JCLENBQUM7QUFBQSxNQUNELFVBQVU7QUFBQTtBQUFBLE1BRVY7QUFBQSxRQUNFLE1BQU07QUFBQSxRQUNOLGdCQUFnQixRQUFRO0FBQ3RCLGlCQUFPLFlBQVksSUFBSSxDQUFDLEtBQUssTUFBTSxTQUFTO0FBQzFDLGdCQUFJLElBQUksS0FBSyxXQUFXLFVBQVUsR0FBRztBQUNuQyxrQkFBSSxNQUFNLElBQUksSUFBSSxRQUFRLFdBQVcsRUFBRTtBQUFBLFlBQ3pDO0FBQ0EsaUJBQUs7QUFBQSxVQUNQLENBQUM7QUFBQSxRQUNIO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFBQSxJQUVBLFFBQVE7QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxNQUNaLEtBQUssRUFBRSxNQUFNLE1BQU0sWUFBWSxLQUFLO0FBQUEsTUFDcEMsSUFBSSxFQUFFLE9BQU8sQ0FBQyxNQUFNLFFBQVEsRUFBRTtBQUFBLE1BQzlCLE9BQU87QUFBQTtBQUFBLFFBRUwsd0JBQXdCLEVBQUUsUUFBUSx5QkFBeUIsY0FBYyxNQUFNLFNBQVMsVUFBUSxLQUFLLFFBQVEsNEJBQTRCLEVBQUUsRUFBRTtBQUFBLFFBQzdJLGtCQUFrQixFQUFFLFFBQVEseUJBQXlCLGNBQWMsTUFBTSxTQUFTLFVBQVEsS0FBSyxRQUFRLHNCQUFzQixFQUFFLEVBQUU7QUFBQSxRQUNqSSxtQkFBbUIsRUFBRSxRQUFRLHlCQUF5QixjQUFjLE1BQU0sU0FBUyxVQUFRLEtBQUssUUFBUSx1QkFBdUIsRUFBRSxFQUFFO0FBQUEsUUFDbkksa0JBQWtCLEVBQUUsUUFBUSx5QkFBeUIsY0FBYyxNQUFNLFNBQVMsVUFBUSxLQUFLLFFBQVEsc0JBQXNCLEVBQUUsRUFBRTtBQUFBO0FBQUEsUUFHakksZUFBZSxFQUFFLFFBQVEseUJBQXlCLGNBQWMsTUFBTSxTQUFTLFVBQVEsS0FBSyxRQUFRLG1CQUFtQixVQUFVLEVBQUU7QUFBQSxRQUNuSSxjQUFjLEVBQUUsUUFBUSx5QkFBeUIsY0FBYyxNQUFNLFNBQVMsVUFBUSxLQUFLLFFBQVEsa0JBQWtCLFNBQVMsRUFBRTtBQUFBLFFBQ2hJLGtCQUFrQixFQUFFLFFBQVEseUJBQXlCLGNBQWMsTUFBTSxTQUFTLFVBQVEsS0FBSyxRQUFRLHNCQUFzQixTQUFTLEVBQUU7QUFBQSxRQUN4SSxrQkFBa0IsRUFBRSxRQUFRLHlCQUF5QixjQUFjLE1BQU0sU0FBUyxVQUFRLEtBQUssUUFBUSxzQkFBc0IsYUFBYSxFQUFFO0FBQUEsUUFDNUksbUJBQW1CLEVBQUUsUUFBUSx5QkFBeUIsY0FBYyxNQUFNLFNBQVMsVUFBUSxLQUFLLFFBQVEsdUJBQXVCLGNBQWMsRUFBRTtBQUFBO0FBQUEsUUFHL0ksZUFBZSxFQUFFLFFBQVEsMEJBQTBCLGNBQWMsTUFBTSxTQUFTLFVBQVEsS0FBSyxRQUFRLGtCQUFrQixNQUFNLEVBQUU7QUFBQTtBQUFBLFFBRy9ILHFCQUFxQixFQUFFLFFBQVEseUJBQXlCLGNBQWMsTUFBTSxTQUFTLFVBQVEsS0FBSyxRQUFRLHdCQUF3QixFQUFFLEVBQUU7QUFBQSxRQUN0SSxjQUFjLEVBQUUsUUFBUSx5QkFBeUIsY0FBYyxNQUFNLFNBQVMsVUFBUSxLQUFLLFFBQVEsaUJBQWlCLEVBQUUsRUFBRTtBQUFBLE1BQzFIO0FBQUEsSUFDRjtBQUFBLElBRUEsU0FBUyxFQUFFLE1BQU0sZ0JBQWdCLEtBQU0sTUFBTSxXQUFXLE1BQU0sTUFBTSxZQUFZLE1BQU07QUFBQSxJQUV0RixPQUFPO0FBQUEsTUFDTCxRQUFRO0FBQUEsTUFDUixRQUFRLFNBQVMsWUFBWTtBQUFBLE1BQzdCLFdBQVcsQ0FBQztBQUFBLE1BQ1osbUJBQW1CO0FBQUEsTUFDbkIsdUJBQXVCO0FBQUEsTUFDdkIsZUFBZTtBQUFBLFFBQ2IsUUFBUTtBQUFBLFVBQ04sY0FBYztBQUFBLFlBQ1osZ0JBQWdCLENBQUMsV0FBVyxpQkFBaUI7QUFBQSxZQUM3QyxTQUFTLENBQUMsZUFBZTtBQUFBLFlBQ3pCLG9CQUFvQixDQUFDLFVBQVUsa0JBQWtCLGNBQWM7QUFBQSxZQUMvRCxjQUFjLENBQUMsVUFBVSxlQUFlLGdCQUFnQjtBQUFBLFlBQ3hELGlCQUFpQixDQUFDLFVBQVUsV0FBVyxrQkFBa0I7QUFBQSxZQUN6RCxlQUFlLENBQUMsVUFBVSxTQUFTO0FBQUEsWUFDbkMsY0FBYyxDQUFDLE9BQU8sc0JBQXNCO0FBQUEsVUFDOUM7QUFBQSxVQUNBLGdCQUFnQixlQUFhO0FBQzNCLGtCQUFNLE9BQU8sVUFBVSxhQUFhLE1BQU0sUUFBUSxVQUFVLEtBQUssSUFBSSxVQUFVLE1BQU0sQ0FBQyxJQUFJO0FBQzFGLGtCQUFNLE9BQU8sS0FBSyxNQUFNLEdBQUcsRUFBRSxJQUFJLEtBQUssSUFBSSxZQUFZO0FBQ3RELGdCQUFJLGtDQUFrQyxLQUFLLEdBQUcsRUFBRyxRQUFPO0FBQ3hELGdCQUFJLDBCQUEwQixLQUFLLEdBQUcsRUFBRyxRQUFPO0FBQ2hELG1CQUFPO0FBQUEsVUFDVDtBQUFBLFVBQ0EsZ0JBQWdCO0FBQUEsVUFDaEIsZ0JBQWdCO0FBQUEsUUFDbEI7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUFBLElBRUEsY0FBYztBQUFBLE1BQ1osU0FBUyxDQUFDLGlCQUFnQixVQUFTLGtCQUFpQixVQUFTLFdBQVUsVUFBUyxXQUFVLFdBQVUsbUJBQWtCLE9BQU0sa0JBQWtCO0FBQUEsSUFDaEo7QUFBQSxJQUVBLFNBQVM7QUFBQSxNQUNQLE9BQU87QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFFBS1gsV0FBV0EsU0FBUSxtQ0FBbUM7QUFBQSxRQUNoRCxNQUFNQSxTQUFRLFdBQVc7QUFBQSxRQUN6QixhQUFhQSxTQUFRLHNCQUFzQjtBQUFBLFFBQzNDLFNBQVNBLFNBQVEsa0JBQWtCO0FBQUEsUUFDbkMsUUFBUUEsU0FBUSxpQkFBaUI7QUFBQSxRQUNqQyxXQUFXQSxTQUFRLG9CQUFvQjtBQUFBLFFBQ3ZDLFNBQVNBLFNBQVEsa0JBQWtCO0FBQUEsUUFDbkMsUUFBUUEsU0FBUSxpQkFBaUI7QUFBQSxNQUNuQztBQUFBLElBQ0Y7QUFBQSxJQUVBLFFBQVE7QUFBQSxNQUNOLGVBQWU7QUFBQSxRQUNiLFFBQVEsRUFBRSxRQUFRLE1BQU0sZ0JBQWdCLDRCQUE0QixnQkFBZ0Isa0NBQWtDO0FBQUEsTUFDeEg7QUFBQSxJQUNGO0FBQUEsSUFFQSxRQUFRO0FBQUEsTUFDTixTQUFTLENBQUM7QUFBQSxNQUNWLFVBQVU7QUFBQSxNQUNkLGFBQWEsS0FBSyxVQUFVLGNBQWMsT0FBTztBQUFBLE1BQzdDLGdCQUFnQixLQUFLLFdBQVUsb0JBQUksS0FBSyxHQUFFLFlBQVksQ0FBQztBQUFBLE1BQ3ZELGVBQWU7QUFBQSxJQUNqQjtBQUFBLEVBQ0Y7QUFDRixDQUFDOyIsCiAgIm5hbWVzIjogWyJyZXNvbHZlIiwgInJlYWRGaWxlU3luYyIsICJyZWFkRmlsZVN5bmMiLCAicmVzb2x2ZSJdCn0K
