// vite.config.js
import { sveltekit } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/vite/dist/node/index.js";
import UnoCSS from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@unocss/vite/dist/index.mjs";
var vite_config_default = defineConfig({
  plugins: [
    UnoCSS(),
    sveltekit(),
    // Custom plugin for Go server integration
    {
      name: "vite-plugin-go-integration",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith("/api/go/")) {
            req.url = req.url.replace("/api/go", "");
          }
          next();
        });
      }
    }
  ],
  server: {
    port: 5173,
    host: true,
    hmr: {
      port: 5174,
      overlay: true
    },
    // Proxy configuration for backend services
    proxy: {
      "/api/go": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, ""),
        configure: (proxy, options) => {
          proxy.on("error", (err, req, res) => {
            console.log("proxy error", err);
          });
          proxy.on("proxyReq", (proxyReq, req, res) => {
            console.log("Sending Request to Go server:", req.method, req.url);
          });
        }
      },
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
        changeOrigin: true
      },
      "/api/redis": {
        target: "http://localhost:6379",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/redis/, "")
      },
      "/api/qdrant": {
        target: "http://localhost:6333",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/qdrant/, "")
      }
    }
  },
  preview: {
    port: 4173,
    host: true,
    proxy: {
      "/api/go": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, "")
      }
    }
  },
  css: {
    postcss: "./postcss.config.js"
  },
  // Enhanced optimization for production builds
  optimizeDeps: {
    include: [
      "lucide-svelte",
      "xstate",
      "@xstate/svelte",
      "bullmq",
      "ioredis",
      "lokijs",
      "fuse.js",
      "bits-ui",
      "@melt-ui/svelte",
      "zod",
      "socket.io-client"
    ],
    exclude: ["@sveltejs/kit"],
    esbuildOptions: {
      // Use multiple CPU cores for faster builds
      target: "esnext",
      platform: "browser",
      // Enable optimizations
      minify: process.env.NODE_ENV === "production",
      treeShaking: true,
      splitting: true
    }
  },
  build: {
    // Enable CSS code splitting for better performance
    cssCodeSplit: true,
    // Increase chunk size warning limit for AI models
    chunkSizeWarningLimit: 1e3,
    // Use Terser for better minification in production
    minify: process.env.NODE_ENV === "production" ? "terser" : "esbuild",
    terserOptions: {
      compress: {
        drop_console: process.env.NODE_ENV === "production",
        drop_debugger: true
      }
    },
    // Rollup configuration for optimal chunking
    rollupOptions: {
      output: {
        manualChunks: {
          // UI frameworks
          "ui-framework": ["bits-ui", "@melt-ui/svelte"],
          // CSS and styling
          "css-engine": ["unocss", "tailwindcss", "tailwind-merge"],
          // Icons
          "icons": ["lucide-svelte"],
          // State management
          "state-management": ["xstate", "@xstate/svelte", "svelte/store"],
          // AI and processing
          "ai-processing": ["bullmq", "ioredis", "socket.io-client"],
          // Client-side data
          "client-data": ["lokijs", "fuse.js"],
          // Validation
          "validation": ["zod", "sveltekit-superforms"]
        },
        // Asset file naming for cache busting
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split(".");
          const ext = info[info.length - 1];
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return `assets/images/[name]-[hash][extname]`;
          } else if (/woff|woff2|eot|ttf|otf/i.test(ext)) {
            return `assets/fonts/[name]-[hash][extname]`;
          }
          return `assets/[name]-[hash][extname]`;
        },
        chunkFileNames: "chunks/[name]-[hash].js",
        entryFileNames: "entries/[name]-[hash].js"
      },
      // External dependencies (if using CDN)
      external: [],
      // Plugins for additional optimizations
      plugins: []
    },
    // Source maps for debugging
    sourcemap: process.env.NODE_ENV === "development",
    // Report compressed size
    reportCompressedSize: true,
    // Target modern browsers for better performance
    target: "esnext",
    // Asset inlining threshold
    assetsInlineLimit: 4096
  },
  // Worker configuration for Web Workers
  worker: {
    format: "es",
    rollupOptions: {
      output: {
        entryFileNames: "workers/[name]-[hash].js"
      }
    }
  },
  // Environment variables
  define: {
    "__APP_VERSION__": JSON.stringify(process.env.npm_package_version),
    "__BUILD_TIME__": JSON.stringify((/* @__PURE__ */ new Date()).toISOString()),
    "__GO_SERVER_URL__": JSON.stringify(process.env.VITE_GO_SERVER_URL || "http://localhost:8080"),
    "__REDIS_URL__": JSON.stringify(process.env.VITE_REDIS_URL || "localhost:6379"),
    "__USE_GPU__": JSON.stringify(process.env.VITE_USE_GPU !== "false")
  },
  // Performance optimizations
  esbuild: {
    // Use Go's esbuild for faster builds
    logLevel: "info",
    logLimit: 10,
    legalComments: "none"
  },
  // JSON handling optimization
  json: {
    namedExports: true,
    stringify: false
  },
  // Resolve configuration
  resolve: {
    alias: {
      "$lib": "/src/lib",
      "$components": "/src/lib/components",
      "$stores": "/src/lib/stores",
      "$machines": "/src/lib/machines",
      "$utils": "/src/lib/utils",
      "$types": "/src/lib/types"
    }
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHN2ZWx0ZWtpdC1mcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcamFtZXNcXFxcRGVza3RvcFxcXFxkZWVkcy13ZWJcXFxcZGVlZHMtd2ViLWFwcFxcXFxzdmVsdGVraXQtZnJvbnRlbmRcXFxcdml0ZS5jb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzL2phbWVzL0Rlc2t0b3AvZGVlZHMtd2ViL2RlZWRzLXdlYi1hcHAvc3ZlbHRla2l0LWZyb250ZW5kL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSBcIkBzdmVsdGVqcy9raXQvdml0ZVwiO1xyXG5pbXBvcnQgeyBkZWZpbmVDb25maWcgfSBmcm9tIFwidml0ZVwiO1xyXG5pbXBvcnQgVW5vQ1NTIGZyb20gXCJAdW5vY3NzL3ZpdGVcIjtcclxuXHJcbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZyh7XHJcbiAgcGx1Z2luczogW1xyXG4gICAgVW5vQ1NTKCksXHJcbiAgICBzdmVsdGVraXQoKSxcclxuICAgIC8vIEN1c3RvbSBwbHVnaW4gZm9yIEdvIHNlcnZlciBpbnRlZ3JhdGlvblxyXG4gICAge1xyXG4gICAgICBuYW1lOiAndml0ZS1wbHVnaW4tZ28taW50ZWdyYXRpb24nLFxyXG4gICAgICBjb25maWd1cmVTZXJ2ZXIoc2VydmVyKSB7XHJcbiAgICAgICAgLy8gUHJveHkgQVBJIHJlcXVlc3RzIHRvIEdvIEdQVSBzZXJ2ZXJcclxuICAgICAgICBzZXJ2ZXIubWlkZGxld2FyZXMudXNlKChyZXEsIHJlcywgbmV4dCkgPT4ge1xyXG4gICAgICAgICAgaWYgKHJlcS51cmw/LnN0YXJ0c1dpdGgoJy9hcGkvZ28vJykpIHtcclxuICAgICAgICAgICAgLy8gUmV3cml0ZSB0byBHbyBzZXJ2ZXJcclxuICAgICAgICAgICAgcmVxLnVybCA9IHJlcS51cmwucmVwbGFjZSgnL2FwaS9nbycsICcnKTtcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIG5leHQoKTtcclxuICAgICAgICB9KTtcclxuICAgICAgfVxyXG4gICAgfVxyXG4gIF0sXHJcbiAgXHJcbiAgc2VydmVyOiB7XHJcbiAgICBwb3J0OiA1MTczLFxyXG4gICAgaG9zdDogdHJ1ZSxcclxuICAgIGhtcjoge1xyXG4gICAgICBwb3J0OiA1MTc0LFxyXG4gICAgICBvdmVybGF5OiB0cnVlXHJcbiAgICB9LFxyXG4gICAgLy8gUHJveHkgY29uZmlndXJhdGlvbiBmb3IgYmFja2VuZCBzZXJ2aWNlc1xyXG4gICAgcHJveHk6IHtcclxuICAgICAgJy9hcGkvZ28nOiB7XHJcbiAgICAgICAgdGFyZ2V0OiAnaHR0cDovL2xvY2FsaG9zdDo4MDgwJyxcclxuICAgICAgICBjaGFuZ2VPcmlnaW46IHRydWUsXHJcbiAgICAgICAgcmV3cml0ZTogKHBhdGgpID0+IHBhdGgucmVwbGFjZSgvXlxcL2FwaVxcL2dvLywgJycpLFxyXG4gICAgICAgIGNvbmZpZ3VyZTogKHByb3h5LCBvcHRpb25zKSA9PiB7XHJcbiAgICAgICAgICBwcm94eS5vbignZXJyb3InLCAoZXJyLCByZXEsIHJlcykgPT4ge1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZygncHJveHkgZXJyb3InLCBlcnIpO1xyXG4gICAgICAgICAgfSk7XHJcbiAgICAgICAgICBwcm94eS5vbigncHJveHlSZXEnLCAocHJveHlSZXEsIHJlcSwgcmVzKSA9PiB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdTZW5kaW5nIFJlcXVlc3QgdG8gR28gc2VydmVyOicsIHJlcS5tZXRob2QsIHJlcS51cmwpO1xyXG4gICAgICAgICAgfSk7XHJcbiAgICAgICAgfVxyXG4gICAgICB9LFxyXG4gICAgICAnL3dzJzoge1xyXG4gICAgICAgIHRhcmdldDogJ3dzOi8vbG9jYWxob3N0OjgwODAnLFxyXG4gICAgICAgIHdzOiB0cnVlLFxyXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZVxyXG4gICAgICB9LFxyXG4gICAgICAnL2FwaS9yZWRpcyc6IHtcclxuICAgICAgICB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjYzNzknLFxyXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcclxuICAgICAgICByZXdyaXRlOiAocGF0aCkgPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvcmVkaXMvLCAnJylcclxuICAgICAgfSxcclxuICAgICAgJy9hcGkvcWRyYW50Jzoge1xyXG4gICAgICAgIHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6NjMzMycsXHJcbiAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxyXG4gICAgICAgIHJld3JpdGU6IChwYXRoKSA9PiBwYXRoLnJlcGxhY2UoL15cXC9hcGlcXC9xZHJhbnQvLCAnJylcclxuICAgICAgfVxyXG4gICAgfVxyXG4gIH0sXHJcbiAgXHJcbiAgcHJldmlldzoge1xyXG4gICAgcG9ydDogNDE3MyxcclxuICAgIGhvc3Q6IHRydWUsXHJcbiAgICBwcm94eToge1xyXG4gICAgICAnL2FwaS9nbyc6IHtcclxuICAgICAgICB0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwODAnLFxyXG4gICAgICAgIGNoYW5nZU9yaWdpbjogdHJ1ZSxcclxuICAgICAgICByZXdyaXRlOiAocGF0aCkgPT4gcGF0aC5yZXBsYWNlKC9eXFwvYXBpXFwvZ28vLCAnJylcclxuICAgICAgfVxyXG4gICAgfVxyXG4gIH0sXHJcbiAgXHJcbiAgY3NzOiB7XHJcbiAgICBwb3N0Y3NzOiBcIi4vcG9zdGNzcy5jb25maWcuanNcIixcclxuICB9LFxyXG4gIFxyXG4gIC8vIEVuaGFuY2VkIG9wdGltaXphdGlvbiBmb3IgcHJvZHVjdGlvbiBidWlsZHNcclxuICBvcHRpbWl6ZURlcHM6IHtcclxuICAgIGluY2x1ZGU6IFtcclxuICAgICAgXCJsdWNpZGUtc3ZlbHRlXCIsXHJcbiAgICAgIFwieHN0YXRlXCIsXHJcbiAgICAgIFwiQHhzdGF0ZS9zdmVsdGVcIixcclxuICAgICAgXCJidWxsbXFcIixcclxuICAgICAgXCJpb3JlZGlzXCIsXHJcbiAgICAgIFwibG9raWpzXCIsXHJcbiAgICAgIFwiZnVzZS5qc1wiLFxyXG4gICAgICBcImJpdHMtdWlcIixcclxuICAgICAgXCJAbWVsdC11aS9zdmVsdGVcIixcclxuICAgICAgXCJ6b2RcIixcclxuICAgICAgXCJzb2NrZXQuaW8tY2xpZW50XCJcclxuICAgIF0sXHJcbiAgICBleGNsdWRlOiBbJ0BzdmVsdGVqcy9raXQnXSxcclxuICAgIGVzYnVpbGRPcHRpb25zOiB7XHJcbiAgICAgIC8vIFVzZSBtdWx0aXBsZSBDUFUgY29yZXMgZm9yIGZhc3RlciBidWlsZHNcclxuICAgICAgdGFyZ2V0OiAnZXNuZXh0JyxcclxuICAgICAgcGxhdGZvcm06ICdicm93c2VyJyxcclxuICAgICAgLy8gRW5hYmxlIG9wdGltaXphdGlvbnNcclxuICAgICAgbWluaWZ5OiBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ3Byb2R1Y3Rpb24nLFxyXG4gICAgICB0cmVlU2hha2luZzogdHJ1ZSxcclxuICAgICAgc3BsaXR0aW5nOiB0cnVlXHJcbiAgICB9XHJcbiAgfSxcclxuICBcclxuICBidWlsZDoge1xyXG4gICAgLy8gRW5hYmxlIENTUyBjb2RlIHNwbGl0dGluZyBmb3IgYmV0dGVyIHBlcmZvcm1hbmNlXHJcbiAgICBjc3NDb2RlU3BsaXQ6IHRydWUsXHJcbiAgICBcclxuICAgIC8vIEluY3JlYXNlIGNodW5rIHNpemUgd2FybmluZyBsaW1pdCBmb3IgQUkgbW9kZWxzXHJcbiAgICBjaHVua1NpemVXYXJuaW5nTGltaXQ6IDEwMDAsXHJcbiAgICBcclxuICAgIC8vIFVzZSBUZXJzZXIgZm9yIGJldHRlciBtaW5pZmljYXRpb24gaW4gcHJvZHVjdGlvblxyXG4gICAgbWluaWZ5OiBwcm9jZXNzLmVudi5OT0RFX0VOViA9PT0gJ3Byb2R1Y3Rpb24nID8gJ3RlcnNlcicgOiAnZXNidWlsZCcsXHJcbiAgICBcclxuICAgIHRlcnNlck9wdGlvbnM6IHtcclxuICAgICAgY29tcHJlc3M6IHtcclxuICAgICAgICBkcm9wX2NvbnNvbGU6IHByb2Nlc3MuZW52Lk5PREVfRU5WID09PSAncHJvZHVjdGlvbicsXHJcbiAgICAgICAgZHJvcF9kZWJ1Z2dlcjogdHJ1ZVxyXG4gICAgICB9XHJcbiAgICB9LFxyXG4gICAgXHJcbiAgICAvLyBSb2xsdXAgY29uZmlndXJhdGlvbiBmb3Igb3B0aW1hbCBjaHVua2luZ1xyXG4gICAgcm9sbHVwT3B0aW9uczoge1xyXG4gICAgICBvdXRwdXQ6IHtcclxuICAgICAgICBtYW51YWxDaHVua3M6IHtcclxuICAgICAgICAgIC8vIFVJIGZyYW1ld29ya3NcclxuICAgICAgICAgIFwidWktZnJhbWV3b3JrXCI6IFtcImJpdHMtdWlcIiwgXCJAbWVsdC11aS9zdmVsdGVcIl0sXHJcbiAgICAgICAgICBcclxuICAgICAgICAgIC8vIENTUyBhbmQgc3R5bGluZ1xyXG4gICAgICAgICAgXCJjc3MtZW5naW5lXCI6IFtcInVub2Nzc1wiLCBcInRhaWx3aW5kY3NzXCIsIFwidGFpbHdpbmQtbWVyZ2VcIl0sXHJcbiAgICAgICAgICBcclxuICAgICAgICAgIC8vIEljb25zXHJcbiAgICAgICAgICBcImljb25zXCI6IFtcImx1Y2lkZS1zdmVsdGVcIl0sXHJcbiAgICAgICAgICBcclxuICAgICAgICAgIC8vIFN0YXRlIG1hbmFnZW1lbnRcclxuICAgICAgICAgIFwic3RhdGUtbWFuYWdlbWVudFwiOiBbXCJ4c3RhdGVcIiwgXCJAeHN0YXRlL3N2ZWx0ZVwiLCBcInN2ZWx0ZS9zdG9yZVwiXSxcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgLy8gQUkgYW5kIHByb2Nlc3NpbmdcclxuICAgICAgICAgIFwiYWktcHJvY2Vzc2luZ1wiOiBbXCJidWxsbXFcIiwgXCJpb3JlZGlzXCIsIFwic29ja2V0LmlvLWNsaWVudFwiXSxcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgLy8gQ2xpZW50LXNpZGUgZGF0YVxyXG4gICAgICAgICAgXCJjbGllbnQtZGF0YVwiOiBbXCJsb2tpanNcIiwgXCJmdXNlLmpzXCJdLFxyXG4gICAgICAgICAgXHJcbiAgICAgICAgICAvLyBWYWxpZGF0aW9uXHJcbiAgICAgICAgICBcInZhbGlkYXRpb25cIjogW1wiem9kXCIsIFwic3ZlbHRla2l0LXN1cGVyZm9ybXNcIl1cclxuICAgICAgICB9LFxyXG4gICAgICAgIFxyXG4gICAgICAgIC8vIEFzc2V0IGZpbGUgbmFtaW5nIGZvciBjYWNoZSBidXN0aW5nXHJcbiAgICAgICAgYXNzZXRGaWxlTmFtZXM6IChhc3NldEluZm8pID0+IHtcclxuICAgICAgICAgIGNvbnN0IGluZm8gPSBhc3NldEluZm8ubmFtZS5zcGxpdCgnLicpO1xyXG4gICAgICAgICAgY29uc3QgZXh0ID0gaW5mb1tpbmZvLmxlbmd0aCAtIDFdO1xyXG4gICAgICAgICAgaWYgKC9wbmd8anBlP2d8c3ZnfGdpZnx0aWZmfGJtcHxpY28vaS50ZXN0KGV4dCkpIHtcclxuICAgICAgICAgICAgcmV0dXJuIGBhc3NldHMvaW1hZ2VzL1tuYW1lXS1baGFzaF1bZXh0bmFtZV1gO1xyXG4gICAgICAgICAgfSBlbHNlIGlmICgvd29mZnx3b2ZmMnxlb3R8dHRmfG90Zi9pLnRlc3QoZXh0KSkge1xyXG4gICAgICAgICAgICByZXR1cm4gYGFzc2V0cy9mb250cy9bbmFtZV0tW2hhc2hdW2V4dG5hbWVdYDtcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIHJldHVybiBgYXNzZXRzL1tuYW1lXS1baGFzaF1bZXh0bmFtZV1gO1xyXG4gICAgICAgIH0sXHJcbiAgICAgICAgXHJcbiAgICAgICAgY2h1bmtGaWxlTmFtZXM6ICdjaHVua3MvW25hbWVdLVtoYXNoXS5qcycsXHJcbiAgICAgICAgZW50cnlGaWxlTmFtZXM6ICdlbnRyaWVzL1tuYW1lXS1baGFzaF0uanMnXHJcbiAgICAgIH0sXHJcbiAgICAgIFxyXG4gICAgICAvLyBFeHRlcm5hbCBkZXBlbmRlbmNpZXMgKGlmIHVzaW5nIENETilcclxuICAgICAgZXh0ZXJuYWw6IFtdLFxyXG4gICAgICBcclxuICAgICAgLy8gUGx1Z2lucyBmb3IgYWRkaXRpb25hbCBvcHRpbWl6YXRpb25zXHJcbiAgICAgIHBsdWdpbnM6IFtdXHJcbiAgICB9LFxyXG4gICAgXHJcbiAgICAvLyBTb3VyY2UgbWFwcyBmb3IgZGVidWdnaW5nXHJcbiAgICBzb3VyY2VtYXA6IHByb2Nlc3MuZW52Lk5PREVfRU5WID09PSAnZGV2ZWxvcG1lbnQnLFxyXG4gICAgXHJcbiAgICAvLyBSZXBvcnQgY29tcHJlc3NlZCBzaXplXHJcbiAgICByZXBvcnRDb21wcmVzc2VkU2l6ZTogdHJ1ZSxcclxuICAgIFxyXG4gICAgLy8gVGFyZ2V0IG1vZGVybiBicm93c2VycyBmb3IgYmV0dGVyIHBlcmZvcm1hbmNlXHJcbiAgICB0YXJnZXQ6ICdlc25leHQnLFxyXG4gICAgXHJcbiAgICAvLyBBc3NldCBpbmxpbmluZyB0aHJlc2hvbGRcclxuICAgIGFzc2V0c0lubGluZUxpbWl0OiA0MDk2XHJcbiAgfSxcclxuICBcclxuICAvLyBXb3JrZXIgY29uZmlndXJhdGlvbiBmb3IgV2ViIFdvcmtlcnNcclxuICB3b3JrZXI6IHtcclxuICAgIGZvcm1hdDogJ2VzJyxcclxuICAgIHJvbGx1cE9wdGlvbnM6IHtcclxuICAgICAgb3V0cHV0OiB7XHJcbiAgICAgICAgZW50cnlGaWxlTmFtZXM6ICd3b3JrZXJzL1tuYW1lXS1baGFzaF0uanMnXHJcbiAgICAgIH1cclxuICAgIH1cclxuICB9LFxyXG4gIFxyXG4gIC8vIEVudmlyb25tZW50IHZhcmlhYmxlc1xyXG4gIGRlZmluZToge1xyXG4gICAgJ19fQVBQX1ZFUlNJT05fXyc6IEpTT04uc3RyaW5naWZ5KHByb2Nlc3MuZW52Lm5wbV9wYWNrYWdlX3ZlcnNpb24pLFxyXG4gICAgJ19fQlVJTERfVElNRV9fJzogSlNPTi5zdHJpbmdpZnkobmV3IERhdGUoKS50b0lTT1N0cmluZygpKSxcclxuICAgICdfX0dPX1NFUlZFUl9VUkxfXyc6IEpTT04uc3RyaW5naWZ5KHByb2Nlc3MuZW52LlZJVEVfR09fU0VSVkVSX1VSTCB8fCAnaHR0cDovL2xvY2FsaG9zdDo4MDgwJyksXHJcbiAgICAnX19SRURJU19VUkxfXyc6IEpTT04uc3RyaW5naWZ5KHByb2Nlc3MuZW52LlZJVEVfUkVESVNfVVJMIHx8ICdsb2NhbGhvc3Q6NjM3OScpLFxyXG4gICAgJ19fVVNFX0dQVV9fJzogSlNPTi5zdHJpbmdpZnkocHJvY2Vzcy5lbnYuVklURV9VU0VfR1BVICE9PSAnZmFsc2UnKVxyXG4gIH0sXHJcbiAgXHJcbiAgLy8gUGVyZm9ybWFuY2Ugb3B0aW1pemF0aW9uc1xyXG4gIGVzYnVpbGQ6IHtcclxuICAgIC8vIFVzZSBHbydzIGVzYnVpbGQgZm9yIGZhc3RlciBidWlsZHNcclxuICAgIGxvZ0xldmVsOiAnaW5mbycsXHJcbiAgICBsb2dMaW1pdDogMTAsXHJcbiAgICBsZWdhbENvbW1lbnRzOiAnbm9uZSdcclxuICB9LFxyXG4gIFxyXG4gIC8vIEpTT04gaGFuZGxpbmcgb3B0aW1pemF0aW9uXHJcbiAganNvbjoge1xyXG4gICAgbmFtZWRFeHBvcnRzOiB0cnVlLFxyXG4gICAgc3RyaW5naWZ5OiBmYWxzZVxyXG4gIH0sXHJcbiAgXHJcbiAgLy8gUmVzb2x2ZSBjb25maWd1cmF0aW9uXHJcbiAgcmVzb2x2ZToge1xyXG4gICAgYWxpYXM6IHtcclxuICAgICAgJyRsaWInOiAnL3NyYy9saWInLFxyXG4gICAgICAnJGNvbXBvbmVudHMnOiAnL3NyYy9saWIvY29tcG9uZW50cycsXHJcbiAgICAgICckc3RvcmVzJzogJy9zcmMvbGliL3N0b3JlcycsXHJcbiAgICAgICckbWFjaGluZXMnOiAnL3NyYy9saWIvbWFjaGluZXMnLFxyXG4gICAgICAnJHV0aWxzJzogJy9zcmMvbGliL3V0aWxzJyxcclxuICAgICAgJyR0eXBlcyc6ICcvc3JjL2xpYi90eXBlcydcclxuICAgIH1cclxuICB9XHJcbn0pO1xyXG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQW1ZLFNBQVMsaUJBQWlCO0FBQzdaLFNBQVMsb0JBQW9CO0FBQzdCLE9BQU8sWUFBWTtBQUVuQixJQUFPLHNCQUFRLGFBQWE7QUFBQSxFQUMxQixTQUFTO0FBQUEsSUFDUCxPQUFPO0FBQUEsSUFDUCxVQUFVO0FBQUE7QUFBQSxJQUVWO0FBQUEsTUFDRSxNQUFNO0FBQUEsTUFDTixnQkFBZ0IsUUFBUTtBQUV0QixlQUFPLFlBQVksSUFBSSxDQUFDLEtBQUssS0FBSyxTQUFTO0FBQ3pDLGNBQUksSUFBSSxLQUFLLFdBQVcsVUFBVSxHQUFHO0FBRW5DLGdCQUFJLE1BQU0sSUFBSSxJQUFJLFFBQVEsV0FBVyxFQUFFO0FBQUEsVUFDekM7QUFDQSxlQUFLO0FBQUEsUUFDUCxDQUFDO0FBQUEsTUFDSDtBQUFBLElBQ0Y7QUFBQSxFQUNGO0FBQUEsRUFFQSxRQUFRO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixLQUFLO0FBQUEsTUFDSCxNQUFNO0FBQUEsTUFDTixTQUFTO0FBQUEsSUFDWDtBQUFBO0FBQUEsSUFFQSxPQUFPO0FBQUEsTUFDTCxXQUFXO0FBQUEsUUFDVCxRQUFRO0FBQUEsUUFDUixjQUFjO0FBQUEsUUFDZCxTQUFTLENBQUMsU0FBUyxLQUFLLFFBQVEsY0FBYyxFQUFFO0FBQUEsUUFDaEQsV0FBVyxDQUFDLE9BQU8sWUFBWTtBQUM3QixnQkFBTSxHQUFHLFNBQVMsQ0FBQyxLQUFLLEtBQUssUUFBUTtBQUNuQyxvQkFBUSxJQUFJLGVBQWUsR0FBRztBQUFBLFVBQ2hDLENBQUM7QUFDRCxnQkFBTSxHQUFHLFlBQVksQ0FBQyxVQUFVLEtBQUssUUFBUTtBQUMzQyxv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
