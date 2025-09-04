// vite.config.js
import { sveltekit } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/vite/dist/node/index.js";
import UnoCSS from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@unocss/vite/dist/index.mjs";
import { nodePolyfills } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/vite-plugin-node-polyfills/dist/index.js";
import path from "path";
var vite_config_default = defineConfig({
  plugins: [
    sveltekit(),
    UnoCSS(),
    nodePolyfills({
      // Enable polyfills for Node.js globals and modules
      include: ["process", "buffer", "util", "stream", "events"],
      globals: {
        Buffer: true,
        global: true,
        process: true
      }
    })
  ],
  // Enhanced logging configuration
  logLevel: "info",
  // 'error' | 'warn' | 'info' | 'silent'
  resolve: {
    alias: {
      $lib: path.resolve("./src/lib"),
      $components: path.resolve("./src/lib/components"),
      $services: path.resolve("./src/lib/services"),
      $types: path.resolve("./src/lib/types"),
      // Force fabric to use the browser-specific build
      "fabric": path.resolve("./node_modules/fabric/dist/fabric.js")
    }
  },
  // Define global constants for browser compatibility
  define: {
    global: "globalThis",
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV || "development"),
    __DEV__: JSON.stringify(process.env.NODE_ENV !== "production")
  },
  server: {
    port: 5173,
    strictPort: false,
    host: "0.0.0.0",
    hmr: {
      port: 24678,
      clientPort: 24678
    },
    // Enhanced proxy logging
    proxy: {
      "/health": {
        target: "http://localhost:8080",
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on("proxyReq", (proxyReq, req, res) => {
            console.log(`[PROXY] ${req.method} ${req.url} -> ${options.target}`);
          });
          proxy.on("proxyRes", (proxyRes, req, res) => {
            console.log(`[PROXY] ${req.method} ${req.url} <- ${proxyRes.statusCode}`);
          });
        }
      },
      "/api/v1": {
        target: "http://localhost:8080",
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on("proxyReq", (proxyReq, req, res) => {
            console.log(`[PROXY] ${req.method} ${req.url} -> ${options.target}`);
          });
          proxy.on("proxyRes", (proxyRes, req, res) => {
            console.log(`[PROXY] ${req.method} ${req.url} <- ${proxyRes.statusCode}`);
          });
        }
      }
    }
  },
  optimizeDeps: {
    include: [
      "socket.io-client",
      // Bits UI and Melt UI dependencies
      "bits-ui",
      "melt",
      // Browser polyfills for Node.js modules
      "fuse.js",
      "zod",
      "class-variance-authority",
      "clsx",
      "tailwind-merge",
      // Vector/AI dependencies
      "@xenova/transformers"
    ],
    exclude: [
      "@tauri-apps/api",
      // Tauri should not be optimized
      "pdf-lib",
      // Problematic module
      "@langchain/core",
      // Missing exports
      "@langchain/community",
      "canvas"
      // Native module issues (fabric.js now properly configured)
    ]
  },
  ssr: {
    noExternal: ["bits-ui", "melt"],
    external: ["fabric", "canvas"]
    // Exclude problematic canvas modules from SSR
  },
  // Enhanced build configuration for browser compatibility
  build: {
    target: ["es2020", "chrome80", "firefox78", "safari14"],
    modulePreload: { polyfill: true },
    rollupOptions: {
      output: {
        // Separate chunks for better caching
        manualChunks: {
          "bits-ui": ["bits-ui"],
          "melt-ui": ["melt"],
          "search": ["fuse.js"],
          "vector": ["@xenova/transformers"],
          "ai": ["@langchain/core", "@langchain/community"],
          "utils": ["zod", "clsx", "tailwind-merge", "class-variance-authority"]
        }
      }
    },
    reportCompressedSize: true,
    chunkSizeWarningLimit: 1e3
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHN2ZWx0ZWtpdC1mcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcamFtZXNcXFxcRGVza3RvcFxcXFxkZWVkcy13ZWJcXFxcZGVlZHMtd2ViLWFwcFxcXFxzdmVsdGVraXQtZnJvbnRlbmRcXFxcdml0ZS5jb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzL2phbWVzL0Rlc2t0b3AvZGVlZHMtd2ViL2RlZWRzLXdlYi1hcHAvc3ZlbHRla2l0LWZyb250ZW5kL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcbmltcG9ydCB7IGRlZmluZUNvbmZpZyB9IGZyb20gJ3ZpdGUnO1xuaW1wb3J0IFVub0NTUyBmcm9tICdAdW5vY3NzL3ZpdGUnO1xuaW1wb3J0IHsgbm9kZVBvbHlmaWxscyB9IGZyb20gJ3ZpdGUtcGx1Z2luLW5vZGUtcG9seWZpbGxzJztcbmltcG9ydCBwYXRoIGZyb20gJ3BhdGgnO1xuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xuXHRwbHVnaW5zOiBbXG5cdFx0c3ZlbHRla2l0KCksIFxuXHRcdFVub0NTUygpLFxuXHRcdG5vZGVQb2x5ZmlsbHMoe1xuXHRcdFx0Ly8gRW5hYmxlIHBvbHlmaWxscyBmb3IgTm9kZS5qcyBnbG9iYWxzIGFuZCBtb2R1bGVzXG5cdFx0XHRpbmNsdWRlOiBbJ3Byb2Nlc3MnLCAnYnVmZmVyJywgJ3V0aWwnLCAnc3RyZWFtJywgJ2V2ZW50cyddLFxuXHRcdFx0Z2xvYmFsczoge1xuXHRcdFx0XHRCdWZmZXI6IHRydWUsXG5cdFx0XHRcdGdsb2JhbDogdHJ1ZSxcblx0XHRcdFx0cHJvY2VzczogdHJ1ZSxcblx0XHRcdH0sXG5cdFx0fSlcblx0XSxcblx0XG5cdC8vIEVuaGFuY2VkIGxvZ2dpbmcgY29uZmlndXJhdGlvblxuXHRsb2dMZXZlbDogJ2luZm8nLCAvLyAnZXJyb3InIHwgJ3dhcm4nIHwgJ2luZm8nIHwgJ3NpbGVudCdcblx0XG5cdHJlc29sdmU6IHtcblx0XHRhbGlhczoge1xuXHRcdFx0JGxpYjogcGF0aC5yZXNvbHZlKCcuL3NyYy9saWInKSxcblx0XHRcdCRjb21wb25lbnRzOiBwYXRoLnJlc29sdmUoJy4vc3JjL2xpYi9jb21wb25lbnRzJyksXG5cdFx0XHQkc2VydmljZXM6IHBhdGgucmVzb2x2ZSgnLi9zcmMvbGliL3NlcnZpY2VzJyksXG5cdFx0XHQkdHlwZXM6IHBhdGgucmVzb2x2ZSgnLi9zcmMvbGliL3R5cGVzJyksXG5cdFx0XHQvLyBGb3JjZSBmYWJyaWMgdG8gdXNlIHRoZSBicm93c2VyLXNwZWNpZmljIGJ1aWxkXG5cdFx0XHQnZmFicmljJzogcGF0aC5yZXNvbHZlKCcuL25vZGVfbW9kdWxlcy9mYWJyaWMvZGlzdC9mYWJyaWMuanMnKVxuXHRcdH1cblx0fSxcblx0XG5cdC8vIERlZmluZSBnbG9iYWwgY29uc3RhbnRzIGZvciBicm93c2VyIGNvbXBhdGliaWxpdHlcblx0ZGVmaW5lOiB7XG5cdFx0Z2xvYmFsOiAnZ2xvYmFsVGhpcycsXG5cdFx0J3Byb2Nlc3MuZW52Lk5PREVfRU5WJzogSlNPTi5zdHJpbmdpZnkocHJvY2Vzcy5lbnYuTk9ERV9FTlYgfHwgJ2RldmVsb3BtZW50JyksXG5cdFx0X19ERVZfXzogSlNPTi5zdHJpbmdpZnkocHJvY2Vzcy5lbnYuTk9ERV9FTlYgIT09ICdwcm9kdWN0aW9uJylcblx0fSxcblx0c2VydmVyOiB7XG5cdFx0cG9ydDogNTE3Myxcblx0XHRzdHJpY3RQb3J0OiBmYWxzZSxcblx0XHRob3N0OiAnMC4wLjAuMCcsXG5cdFx0aG1yOiB7IFxuXHRcdFx0cG9ydDogMjQ2NzgsIFxuXHRcdFx0Y2xpZW50UG9ydDogMjQ2NzggXG5cdFx0fSxcblx0XHQvLyBFbmhhbmNlZCBwcm94eSBsb2dnaW5nXG5cdFx0cHJveHk6IHtcblx0XHRcdCcvaGVhbHRoJzoge1xuXHRcdFx0XHR0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwODAnLFxuXHRcdFx0XHRjaGFuZ2VPcmlnaW46IHRydWUsXG5cdFx0XHRcdGNvbmZpZ3VyZTogKHByb3h5LCBvcHRpb25zKSA9PiB7XG5cdFx0XHRcdFx0cHJveHkub24oJ3Byb3h5UmVxJywgKHByb3h5UmVxLCByZXEsIHJlcykgPT4ge1xuXHRcdFx0XHRcdFx0Y29uc29sZS5sb2coYFtQUk9YWV0gJHtyZXEubWV0aG9kfSAke3JlcS51cmx9IC0+ICR7b3B0aW9ucy50YXJnZXR9YCk7XG5cdFx0XHRcdFx0fSk7XG5cdFx0XHRcdFx0cHJveHkub24oJ3Byb3h5UmVzJywgKHByb3h5UmVzLCByZXEsIHJlcykgPT4ge1xuXHRcdFx0XHRcdFx0Y29uc29sZS5sb2coYFtQUk9YWV0gJHtyZXEubWV0aG9kfSAke3JlcS51cmx9IDwtICR7cHJveHlSZXMuc3RhdHVzQ29kZX1gKTtcblx0XHRcdFx0XHR9KTtcblx0XHRcdFx0fVxuXHRcdFx0fSxcblx0XHRcdCcvYXBpL3YxJzoge1xuXHRcdFx0XHR0YXJnZXQ6ICdodHRwOi8vbG9jYWxob3N0OjgwODAnLFxuXHRcdFx0XHRjaGFuZ2VPcmlnaW46IHRydWUsXG5cdFx0XHRcdGNvbmZpZ3VyZTogKHByb3h5LCBvcHRpb25zKSA9PiB7XG5cdFx0XHRcdFx0cHJveHkub24oJ3Byb3h5UmVxJywgKHByb3h5UmVxLCByZXEsIHJlcykgPT4ge1xuXHRcdFx0XHRcdFx0Y29uc29sZS5sb2coYFtQUk9YWV0gJHtyZXEubWV0aG9kfSAke3JlcS51cmx9IC0+ICR7b3B0aW9ucy50YXJnZXR9YCk7XG5cdFx0XHRcdFx0fSk7XG5cdFx0XHRcdFx0cHJveHkub24oJ3Byb3h5UmVzJywgKHByb3h5UmVzLCByZXEsIHJlcykgPT4ge1xuXHRcdFx0XHRcdFx0Y29uc29sZS5sb2coYFtQUk9YWV0gJHtyZXEubWV0aG9kfSAke3JlcS51cmx9IDwtICR7cHJveHlSZXMuc3RhdHVzQ29kZX1gKTtcblx0XHRcdFx0XHR9KTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblx0fSxcblx0b3B0aW1pemVEZXBzOiB7XG5cdFx0aW5jbHVkZTogW1xuXHRcdFx0J3NvY2tldC5pby1jbGllbnQnLFxuXHRcdFx0Ly8gQml0cyBVSSBhbmQgTWVsdCBVSSBkZXBlbmRlbmNpZXNcblx0XHRcdCdiaXRzLXVpJyxcblx0XHRcdCdtZWx0Jyxcblx0XHRcdC8vIEJyb3dzZXIgcG9seWZpbGxzIGZvciBOb2RlLmpzIG1vZHVsZXNcblx0XHRcdCdmdXNlLmpzJyxcblx0XHRcdCd6b2QnLFxuXHRcdFx0J2NsYXNzLXZhcmlhbmNlLWF1dGhvcml0eScsXG5cdFx0XHQnY2xzeCcsXG5cdFx0XHQndGFpbHdpbmQtbWVyZ2UnLFxuXHRcdFx0Ly8gVmVjdG9yL0FJIGRlcGVuZGVuY2llc1xuXHRcdFx0J0B4ZW5vdmEvdHJhbnNmb3JtZXJzJ1xuXHRcdF0sXG5cdFx0ZXhjbHVkZTogW1xuXHRcdFx0J0B0YXVyaS1hcHBzL2FwaScsIC8vIFRhdXJpIHNob3VsZCBub3QgYmUgb3B0aW1pemVkXG5cdFx0XHQncGRmLWxpYicsIC8vIFByb2JsZW1hdGljIG1vZHVsZVxuXHRcdFx0J0BsYW5nY2hhaW4vY29yZScsIC8vIE1pc3NpbmcgZXhwb3J0c1xuXHRcdFx0J0BsYW5nY2hhaW4vY29tbXVuaXR5Jyxcblx0XHRcdCdjYW52YXMnIC8vIE5hdGl2ZSBtb2R1bGUgaXNzdWVzIChmYWJyaWMuanMgbm93IHByb3Blcmx5IGNvbmZpZ3VyZWQpXG5cdFx0XVxuXHR9LFxuXHRcblx0c3NyOiB7XG5cdFx0bm9FeHRlcm5hbDogWydiaXRzLXVpJywgJ21lbHQnXSxcblx0XHRleHRlcm5hbDogWydmYWJyaWMnLCAnY2FudmFzJ10gLy8gRXhjbHVkZSBwcm9ibGVtYXRpYyBjYW52YXMgbW9kdWxlcyBmcm9tIFNTUlxuXHR9LFxuXHRcblx0Ly8gRW5oYW5jZWQgYnVpbGQgY29uZmlndXJhdGlvbiBmb3IgYnJvd3NlciBjb21wYXRpYmlsaXR5XG5cdGJ1aWxkOiB7XG5cdFx0dGFyZ2V0OiBbJ2VzMjAyMCcsICdjaHJvbWU4MCcsICdmaXJlZm94NzgnLCAnc2FmYXJpMTQnXSxcblx0XHRtb2R1bGVQcmVsb2FkOiB7IHBvbHlmaWxsOiB0cnVlIH0sXG5cdFx0cm9sbHVwT3B0aW9uczoge1xuXHRcdFx0b3V0cHV0OiB7XG5cdFx0XHRcdC8vIFNlcGFyYXRlIGNodW5rcyBmb3IgYmV0dGVyIGNhY2hpbmdcblx0XHRcdFx0bWFudWFsQ2h1bmtzOiB7XG5cdFx0XHRcdFx0J2JpdHMtdWknOiBbJ2JpdHMtdWknXSxcblx0XHRcdFx0XHQnbWVsdC11aSc6IFsnbWVsdCddLFxuXHRcdFx0XHRcdCdzZWFyY2gnOiBbJ2Z1c2UuanMnXSxcblx0XHRcdFx0XHQndmVjdG9yJzogWydAeGVub3ZhL3RyYW5zZm9ybWVycyddLFxuXHRcdFx0XHRcdCdhaSc6IFsnQGxhbmdjaGFpbi9jb3JlJywgJ0BsYW5nY2hhaW4vY29tbXVuaXR5J10sXG5cdFx0XHRcdFx0J3V0aWxzJzogWyd6b2QnLCAnY2xzeCcsICd0YWlsd2luZC1tZXJnZScsICdjbGFzcy12YXJpYW5jZS1hdXRob3JpdHknXVxuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXHRcdCxcblx0XHRyZXBvcnRDb21wcmVzc2VkU2l6ZTogdHJ1ZSxcblx0XHRjaHVua1NpemVXYXJuaW5nTGltaXQ6IDEwMDBcblx0fVxufSk7Il0sCiAgIm1hcHBpbmdzIjogIjtBQUFtWSxTQUFTLGlCQUFpQjtBQUM3WixTQUFTLG9CQUFvQjtBQUM3QixPQUFPLFlBQVk7QUFDbkIsU0FBUyxxQkFBcUI7QUFDOUIsT0FBTyxVQUFVO0FB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
