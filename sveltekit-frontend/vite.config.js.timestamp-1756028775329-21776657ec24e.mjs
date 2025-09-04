// vite.config.js
import { sveltekit } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/node_modules/vite/dist/node/index.js";
import path from "path";
var vite_config_default = defineConfig({
  plugins: [sveltekit()],
  // Enhanced logging configuration
  logLevel: "info",
  // 'error' | 'warn' | 'info' | 'silent'
  resolve: {
    alias: {
      $lib: path.resolve("./src/lib"),
      $components: path.resolve("./src/lib/components"),
      $services: path.resolve("./src/lib/services"),
      $types: path.resolve("./src/lib/types")
    }
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
    include: ["fabric", "pdf-lib", "socket.io-client"]
  },
  // Build logging
  build: {
    reportCompressedSize: true,
    chunkSizeWarningLimit: 1e3
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHN2ZWx0ZWtpdC1mcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiQzpcXFxcVXNlcnNcXFxcamFtZXNcXFxcRGVza3RvcFxcXFxkZWVkcy13ZWJcXFxcZGVlZHMtd2ViLWFwcFxcXFxzdmVsdGVraXQtZnJvbnRlbmRcXFxcdml0ZS5jb25maWcuanNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0M6L1VzZXJzL2phbWVzL0Rlc2t0b3AvZGVlZHMtd2ViL2RlZWRzLXdlYi1hcHAvc3ZlbHRla2l0LWZyb250ZW5kL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcbmltcG9ydCB7IGRlZmluZUNvbmZpZyB9IGZyb20gJ3ZpdGUnO1xuaW1wb3J0IHBhdGggZnJvbSAncGF0aCc7XG5cbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZyh7XG5cdHBsdWdpbnM6IFtzdmVsdGVraXQoKV0sXG5cdFxuXHQvLyBFbmhhbmNlZCBsb2dnaW5nIGNvbmZpZ3VyYXRpb25cblx0bG9nTGV2ZWw6ICdpbmZvJywgLy8gJ2Vycm9yJyB8ICd3YXJuJyB8ICdpbmZvJyB8ICdzaWxlbnQnXG5cdFxuXHRyZXNvbHZlOiB7XG5cdFx0YWxpYXM6IHtcblx0XHRcdCRsaWI6IHBhdGgucmVzb2x2ZSgnLi9zcmMvbGliJyksXG5cdFx0XHQkY29tcG9uZW50czogcGF0aC5yZXNvbHZlKCcuL3NyYy9saWIvY29tcG9uZW50cycpLFxuXHRcdFx0JHNlcnZpY2VzOiBwYXRoLnJlc29sdmUoJy4vc3JjL2xpYi9zZXJ2aWNlcycpLFxuXHRcdFx0JHR5cGVzOiBwYXRoLnJlc29sdmUoJy4vc3JjL2xpYi90eXBlcycpXG5cdFx0fVxuXHR9LFxuXHRzZXJ2ZXI6IHtcblx0XHRwb3J0OiA1MTczLFxuXHRcdHN0cmljdFBvcnQ6IGZhbHNlLFxuXHRcdGhvc3Q6ICcwLjAuMC4wJyxcblx0XHRobXI6IHsgXG5cdFx0XHRwb3J0OiAyNDY3OCwgXG5cdFx0XHRjbGllbnRQb3J0OiAyNDY3OCBcblx0XHR9LFxuXHRcdC8vIEVuaGFuY2VkIHByb3h5IGxvZ2dpbmdcblx0XHRwcm94eToge1xuXHRcdFx0Jy9oZWFsdGgnOiB7XG5cdFx0XHRcdHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4MCcsXG5cdFx0XHRcdGNoYW5nZU9yaWdpbjogdHJ1ZSxcblx0XHRcdFx0Y29uZmlndXJlOiAocHJveHksIG9wdGlvbnMpID0+IHtcblx0XHRcdFx0XHRwcm94eS5vbigncHJveHlSZXEnLCAocHJveHlSZXEsIHJlcSwgcmVzKSA9PiB7XG5cdFx0XHRcdFx0XHRjb25zb2xlLmxvZyhgW1BST1hZXSAke3JlcS5tZXRob2R9ICR7cmVxLnVybH0gLT4gJHtvcHRpb25zLnRhcmdldH1gKTtcblx0XHRcdFx0XHR9KTtcblx0XHRcdFx0XHRwcm94eS5vbigncHJveHlSZXMnLCAocHJveHlSZXMsIHJlcSwgcmVzKSA9PiB7XG5cdFx0XHRcdFx0XHRjb25zb2xlLmxvZyhgW1BST1hZXSAke3JlcS5tZXRob2R9ICR7cmVxLnVybH0gPC0gJHtwcm94eVJlcy5zdGF0dXNDb2RlfWApO1xuXHRcdFx0XHRcdH0pO1xuXHRcdFx0XHR9XG5cdFx0XHR9LFxuXHRcdFx0Jy9hcGkvdjEnOiB7XG5cdFx0XHRcdHRhcmdldDogJ2h0dHA6Ly9sb2NhbGhvc3Q6ODA4MCcsXG5cdFx0XHRcdGNoYW5nZU9yaWdpbjogdHJ1ZSxcblx0XHRcdFx0Y29uZmlndXJlOiAocHJveHksIG9wdGlvbnMpID0+IHtcblx0XHRcdFx0XHRwcm94eS5vbigncHJveHlSZXEnLCAocHJveHlSZXEsIHJlcSwgcmVzKSA9PiB7XG5cdFx0XHRcdFx0XHRjb25zb2xlLmxvZyhgW1BST1hZXSAke3JlcS5tZXRob2R9ICR7cmVxLnVybH0gLT4gJHtvcHRpb25zLnRhcmdldH1gKTtcblx0XHRcdFx0XHR9KTtcblx0XHRcdFx0XHRwcm94eS5vbigncHJveHlSZXMnLCAocHJveHlSZXMsIHJlcSwgcmVzKSA9PiB7XG5cdFx0XHRcdFx0XHRjb25zb2xlLmxvZyhgW1BST1hZXSAke3JlcS5tZXRob2R9ICR7cmVxLnVybH0gPC0gJHtwcm94eVJlcy5zdGF0dXNDb2RlfWApO1xuXHRcdFx0XHRcdH0pO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXHR9LFxuXHRvcHRpbWl6ZURlcHM6IHtcblx0XHRpbmNsdWRlOiBbJ2ZhYnJpYycsICdwZGYtbGliJywgJ3NvY2tldC5pby1jbGllbnQnXVxuXHR9LFxuXHRcblx0Ly8gQnVpbGQgbG9nZ2luZ1xuXHRidWlsZDoge1xuXHRcdHJlcG9ydENvbXByZXNzZWRTaXplOiB0cnVlLFxuXHRcdGNodW5rU2l6ZVdhcm5pbmdMaW1pdDogMTAwMFxuXHR9XG59KTsiXSwKICAibWFwcGluZ3MiOiAiO0FBQW1ZLFNBQVMsaUJBQWlCO0FBQzdaLFNBQVMsb0JBQW9CO0FBQzdCLE9BQU8sVUFBVTtBQUVqQixJQUFPLHNCQUFRLGFBQWE7QUFBQSxFQUMzQixTQUFTLENBQUMsVUFBVSxDQUFDO0FBQUE7QUFBQSxFQUdyQixVQUFVO0FBQUE7QUFBQSxFQUVWLFNBQVM7QUFBQSxJQUNSLE9BQU87QUFBQSxNQUNOLE1BQU0sS0FBSyxRQUFRLFdBQVc7QUFBQSxNQUM5QixhQUFhLEtBQUssUUFBUSxzQkFBc0I7QUFBQSxNQUNoRCxXQUFXLEtBQUssUUFBUSxvQkFBb0I7QUFBQSxNQUM1QyxRQUFRLEtBQUssUUFBUSxpQkFBaUI7QUFBQSxJQUN2QztBQUFBLEVBQ0Q7QUFBQSxFQUNBLFFBQVE7QUFBQSxJQUNQLE1BQU07QUFBQSxJQUNOLFlBQVk7QUFBQSxJQUNaLE1BQU07QUFBQSxJQUNOLEtBQUs7QUFBQSxNQUNKLE1BQU07QUFBQSxNQUNOLFlBQVk7QUFBQSxJQUNiO0FBQUE7QUFBQSxJQUVBLE9BQU87QUFBQSxNQUNOLFdBQVc7QUFBQSxRQUNWLFFBQVE7QUFBQSxRQUNSLGNBQWM7QUFBQSxRQUNkLFdBQVcsQ0FBQyxPQUFPLFlBQVk7QUFDOUIsZ0JBQU0sR0FBRyxZQUFZLENBQUMsVUFBVSxLQUFLLFFBQVE7QUFDNUMsb0JBQVEsSUFBSSxXQUFXLElBQUksTUFBTSxJQUFJLElBQUksR0FBRyxPQUFPLFFBQVEsTUFBTSxFQUFFO0FBQUEsVUFDcEUsQ0FBQztBQUNELGdCQUFNLEdBQUcsWUFBWSxDQUFDLFVBQVUsS0FBSyxRQUFRO0FBQzVDLG9CQUFRLElBQUksV0FBVyxJQUFJLE1BQU0sSUFBSSxJQUFJLEdBQUcsT0FBTyxTQUFTLFVBQVUsRUFBRTtBQUFBLFVBQ3pFLENBQUM7QUFBQSxRQUNGO0FBQUEsTUFDRDtBQUFBLE1BQ0EsV0FBVztBQUFBLFFBQ1YsUUFBUTtBQUFBLFFBQ1IsY0FBYztBQUFBLFFBQ2QsV0FBVyxDQUFDLE9BQU8sWUFBWTtBQUM5QixnQkFBTSxHQUFHLFlBQVksQ0FBQyxVQUFVLEtBQUssUUFBUTtBQUM1QyxvQkFBUSxJQUFJLFdBQVcsSUFBSSxNQUFNLElBQUksSUFBSSxHQUFHLE9BQU8sUUFBUSxNQUFNLEVBQUU7QUFBQSxVQUNwRSxDQUFDO0FBQ0QsZ0JBQU0sR0FBRyxZQUFZLENBQUMsVUFBVSxLQUFLLFFBQVE7QUFDNUMsb0JBQVEsSUFBSSxXQUFXLElBQUksTUFBTSxJQUFJLElBQUksR0FBRyxPQUFPLFNBQVMsVUFBVSxFQUFFO0FBQUEsVUFDekUsQ0FBQztBQUFBLFFBQ0Y7QUFBQSxNQUNEO0FBQUEsSUFDRDtBQUFBLEVBQ0Q7QUFBQSxFQUNBLGNBQWM7QUFBQSxJQUNiLFNBQVMsQ0FBQyxVQUFVLFdBQVcsa0JBQWtCO0FBQUEsRUFDbEQ7QUFBQTtBQUFBLEVBR0EsT0FBTztBQUFBLElBQ04sc0JBQXNCO0FBQUEsSUFDdEIsdUJBQXVCO0FBQUEsRUFDeEI7QUFDRCxDQUFDOyIsCiAgIm5hbWVzIjogW10KfQo=
