// vite.config.js
import { sveltekit } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/node_modules/vite/dist/node/index.js";
var vite_config_default = defineConfig({
  plugins: [sveltekit()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    hmr: {
      clientPort: 5173
    }
  },
  optimizeDeps: {
    exclude: ["@grpc/grpc-js", "@grpc/proto-loader"]
  },
  build: {
    target: "esnext",
    minify: "esbuild",
    rollupOptions: {
      external: ["@grpc/grpc-js", "@grpc/proto-loader"]
    }
  },
  define: {
    global: "globalThis"
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXGFwcHNcXFxcd2ViXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXGFwcHNcXFxcd2ViXFxcXHZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9DOi9Vc2Vycy9qYW1lcy9EZXNrdG9wL2RlZWRzLXdlYi9kZWVkcy13ZWItYXBwL2FwcHMvd2ViL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcbmltcG9ydCB7IGRlZmluZUNvbmZpZyB9IGZyb20gJ3ZpdGUnO1xuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xuXHRwbHVnaW5zOiBbc3ZlbHRla2l0KCldLFxuXHRzZXJ2ZXI6IHtcblx0XHRob3N0OiB0cnVlLFxuXHRcdHBvcnQ6IDUxNzMsXG5cdFx0c3RyaWN0UG9ydDogdHJ1ZSxcblx0XHRobXI6IHtcblx0XHRcdGNsaWVudFBvcnQ6IDUxNzNcblx0XHR9XG5cdH0sXG5cdG9wdGltaXplRGVwczoge1xuXHRcdGV4Y2x1ZGU6IFsnQGdycGMvZ3JwYy1qcycsICdAZ3JwYy9wcm90by1sb2FkZXInXVxuXHR9LFxuXHRidWlsZDoge1xuXHRcdHRhcmdldDogJ2VzbmV4dCcsXG5cdFx0bWluaWZ5OiAnZXNidWlsZCcsXG5cdFx0cm9sbHVwT3B0aW9uczoge1xuXHRcdFx0ZXh0ZXJuYWw6IFsnQGdycGMvZ3JwYy1qcycsICdAZ3JwYy9wcm90by1sb2FkZXInXVxuXHRcdH1cblx0fSxcblx0ZGVmaW5lOiB7XG5cdFx0Z2xvYmFsOiAnZ2xvYmFsVGhpcydcblx0fVxufSk7XG4iXSwKICAibWFwcGluZ3MiOiAiO0FBQXVXLFNBQVMsaUJBQWlCO0FBQ2pZLFNBQVMsb0JBQW9CO0FBRTdCLElBQU8sc0JBQVEsYUFBYTtBQUFBLEVBQzNCLFNBQVMsQ0FBQyxVQUFVLENBQUM7QUFBQSxFQUNyQixRQUFRO0FBQUEsSUFDUCxNQUFNO0FBQUEsSUFDTixNQUFNO0FBQUEsSUFDTixZQUFZO0FBQUEsSUFDWixLQUFLO0FBQUEsTUFDSixZQUFZO0FBQUEsSUFDYjtBQUFBLEVBQ0Q7QUFBQSxFQUNBLGNBQWM7QUFBQSxJQUNiLFNBQVMsQ0FBQyxpQkFBaUIsb0JBQW9CO0FBQUEsRUFDaEQ7QUFBQSxFQUNBLE9BQU87QUFBQSxJQUNOLFFBQVE7QUFBQSxJQUNSLFFBQVE7QUFBQSxJQUNSLGVBQWU7QUFBQSxNQUNkLFVBQVUsQ0FBQyxpQkFBaUIsb0JBQW9CO0FBQUEsSUFDakQ7QUFBQSxFQUNEO0FBQUEsRUFDQSxRQUFRO0FBQUEsSUFDUCxRQUFRO0FBQUEsRUFDVDtBQUNELENBQUM7IiwKICAibmFtZXMiOiBbXQp9Cg==
