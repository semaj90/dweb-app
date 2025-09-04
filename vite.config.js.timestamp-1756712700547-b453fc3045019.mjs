// vite.config.js
import { sveltekit } from "file:///c:/Users/james/Desktop/deeds-web/deeds-web-app/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///c:/Users/james/Desktop/deeds-web/deeds-web-app/node_modules/vite/dist/node/index.js";
import path from "path";
var vite_config_default = defineConfig({
  plugins: [sveltekit()],
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
    strictPort: false
  },
  optimizeDeps: {
    include: ["fabric", "pdf-lib", "socket.io-client"],
    exclude: ["@langchain/core", "@langchain/community", "langchain"]
  },
  define: {
    global: "globalThis"
  },
  ssr: {
    noExternal: ["@langchain/core", "@langchain/community", "langchain"]
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJjOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJjOlxcXFxVc2Vyc1xcXFxqYW1lc1xcXFxEZXNrdG9wXFxcXGRlZWRzLXdlYlxcXFxkZWVkcy13ZWItYXBwXFxcXHZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9jOi9Vc2Vycy9qYW1lcy9EZXNrdG9wL2RlZWRzLXdlYi9kZWVkcy13ZWItYXBwL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcclxuaW1wb3J0IHsgZGVmaW5lQ29uZmlnIH0gZnJvbSAndml0ZSc7XHJcbmltcG9ydCBwYXRoIGZyb20gJ3BhdGgnO1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgZGVmaW5lQ29uZmlnKHtcclxuXHRwbHVnaW5zOiBbc3ZlbHRla2l0KCldLFxyXG5cdHJlc29sdmU6IHtcclxuXHRcdGFsaWFzOiB7XHJcblx0XHRcdCRsaWI6IHBhdGgucmVzb2x2ZSgnLi9zcmMvbGliJyksXHJcblx0XHRcdCRjb21wb25lbnRzOiBwYXRoLnJlc29sdmUoJy4vc3JjL2xpYi9jb21wb25lbnRzJyksXHJcblx0XHRcdCRzZXJ2aWNlczogcGF0aC5yZXNvbHZlKCcuL3NyYy9saWIvc2VydmljZXMnKSxcclxuXHRcdFx0JHR5cGVzOiBwYXRoLnJlc29sdmUoJy4vc3JjL2xpYi90eXBlcycpXHJcblx0XHR9XHJcblx0fSxcclxuXHRzZXJ2ZXI6IHtcclxuXHRcdHBvcnQ6IDUxNzMsXHJcblx0XHRzdHJpY3RQb3J0OiBmYWxzZVxyXG5cdH0sXHJcblx0b3B0aW1pemVEZXBzOiB7XHJcblx0XHRpbmNsdWRlOiBbJ2ZhYnJpYycsICdwZGYtbGliJywgJ3NvY2tldC5pby1jbGllbnQnXSxcclxuXHRcdGV4Y2x1ZGU6IFsnQGxhbmdjaGFpbi9jb3JlJywgJ0BsYW5nY2hhaW4vY29tbXVuaXR5JywgJ2xhbmdjaGFpbiddXHJcblx0fSxcclxuXHRkZWZpbmU6IHtcclxuXHRcdGdsb2JhbDogJ2dsb2JhbFRoaXMnXHJcblx0fSxcclxuXHRzc3I6IHtcclxuXHRcdG5vRXh0ZXJuYWw6IFsnQGxhbmdjaGFpbi9jb3JlJywgJ0BsYW5nY2hhaW4vY29tbXVuaXR5JywgJ2xhbmdjaGFpbiddXHJcblx0fVxyXG59KTsiXSwKICAibWFwcGluZ3MiOiAiO0FBQXdVLFNBQVMsaUJBQWlCO0FBQ2xXLFNBQVMsb0JBQW9CO0FBQzdCLE9BQU8sVUFBVTtBQUVqQixJQUFPLHNCQUFRLGFBQWE7QUFBQSxFQUMzQixTQUFTLENBQUMsVUFBVSxDQUFDO0FBQUEsRUFDckIsU0FBUztBQUFBLElBQ1IsT0FBTztBQUFBLE1BQ04sTUFBTSxLQUFLLFFBQVEsV0FBVztBQUFBLE1BQzlCLGFBQWEsS0FBSyxRQUFRLHNCQUFzQjtBQUFBLE1BQ2hELFdBQVcsS0FBSyxRQUFRLG9CQUFvQjtBQUFBLE1BQzVDLFFBQVEsS0FBSyxRQUFRLGlCQUFpQjtBQUFBLElBQ3ZDO0FBQUEsRUFDRDtBQUFBLEVBQ0EsUUFBUTtBQUFBLElBQ1AsTUFBTTtBQUFBLElBQ04sWUFBWTtBQUFBLEVBQ2I7QUFBQSxFQUNBLGNBQWM7QUFBQSxJQUNiLFNBQVMsQ0FBQyxVQUFVLFdBQVcsa0JBQWtCO0FBQUEsSUFDakQsU0FBUyxDQUFDLG1CQUFtQix3QkFBd0IsV0FBVztBQUFBLEVBQ2pFO0FBQUEsRUFDQSxRQUFRO0FBQUEsSUFDUCxRQUFRO0FBQUEsRUFDVDtBQUFBLEVBQ0EsS0FBSztBQUFBLElBQ0osWUFBWSxDQUFDLG1CQUFtQix3QkFBd0IsV0FBVztBQUFBLEVBQ3BFO0FBQ0QsQ0FBQzsiLAogICJuYW1lcyI6IFtdCn0K
