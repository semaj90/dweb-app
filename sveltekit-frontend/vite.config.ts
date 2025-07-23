import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [tailwindcss(), UnoCSS(), sveltekit()],

  define: {
    global: 'globalThis',
  },

  server: {
    fs: {
      allow: [".."],
    },
    host: true,
    port: 5173,
  },

  optimizeDeps: {
    include: [
      "lucide-svelte", 
      "@tiptap/core", 
      "@tiptap/starter-kit", 
      "fabric",
      "better-sqlite3",
      "drizzle-orm"
    ],
    exclude: ["@auth/sveltekit"]
  },

  build: {
    target: "esnext",
    minify: "terser",
    rollupOptions: {
      external: (id) => {
        return id.includes('node:') || id.includes('@node-rs');
      }
    }
  },

  ssr: {
    noExternal: ['@auth/core', '@auth/sveltekit']
  }
});
