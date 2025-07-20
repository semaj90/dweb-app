import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";

export default defineConfig({
  plugins: [UnoCSS(), sveltekit()],

  define: {
    // Add any global defines here
  },

  server: {
    fs: {
      allow: [".."],
    },
  },

  optimizeDeps: {
    include: ["lucide-svelte", "@tiptap/core", "@tiptap/starter-kit", "fabric"],
  },

  build: {
    target: "esnext",
    minify: "terser",
  },
});
