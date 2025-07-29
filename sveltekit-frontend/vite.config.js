import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "@unocss/vite";

export default defineConfig({
  plugins: [UnoCSS(), sveltekit()],
  server: {
    port: 5173,
    host: true,
    hmr: {
      port: 5174,
    },
  },
  preview: {
    port: 4173,
    host: true,
  },
  css: {
    postcss: "./postcss.config.js",
  },
  optimizeDeps: {
    include: ["lucide-svelte"],
  },
  build: {
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: {
          "ui-framework": ["bits-ui", "@melt-ui/svelte"],
          "css-engine": ["unocss", "tailwindcss", "tailwind-merge"],
          icons: ["lucide-svelte"],
        },
      },
    },
  },
});
