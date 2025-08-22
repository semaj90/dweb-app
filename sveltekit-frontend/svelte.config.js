import adapter from "@sveltejs/adapter-auto";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Context7 MCP Performance Optimization - Svelte 5 + TypeScript support
  preprocess: vitePreprocess({
    // Enable script preprocessing for Svelte 5 + TypeScript complex syntax
    script: true,
    // Enable style and markup preprocessing
    style: true,
    markup: true
  }),

  // Optimize compiler options for performance
  compilerOptions: {
    // Enable dev mode for proper SvelteKit development behavior
    dev: process.env.NODE_ENV === 'development',
    // Reduce bundle size
    css: 'injected'
  },

  kit: {
    adapter: adapter(),
    alias: {
      $lib: "src/lib",
      $components: "src/lib/components",
      $stores: "src/lib/stores",
      $utils: "src/lib/utils",
      $types: "src/lib/types",
      $text: "../shared/text",
      $sharedText: "src/lib/shared-text" // wrapper re-export if preferred
    },
    experimental: {
      remoteFunctions: true
    }
  },
};

export default config;
