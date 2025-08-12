import adapter from "@sveltejs/adapter-auto";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Context7 MCP Performance Optimization
  preprocess: vitePreprocess({
    // Disable script preprocessing for performance - TypeScript handled by tsconfig
    script: false,
    // Only preprocess styles and markup
    style: true,
    markup: true
  }),
  
  // Optimize compiler options for performance
  compilerOptions: {
    // Disable dev mode optimizations that slow down checking
    dev: false,
    // Enable performance optimizations
    immutable: true,
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
    },
  },
};

export default config;
