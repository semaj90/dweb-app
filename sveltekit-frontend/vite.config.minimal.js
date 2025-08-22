import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  
  // Minimal configuration to avoid dependency issues
  optimizeDeps: {
    include: ['lucide-svelte']
  },
  
  server: {
    port: 5174,
    strictPort: false,
    fs: {
      allow: ['..']
    }
  },
  
  build: {
    target: 'esnext',
    minify: false // Disable minification for debugging
  },
  
  // Disable HMR temporarily to avoid hydration issues
  hmr: false
});