import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api/legal': 'http://localhost:8080',
      '/api/jobs': 'http://localhost:3001',
      '/metrics': 'http://localhost:8080'
    },
    port: 5173,
    host: true
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['xstate', '@xstate/svelte'],
          ui: ['bits-ui', 'melt-ui'],
          utils: ['fuse.js', 'lokijs']
        }
      }
    },
    sourcemap: false,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  optimizeDeps: {
    include: ['xstate', '@xstate/svelte', 'fuse.js', 'lokijs'],
    force: true
  },
  define: {
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __GPU_ENABLED__: true,
    __REDIS_ENABLED__: true
  },
  worker: {
    format: 'es'
  }
});