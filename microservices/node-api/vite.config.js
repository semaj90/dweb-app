import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'node20',
    lib: {
      entry: 'src/app.js',
      formats: ['es'],
      fileName: 'index'
    },
    rollupOptions: {
      external: ['nats', 'pg', 'pino', 'drizzle-orm', 'ioredis', 'zod']
    },
    outDir: 'build'
  },
  server: {
    port: 3001,
    host: true
  },
  resolve: {
    alias: {
      '$lib': '/src/lib'
    }
  }
});