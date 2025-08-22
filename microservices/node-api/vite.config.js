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
      // Keep native node modules and heavy native deps external so Rollup
      // doesn't replace them with browser shims. Also treat any import
      // starting with `node:` as external.
      external: [
        'nats',
        'pg',
        'pino',
        'drizzle-orm',
        'ioredis',
        'zod',
        'http',
        'url',
        'node:http',
        'node:url',
        /^node:.*/
      ]
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