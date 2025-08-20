// @ts-nocheck
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";
import { readFileSync } from 'fs';
import { vscodeErrorLogger } from './src/lib/vite/vscode-error-logger.js';

// Smart port discovery
async function findAvailablePort(startPort, maxAttempts = 10) {
  const net = await import('net');
  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    try {
      await new Promise((resolvePromise, reject) => {
        const server = net.createServer();
        server.listen(port, (err) => (err ? reject(err) : server.close(resolvePromise)));
        server.on('error', reject);
      });
      return port;
    } catch {}
  }
  return startPort;
}

export default defineConfig(async ({ mode }) => {
  const preferredPort = 5173;
  const availablePort = await findAvailablePort(preferredPort);

  const isProd = mode === 'production';

  // Resolve package version safely (supports direct node import and Vite env)
  let pkgVersion = process.env.npm_package_version;
  try {
    if (!pkgVersion) {
      const pkgJson = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8'));
      pkgVersion = pkgJson?.version || '1.0.0';
    }
  } catch (e) {
    pkgVersion = pkgVersion || '1.0.0';
  }

  return {
    plugins: [
      UnoCSS(),
      vscodeErrorLogger({
        enabled: !isProd,
        logFile: resolve('.vscode/vite-errors.json'),
        maxEntries: 500,
        includeWarnings: true,
        includeSourceMaps: true
      }),
      sveltekit(),
      // Custom Go server integration middleware
      {
        name: 'vite-plugin-go-integration',
        configureServer(server) {
          server.middlewares.use((req, _res, next) => {
            if (req.url?.startsWith('/api/go/')) {
              req.url = req.url.replace('/api/go', ''); // Rewrite path dynamically
            }
            next();
          });
        }
      }
    ],

    server: {
      port: availablePort,
      host: '0.0.0.0',
      cors: true,
      strictPort: false,
      hmr: { port: 3131, clientPort: 3131 },
      fs: { allow: ['..', '../../'] },
      proxy: {
        '/api/go/enhanced-rag': { target: 'http://localhost:8094', changeOrigin: true, rewrite: path => path.replace(/^\/api\/go\/enhanced-rag/, '') },
        '/api/go/upload': { target: 'http://localhost:8093', changeOrigin: true, rewrite: path => path.replace(/^\/api\/go\/upload/, '') },
        '/api/go/cluster': { target: 'http://localhost:8213', changeOrigin: true, rewrite: path => path.replace(/^\/api\/go\/cluster/, '') },
        '/api/go/xstate': { target: 'http://localhost:8212', changeOrigin: true, rewrite: path => path.replace(/^\/api\/go\/xstate/, '') },
        '/api/ollama': { target: 'http://localhost:11434', changeOrigin: true },
        '/api/nvidia-llama': { target: 'http://localhost:8222', changeOrigin: true, rewrite: path => path.replace(/^\/api\/nvidia-llama/, '') },
        '/api/neo4j': { target: 'http://localhost:7474', changeOrigin: true, rewrite: path => path.replace(/^\/api\/neo4j/, '') }
      }
    },

    preview: { port: availablePort + 1000, host: '0.0.0.0', cors: true, strictPort: false },

    build: {
      target: 'esnext',
      minify: isProd ? 'esbuild' : false,
      sourcemap: !isProd,
      assetsInlineLimit: 4096,
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            "ui-framework": ["bits-ui", "@melt-ui/svelte"],
            "icons": ["lucide-svelte"],
            "state-management": ["xstate", "@xstate/svelte", "svelte/store"],
            "css-engine": ["unocss", "tailwindcss", "tailwind-merge"],
            "ai-processing": ["bullmq", "ioredis", "socket.io-client"],
            "client-data": ["lokijs", "fuse.js"],
            "validation": ["zod", "sveltekit-superforms"]
          },
          assetFileNames: assetInfo => {
            const name = assetInfo.fileName || (Array.isArray(assetInfo.names) ? assetInfo.names[0] : '');
            const ext = (name.split('.').pop() || '').toLowerCase();
            if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) return `assets/images/[name]-[hash][extname]`;
            if (/woff|woff2|eot|ttf|otf/i.test(ext)) return `assets/fonts/[name]-[hash][extname]`;
            return `assets/[name]-[hash][extname]`;
          },
          chunkFileNames: 'chunks/[name]-[hash].js',
          entryFileNames: 'entries/[name]-[hash].js'
        }
      }
    },

    optimizeDeps: {
      include: ["lucide-svelte","xstate","@xstate/svelte","bullmq","ioredis","lokijs","fuse.js","bits-ui","@melt-ui/svelte","zod","socket.io-client"]
    },

    resolve: {
      alias: {
  // Local fallback for bits-ui to avoid incompatible compiled runtime internals
  // During migration to Svelte 5 we use a local shim that exports simple
  // fallback components. Remove this alias once a Svelte-5-compatible
  // bits-ui release or fork is available.
  'bits-ui': resolve('./src/lib/vendor/bits-ui-fallback'),
        $lib: resolve('./src/lib'),
        $components: resolve('./src/lib/components'),
        $stores: resolve('./src/lib/stores'),
        $utils: resolve('./src/lib/utils'),
        $database: resolve('./src/lib/database'),
        $agents: resolve('./src/lib/agents'),
        $legal: resolve('./src/lib/legal')
      }
    },

    worker: {
      rollupOptions: {
        output: { format: 'es', entryFileNames: 'workers/[name]-[hash].js', chunkFileNames: 'workers/chunks/[name]-[hash].js' }
      }
    },

    define: {
      __DEV__: !isProd,
      __PROD__: isProd,
  __VERSION__: JSON.stringify(pkgVersion || '1.0.0'),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __VITE_PORT__: availablePort
    }
  };
});
