// @ts-nocheck
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";
import { vscodeErrorLogger } from './src/lib/vite/vscode-error-logger.js';

// Smart port discovery utility
async function findAvailablePort(startPort: number, maxAttempts: number = 10): Promise<number> {
  const net = await import('net');

  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    try {
      await new Promise<void>((resolve, reject) => {
        const server = net.createServer();
        server.listen(port, (err?: any) => {
          if (err) {
            reject(err);
          } else {
            server.close(() => resolve());
          }
        });
        server.on('error', reject);
      });
      return port;
    } catch (error) {
      console.log(`Port ${port} is occupied, trying next...`);
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

export default defineConfig(async ({ mode }) => {
  // Smart port discovery - prefer 5173, fallback to next available
  const preferredPort = 5173;
  let availablePort: number;

  try {
    availablePort = await findAvailablePort(preferredPort);
    if (availablePort !== preferredPort) {
      console.log(`⚠️  Port ${preferredPort} was occupied, using port ${availablePort} instead`);
    }
  } catch (error) {
    console.error(`❌ Failed to find available port: ${error}`);
    availablePort = preferredPort; // Fallback to default
  }

  return {
  plugins: [
    UnoCSS(),
    vscodeErrorLogger({
      enabled: mode === 'development',
      logFile: resolve('.vscode/vite-errors.json'),
      maxEntries: 500,
      includeWarnings: true,
      includeSourceMaps: true,
      autoOpenProblems: false,
      notificationLevel: 'errors-only',
      integrateTasks: true,
      generateDiagnostics: true
    }),
    sveltekit()
  ],

  // Development server configuration with smart port discovery
  server: {
    port: availablePort,
    host: "0.0.0.0",
    cors: true,
    strictPort: false, // Allow Vite to find alternative ports if needed
    hmr: {
      port: 3131,
      clientPort: 3131
    },
    fs: {
      allow: ['..', '../../']
    },
    // Proxy for API calls during development
    proxy: {
      '/api/llm': {
        target: 'http://localhost:11434',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/llm/, '/api')
      },
      '/api/qdrant': {
        target: 'http://localhost:6333',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/qdrant/, '')
      },
      // Production Go microservices proxy (multi-tier)
      '/api/go/enhanced-rag': {
        target: 'http://localhost:8094',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/enhanced-rag/, ''),
      },
      '/api/go/upload': {
        target: 'http://localhost:8093',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/upload/, ''),
      },
      '/api/go/cluster': {
        target: 'http://localhost:8213',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/cluster/, ''),
      },
      '/api/go/xstate': {
        target: 'http://localhost:8212',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/xstate/, ''),
      },
      // Multi-core Ollama cluster (load balanced)
      '/api/ollama': {
        target: 'http://localhost:11434',
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('Ollama cluster proxy error:', err);
          });
        }
      },
      // NVIDIA go-llama integration
      '/api/nvidia-llama': {
        target: 'http://localhost:8222', // Load balancer port
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/nvidia-llama/, ''),
      },
      '/api/parse': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      '/api/train-som': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      '/api/cuda-infer': {
        target: 'http://localhost:8080',
        changeOrigin: true
      },
      // Neo4j database proxy
      '/api/neo4j': {
        target: 'http://localhost:7474',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/neo4j/, '')
      }
    }
  },

  preview: {
    port: availablePort + 1000, // Use different port for preview
    host: "0.0.0.0",
    cors: true,
    strictPort: false // Allow alternative ports for preview too
  },

  // Build optimizations
  build: {
    target: 'esnext',
    minify: mode === 'production' ? 'esbuild' : false,
    sourcemap: mode === 'development',

    rollupOptions: {
      external: [
        "amqplib",
        "ioredis",
        "@qdrant/js-client-rest",
        "neo4j-driver",
        "@xstate/svelte",
        "xstate",
        "@langchain/community",
        "@langchain/anthropic",
        "@langchain/google-genai",
        "drizzle-orm",
        "minio",
        "sharp"
      ],

      // Optimize chunks for performance
      output: {
        manualChunks: {
          // Vendor chunks
          'vendor-svelte': ['svelte', '@sveltejs/kit'],
          'vendor-ui': ['@melt-ui/svelte', '@melt-ui/pp'],
          'vendor-db': ['drizzle-orm', 'postgres'],
          'vendor-cache': ['ioredis'],
          'vendor-ai': ['@langchain/community', '@langchain/core'],

          // Feature chunks
          'legal-analysis': [
            './src/lib/legal/analysis.js',
            './src/lib/legal/document-processor.js'
          ],
          'agent-orchestration': [
            './src/lib/agents/orchestrator.js',
            './src/lib/agents/crew-ai.js'
          ],
          'database-layer': [
            './src/lib/database/redis.js',
            './src/lib/database/qdrant.js',
            './src/lib/database/postgres.js'
          ]
        }
      }
    },

    // Chunk size warnings threshold
    chunkSizeWarningLimit: 1000,

    // Asset optimization
    assetsInlineLimit: 4096,
  },

  // Dependency optimization
  optimizeDeps: {
    include: [
      'svelte',
      '@sveltejs/kit',
      '@melt-ui/svelte',
      '@melt-ui/pp'
    ],
    exclude: [
      '@langchain/community',
      '@langchain/anthropic',
      '@langchain/google-genai',
      'ioredis',
      'drizzle-orm',
      'postgres',
      '@qdrant/js-client-rest'
    ],

    // Force pre-bundling for better performance
    force: true
  },

  // Path resolution
  resolve: {
    alias: {
      $lib: resolve('./src/lib'),
      $components: resolve('./src/lib/components'),
      $stores: resolve('./src/lib/stores'),
      $utils: resolve('./src/lib/utils'),
      $database: resolve('./src/lib/database'),
      $agents: resolve('./src/lib/agents'),
  $legal: resolve('./src/lib/legal'),
  '@shared': resolve('../shared'),
  '@text': resolve('../shared/text')
    }
  },

  // CSS processing optimizations
  css: {
    devSourcemap: mode === 'development',
    postcss: mode === 'production' ? {
      plugins: [
        require('autoprefixer'),
        require('cssnano')({
          preset: 'default'
        })
      ]
    } : undefined
  },

  // ESBuild configuration for optimal transpilation
  esbuild: {
    target: 'esnext',
    keepNames: mode === 'development',
    minify: mode === 'production',

    // Legal compliance - preserve license comments
    legalComments: 'linked',

    // Drop console/debugger in production
    ...(mode === 'production' && {
      drop: ['console', 'debugger'],
      pure: ['console.log', 'console.warn']
    })
  },

  // Worker configuration for Node.js clustering support
  worker: {
    format: 'es',
    plugins: () => [
      UnoCSS()
    ]
  },

  // Environment variables
  define: {
    __DEV__: mode === 'development',
    __PROD__: mode === 'production',
    __VERSION__: JSON.stringify(import.meta.env.npm_package_version || '1.0.0'),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __VITE_PORT__: availablePort,
    __MCP_SERVER_PORT__: 4100,
    __GRPC_SERVER_PORT__: 8084
  },

  // Performance optimizations
  experimental: {
    renderBuiltUrl(filename, { hostType }) {
      if (hostType === 'js') {
        return `/${filename}`;
      } else {
        return { relative: true };
      }
    }
  }
  };
});
