import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";

// Smart port discovery utility
async function findAvailablePort(startPort: number, maxAttempts: number = 10): Promise<number> {
  const net = await import('net');

  for (let i = 0; i < maxAttempts; i++) {
    const port = startPort + i;
    try {
      await new Promise<void>((resolve, reject) => {
        const server = net.createServer();
        server.listen(port, (err?: unknown) => {
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

// Load dynamic port configuration
async function loadDynamicPorts(): Promise<Record<string, number>> {
  try {
    const fs = await import('fs/promises');
    const path = await import('path');

    const configPath = path.resolve('../.vscode/dynamic-ports.json');
    const data = await fs.readFile(configPath, 'utf8');
    const config = JSON.parse(data);

    console.log('📡 Loaded dynamic port configuration:', config.ports);
    return config.ports || {};
  } catch (error) {
    console.log('ℹ️  No dynamic port configuration found, using defaults');
    return {};
  }
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

  // Load dynamic port configuration for proxy
  const dynamicPorts = await loadDynamicPorts();

  return {
  plugins: [
      UnoCSS(),
    sveltekit()
  ],

  // Development server configuration with smart port discovery
  server: {
    port: availablePort,
    host: "0.0.0.0",
    cors: true,
    strictPort: false, // Allow Vite to find alternative ports if needed
    hmr: {
      port: 24679,
      clientPort: 24679
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
      // Production Go microservices proxy (dynamic ports)
      '/api/go/enhanced-rag': {
        target: `http://localhost:${dynamicPorts['enhanced-rag'] || 8094}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/enhanced-rag/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log(`Enhanced RAG proxy error (port ${dynamicPorts['enhanced-rag'] || 8094}):`, err.message);
          });
        }
      },
      '/api/go/upload': {
        target: `http://localhost:${dynamicPorts['upload-service'] || 8093}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/upload/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log(`Upload Service proxy error (port ${dynamicPorts['upload-service'] || 8093}):`, err.message);
          });
        }
      },
      '/api/go/cluster': {
        target: `http://localhost:${dynamicPorts['cluster-manager'] || 8213}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/cluster/, ''),
      },
      '/api/go/xstate': {
        target: `http://localhost:${dynamicPorts['xstate-manager'] || 8212}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go\/xstate/, ''),
      },
      '/api/quic': {
        target: `http://localhost:${dynamicPorts['quic-gateway'] || 8447}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/quic/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log(`QUIC Gateway proxy error (port ${dynamicPorts['quic-gateway'] || 8447}):`, err.message);
          });
        }
      },
      '/api/grpc': {
        target: `http://localhost:${dynamicPorts['kratos-server'] || 50051}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/grpc/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log(`Kratos gRPC proxy error (port ${dynamicPorts['kratos-server'] || 50051}):`, err.message);
          });
        }
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

    // CSS processing optimizations
    css: {
      postcss: {
        plugins: [
          require('tailwindcss'),
          require('autoprefixer'),
        ],
      },
      devSourcemap: mode === 'development',
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
          'vendor-ui': ['melt', 'bits-ui'],
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
      'melt',
      'bits-ui'
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
