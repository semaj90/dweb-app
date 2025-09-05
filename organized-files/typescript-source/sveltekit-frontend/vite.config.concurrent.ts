/**
 * Enhanced Vite Configuration for Concurrent Development
 * Integrates with concurrent server architecture for optimal performance
 */

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import UnoCSS from 'unocss/vite';
import { resolve } from 'path';
import { cpus, totalmem } from 'os';

// Import our enhanced configurations
import { vscodeErrorLogger } from './src/lib/vite/vscode-error-logger';

// System resource detection
const CPU_COUNT = cpus().length;
const TOTAL_MEMORY_GB = Math.round(totalmem() / 1024 / 1024 / 1024);

// Enhanced port discovery with concurrent server support
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

export default defineConfig(async ({ mode, command }) => {
  const isDevelopment = mode === 'development';
  const isProduction = mode === 'production';
  
  // Smart port discovery with concurrent server awareness
  const preferredPort = 5173;
  let availablePort: number;
  
  try {
    availablePort = await findAvailablePort(preferredPort);
    if (availablePort !== preferredPort) {
      console.log(`⚠️  Port ${preferredPort} was occupied, using port ${availablePort} instead`);
    }
  } catch (error) {
    console.error(`❌ Failed to find available port: ${error}`);
    availablePort = preferredPort;
  }

  // Optimal concurrency settings
  const OPTIMAL_CONFIG = {
    workers: Math.min(Math.max(CPU_COUNT - 1, 1), 8),
    memoryLimit: TOTAL_MEMORY_GB >= 16 ? 4096 : TOTAL_MEMORY_GB >= 8 ? 2048 : 1024,
    chunkSize: TOTAL_MEMORY_GB >= 16 ? 2000 : TOTAL_MEMORY_GB >= 8 ? 1000 : 500,
  };

  return {
    plugins: [
      UnoCSS(),
      vscodeErrorLogger({
        enabled: isDevelopment,
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
    
    // Development server configuration with concurrent support
    server: {
      port: availablePort,
      host: '0.0.0.0',
      cors: {
        origin: isDevelopment ? true : false,
        credentials: true
      },
      strictPort: false,
      
      // Enhanced HMR with concurrent server coordination
      hmr: {
        port: 3131,
        clientPort: 3131,
        // Integrate with concurrent server monitoring
        overlay: isDevelopment
      },
      
      fs: {
        allow: ['..', '../../', '../../../']
      },
      
      // Enhanced proxy configuration for concurrent services
      proxy: {
        // Ollama LLM service
        '/api/llm': {
          target: 'http://localhost:11434',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/llm/, '/api'),
          timeout: 30000,
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              console.log(`🤖 LLM API: ${req.method} ${req.url}`);
            });
          }
        },
        
        // Qdrant vector database
        '/api/qdrant': {
          target: 'http://localhost:6333',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/qdrant/, ''),
          timeout: 15000
        },
        
        // Enhanced Go microservice proxy with load balancing support
        '/api/go': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/go/, ''),
          timeout: 20000,
          configure: (proxy) => {
            proxy.on('error', (err, req, res) => {
              console.error(`❌ Go microservice proxy error:`, err.message);
            });
            proxy.on('proxyReq', (proxyReq, req) => {
              console.log(`🚀 Go API: ${req.method} ${req.url}`);
            });
          }
        },
        
        // Specialized AI processing endpoints
        '/api/parse': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          timeout: 45000 // Longer timeout for document parsing
        },
        
        '/api/train-som': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          timeout: 60000 // Very long timeout for ML training
        },
        
        '/api/cuda-infer': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          timeout: 30000 // GPU inference timeout
        },
        
        // Neo4j graph database
        '/api/neo4j': {
          target: 'http://localhost:7474',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/neo4j/, ''),
          timeout: 10000
        },
        
        // WebGPU accelerator service (if available)
        '/api/webgpu': {
          target: 'http://localhost:8090',
          changeOrigin: true,
          timeout: 25000,
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq, req) => {
              console.log(`⚡ WebGPU: ${req.method} ${req.url}`);
            });
          }
        },
        
        // MinIO file storage
        '/api/storage': {
          target: 'http://localhost:9000',
          changeOrigin: true,
          timeout: 20000
        }
      }
    },
    
    preview: {
      port: availablePort + 1000,
      host: '0.0.0.0',
      cors: true,
      strictPort: false
    },
    
    // Enhanced build configuration with concurrent optimizations
    build: {
      target: isProduction ? 'es2020' : 'esnext',
      minify: isProduction ? 'esbuild' : false,
      sourcemap: isDevelopment ? 'inline' : false,
      
      // Concurrent build settings
      rollupOptions: {
        external: [
          'amqplib',
          'ioredis',
          '@qdrant/js-client-rest',
          'neo4j-driver',
          '@xstate/svelte',
          'xstate',
          '@langchain/community',
          '@langchain/anthropic',
          '@langchain/google-genai',
          'drizzle-orm',
          'lokijs'
        ],
        
        // Enhanced chunk optimization for concurrent loading
        output: {
          manualChunks: {
            // Core framework chunks
            'vendor-svelte': ['svelte', '@sveltejs/kit'],
            'vendor-ui': ['bits-ui', 'lucide-svelte', 'class-variance-authority'],
            
            // Database and caching chunks
            'vendor-db': ['drizzle-orm', 'postgres'],
            'vendor-cache': ['ioredis', 'lokijs'],
            
            // AI and ML chunks
            'vendor-ai': ['@langchain/community', '@langchain/core', '@xenova/transformers'],
            
            // Concurrent processing chunks
            'concurrent-processing': [
              './src/lib/services/webgpu-loki-accelerator.ts',
              './scripts/concurrent-file-processor.mjs'
            ],
            
            // Feature-specific chunks for parallel loading
            'legal-analysis': [
              './src/lib/legal/analysis.js',
              './src/lib/legal/document-processor.js'
            ],
            
            'neural-engine': [
              './src/lib/engines/neural-sprite-engine.ts',
              './src/lib/engines/cyber-elephant-3d.ts'
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
          },
          
          // Concurrent-friendly chunk naming
          chunkFileNames: (chunkInfo) => {
            const facadeModuleId = chunkInfo.facadeModuleId 
              ? chunkInfo.facadeModuleId.split('/').pop()?.replace('.ts', '').replace('.js', '')
              : 'chunk';
            return `chunks/[name]-${facadeModuleId}-[hash].js`;
          }
        }
      },
      
      // Enhanced chunk size limits for concurrent loading
      chunkSizeWarningLimit: OPTIMAL_CONFIG.chunkSize,
      
      // Asset optimization for concurrent serving
      assetsInlineLimit: 4096,
      
      // Worker configuration for build-time concurrency
      worker: {
        format: 'es',
        plugins: () => [UnoCSS()]
      }
    },
    
    // Enhanced dependency optimization with concurrent considerations
    optimizeDeps: {
      include: [
        'svelte',
        '@sveltejs/kit',
        'bits-ui',
        'lucide-svelte',
        'class-variance-authority',
        'clsx',
        'tailwind-merge'
      ],
      
      exclude: [
        '@langchain/community',
        '@langchain/anthropic',
        '@langchain/google-genai',
        'ioredis',
        'drizzle-orm',
        'postgres',
        '@qdrant/js-client-rest',
        'lokijs',
        'amqplib'
      ],
      
      // Enhanced pre-bundling with concurrency
      force: isDevelopment,
      
      // Concurrent optimization entries
      entries: [
        './src/lib/index.ts',
        './src/routes/**/*.svelte'
      ]
    },
    
    // Path resolution with concurrent service awareness
    resolve: {
      alias: {
        $lib: resolve('./src/lib'),
        $components: resolve('./src/lib/components'),
        $stores: resolve('./src/lib/stores'),
        $utils: resolve('./src/lib/utils'),
        $database: resolve('./src/lib/database'),
        $agents: resolve('./src/lib/agents'),
        $legal: resolve('./src/lib/legal'),
        $services: resolve('./src/lib/services'),
        $engines: resolve('./src/lib/engines')
      }
    },
    
    // Enhanced CSS processing with PostCSS integration
    css: {
      devSourcemap: isDevelopment,
      postcss: './postcss.config.enhanced.js',
      
      // CSS module configuration for concurrent processing
      modules: {
        generateScopedName: isProduction 
          ? '[hash:base64:8]' 
          : '[name]__[local]--[hash:base64:5]',
        hashPrefix: 'legal-ai'
      }
    },
    
    // Enhanced ESBuild configuration
    esbuild: {
      target: isProduction ? 'es2020' : 'esnext',
      keepNames: isDevelopment,
      minify: isProduction,
      legalComments: 'linked',
      
      // Production optimizations
      ...(isProduction && {
        drop: ['console', 'debugger'],
        pure: ['console.log', 'console.warn'],
        treeShaking: true
      })
    },
    
    // Environment variables for concurrent server coordination
    define: {
      __DEV__: isDevelopment,
      __PROD__: isProduction,
      __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __VITE_PORT__: availablePort,
      __HMR_PORT__: 3131,
      __CPU_COUNT__: CPU_COUNT,
      __MEMORY_GB__: TOTAL_MEMORY_GB,
      __OPTIMAL_WORKERS__: OPTIMAL_CONFIG.workers,
      __CONCURRENT_SERVER_ENABLED__: true
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
    },
    
    // Concurrent server integration
    logLevel: isDevelopment ? 'info' : 'warn',
    
    // Clear screen enhancement
    clearScreen: isDevelopment,
    
    // Enhanced caching for concurrent development
    cacheDir: '.vite-concurrent'
  };
});