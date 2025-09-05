// vite-config-simd-update.ts
// Updated Vite configuration with SIMD Redis integration
// Add these configurations to your existing vite.config.ts

import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";

// SIMD Performance Plugin
function simdPerformancePlugin() {
  return {
    name: 'simd-performance',
    configureServer(server) {
      // Log SIMD endpoint usage
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/api/go/')) {
          console.log(`[SIMD] ${req.method} ${req.url} - ${new Date().toISOString()}`);
        }
        next();
      });
    },
    
    // Transform imports for SIMD optimization
    transform(code, id) {
      if (id.endsWith('.json') && code.length > 10000) {
        console.log(`[SIMD] Large JSON detected: ${id} (${code.length} bytes)`);
        // Mark for SIMD processing
        return {
          code: code,
          map: null,
          meta: {
            simdOptimized: true
          }
        };
      }
      return null;
    }
  };
}

export default defineConfig(({ mode }) => ({
  plugins: [
    UnoCSS(),
    simdPerformancePlugin(), // Add SIMD performance plugin
    sveltekit()
  ],
  
  // Enhanced server configuration with SIMD Redis
  server: {
    port: 3130,
    host: "0.0.0.0",
    cors: true,
    hmr: {
      port: 3131,
      clientPort: 3131
    },
    fs: {
      allow: ['..', '../../']
    },
    
    // Enhanced proxy configuration for SIMD Redis endpoints
    proxy: {
      // SIMD JSON parsing endpoints
      '/api/go': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.error('[SIMD Proxy Error]:', err.message);
            res.writeHead(502, {
              'Content-Type': 'application/json',
            });
            res.end(JSON.stringify({ 
              error: 'SIMD service unavailable',
              details: err.message 
            }));
          });
          
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Add performance headers
            proxyReq.setHeader('X-Request-Time', Date.now());
            proxyReq.setHeader('X-Client-Type', 'vite-dev');
            
            if (req.method === 'POST' && req.url?.includes('simd')) {
              console.log(`[SIMD Request] ${req.url}`);
            }
          });
          
          proxy.on('proxyRes', (proxyRes, req, res) => {
            // Track response times
            const requestTime = req.headers['x-request-time'];
            if (requestTime) {
              const duration = Date.now() - parseInt(requestTime as string);
              proxyRes.headers['x-response-time'] = duration.toString();
              
              if (duration > 100) {
                console.warn(`[SIMD Slow Response] ${req.url} took ${duration}ms`);
              }
            }
          });
        }
      },
      
      // Direct SIMD endpoints
      '/api/simd-parse': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => '/simd-parse'
      },
      
      '/api/simd-batch': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => '/simd-batch'
      },
      
      // Document processing
      '/api/process-document': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => '/process-document'
      },
      
      // Legal AI endpoints
      '/api/legal': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      
      // Cache management
      '/api/cache': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      
      // Metrics endpoints
      '/api/metrics': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => '/metrics'
      },
      
      '/api/metrics/stream': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => '/metrics/stream',
        configure: (proxy, options) => {
          // Handle SSE for metrics streaming
          proxy.on('proxyRes', (proxyRes, req, res) => {
            proxyRes.headers['cache-control'] = 'no-cache';
            proxyRes.headers['connection'] = 'keep-alive';
          });
        }
      },
      
      // WebSocket proxy for real-time updates
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
        changeOrigin: true
      },
      
      // Existing proxies (Ollama, Qdrant, etc.)
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
      
      '/api/neo4j': {
        target: 'http://localhost:7474',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/neo4j/, '')
      }
    }
  },
  
  preview: {
    port: 4173,
    host: "0.0.0.0",
    cors: true
  },
  
  // Enhanced build optimizations for SIMD
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
        "drizzle-orm"
      ],
      
      // Optimized chunks for SIMD components
      output: {
        manualChunks: {
          // Vendor chunks
          'vendor-svelte': ['svelte', '@sveltejs/kit'],
          'vendor-ui': ['@melt-ui/svelte', '@melt-ui/pp'],
          'vendor-db': ['drizzle-orm', 'postgres'],
          'vendor-cache': ['ioredis'],
          'vendor-ai': ['@langchain/community', '@langchain/core'],
          
          // SIMD-specific chunks
          'simd-redis': [
            './src/lib/services/simd-redis-service.ts',
            './src/lib/components/SIMDPerformanceDashboard.svelte'
          ],
          
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
    
    // Increased chunk size for SIMD optimizations
    chunkSizeWarningLimit: 1500,
    
    // Asset optimization
    assetsInlineLimit: 4096,
    
    // Terser options for production
    terserOptions: mode === 'production' ? {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.info'],
        passes: 2
      },
      mangle: {
        safari10: true
      },
      format: {
        comments: false
      }
    } : undefined
  },
  
  // Enhanced dependency optimization
  optimizeDeps: {
    include: [
      'svelte',
      '@sveltejs/kit',
      '@melt-ui/svelte',
      '@melt-ui/pp',
      'ws', // WebSocket client
      'node-fetch' // For API calls
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
    force: true,
    
    // Entries for SIMD optimization
    entries: [
      'src/lib/services/simd-redis-service.ts',
      'src/lib/components/SIMDPerformanceDashboard.svelte'
    ]
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
      $services: resolve('./src/lib/services'), // Add services alias
      $simd: resolve('./src/lib/services/simd-redis-service.ts') // Direct SIMD import
    }
  },
  
  // CSS processing optimizations
  css: {
    devSourcemap: mode === 'development',
    postcss: mode === 'production' ? {
      plugins: [
        require('autoprefixer'),
        require('cssnano')({
          preset: ['default', {
            discardComments: {
              removeAll: true
            },
            normalizeWhitespace: true
          }]
        })
      ]
    } : undefined
  },
  
  // Enhanced ESBuild configuration
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
    }),
    
    // SIMD-specific optimizations
    supported: {
      'top-level-await': true,
      'async-await': true,
      'object-rest-spread': true
    }
  },
  
  // Worker configuration for SIMD processing
  worker: {
    format: 'es',
    plugins: () => [
      UnoCSS()
    ]
  },
  
  // Environment variables for SIMD configuration
  define: {
    __DEV__: mode === 'development',
    __PROD__: mode === 'production',
    __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    __SIMD_ENABLED__: 'true',
    __SIMD_WORKERS__: JSON.stringify(process.env.SIMD_WORKERS || '16'),
    __REDIS_HOST__: JSON.stringify(process.env.REDIS_HOST || 'localhost'),
    __REDIS_PORT__: JSON.stringify(process.env.REDIS_PORT || '6379')
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
}));
