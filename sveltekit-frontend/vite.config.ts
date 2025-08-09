// @ts-nocheck
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";
import { vscodeErrorLogger } from "./src/lib/vite/vscode-error-logger";

export default defineConfig(({ mode }) => ({
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
  
  // Development server configuration
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
      // Go microservice proxy for high-performance operations
      '/api/go': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('Go microservice proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Proxying to Go microservice:', req.method, req.url);
          });
        }
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
    port: 4173,
    host: "0.0.0.0",
    cors: true
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
        "drizzle-orm"
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
      $legal: resolve('./src/lib/legal')
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
    __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString())
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
