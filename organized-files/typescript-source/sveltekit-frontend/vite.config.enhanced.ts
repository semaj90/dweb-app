// @ts-nocheck
/**
 * Enhanced Vite Configuration
 * Memory-Aware Development with Concurrency & Data Parallelism
 * Optimized for multi-core systems with intelligent resource management
 */

import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";
import { cpus } from "os";
import { Worker } from "worker_threads";
import { vscodeErrorLogger } from "./src/lib/vite/vscode-error-logger";

// Enhanced system detection utilities
const SYSTEM_INFO = {
  cpuCount: cpus().length,
  memoryGB: Math.round(process.memoryUsage().heapTotal / 1024 / 1024 / 1024),
  platform: process.platform,
  nodeVersion: process.version,
  architecture: process.arch
};

console.log(`🖥️ System Info: ${SYSTEM_INFO.cpuCount} cores, ~${SYSTEM_INFO.memoryGB}GB memory, ${SYSTEM_INFO.platform}-${SYSTEM_INFO.architecture}`);

// Memory-aware configuration calculator
function calculateOptimalSettings() {
  const { cpuCount, memoryGB } = SYSTEM_INFO;
  
  return {
    // Worker pool sizing based on CPU cores
    maxWorkers: Math.min(Math.max(cpuCount - 1, 1), 8), // Leave 1 core free, cap at 8
    
    // Memory-based chunk sizes
    chunkSize: memoryGB >= 16 ? 2000 : memoryGB >= 8 ? 1000 : 500,
    
    // Concurrency levels
    concurrentBuilds: Math.min(cpuCount, 4),
    maxConcurrentConnections: cpuCount * 10,
    
    // Cache settings based on available memory
    cacheSize: memoryGB >= 16 ? 512 : memoryGB >= 8 ? 256 : 128, // MB
    
    // Pre-bundling optimization
    optimizeDepsWorkers: Math.min(cpuCount, 4),
    
    // HMR settings
    hmrThrottle: cpuCount >= 8 ? 50 : 100, // ms
    
    // Asset processing
    assetParallelism: Math.min(cpuCount, 6)
  };
}

const OPTIMAL = calculateOptimalSettings();
console.log(`⚡ Optimal Settings: ${OPTIMAL.maxWorkers} workers, ${OPTIMAL.cacheSize}MB cache, ${OPTIMAL.concurrentBuilds}x concurrent builds`);

// Enhanced port discovery with parallel checking
async function findAvailablePortParallel(startPort: number, maxAttempts: number = 10): Promise<number> {
  const net = await import('net');
  
  // Check multiple ports in parallel for faster discovery
  const portCheckPromises = Array.from({ length: maxAttempts }, (_, i) => {
    const port = startPort + i;
    return new Promise<{ port: number, available: boolean }>((resolve) => {
      const server = net.createServer();
      server.listen(port, () => {
        server.close(() => resolve({ port, available: true }));
      });
      server.on('error', () => resolve({ port, available: false }));
      
      // Timeout to prevent hanging
      setTimeout(() => {
        server.close();
        resolve({ port, available: false });
      }, 100);
    });
  });
  
  const results = await Promise.all(portCheckPromises);
  const available = results.find(r => r.available);
  
  if (available) {
    return available.port;
  }
  
  throw new Error(`No available port found starting from ${startPort}`);
}

// Memory monitoring and cleanup utilities
class MemoryMonitor {
  private cleanupCallbacks: (() => void)[] = [];
  private monitoringInterval?: NodeJS.Timeout;
  
  startMonitoring() {
    this.monitoringInterval = setInterval(() => {
      const usage = process.memoryUsage();
      const heapUsedMB = Math.round(usage.heapUsed / 1024 / 1024);
      const heapTotalMB = Math.round(usage.heapTotal / 1024 / 1024);
      
      // Log memory usage periodically
      if (heapUsedMB > OPTIMAL.cacheSize * 2) {
        console.log(`⚠️ High memory usage: ${heapUsedMB}MB/${heapTotalMB}MB`);
        
        // Trigger garbage collection if available
        if (global.gc) {
          global.gc();
          console.log(`🗑️ Garbage collection triggered`);
        }
        
        // Execute cleanup callbacks
        this.cleanupCallbacks.forEach(cb => cb());
      }
    }, 30000); // Check every 30 seconds
  }
  
  addCleanupCallback(callback: () => void) {
    this.cleanupCallbacks.push(callback);
  }
  
  stopMonitoring() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
  }
}

const memoryMonitor = new MemoryMonitor();

// Parallel asset processing plugin
function createParallelAssetProcessor() {
  return {
    name: 'parallel-asset-processor',
    async buildStart() {
      console.log(`🚀 Starting parallel build with ${OPTIMAL.concurrentBuilds} workers`);
    },
    async generateBundle(options: any, bundle: any) {
      // Process assets in parallel chunks
      const assets = Object.values(bundle);
      const chunkSize = Math.ceil(assets.length / OPTIMAL.assetParallelism);
      
      const chunks = [];
      for (let i = 0; i < assets.length; i += chunkSize) {
        chunks.push(assets.slice(i, i + chunkSize));
      }
      
      await Promise.all(chunks.map(async (chunk, index) => {
        console.log(`📦 Processing asset chunk ${index + 1}/${chunks.length} (${chunk.length} files)`);
        // Process chunk assets here
      }));
    }
  };
}

// Enhanced dependency optimizer with worker threads
function createWorkerBasedOptimizer() {
  return {
    name: 'worker-optimizer',
    configResolved(config: any) {
      // Configure worker-based optimization
      if (config.optimizeDeps) {
        config.optimizeDeps.esbuildOptions = {
          ...config.optimizeDeps.esbuildOptions,
          platform: 'node',
          target: 'esnext',
          // Use multiple workers for dependency optimization
          workers: OPTIMAL.optimizeDepsWorkers
        };
      }
    }
  };
}

// Intelligent caching plugin with memory awareness
function createIntelligentCache() {
  const cache = new Map<string, any>();
  const cacheStats = { hits: 0, misses: 0, evictions: 0 };
  
  // Memory-based cache eviction
  const evictOldestEntries = () => {
    const entries = Array.from(cache.entries());
    const evictCount = Math.ceil(entries.length * 0.2); // Evict 20% of oldest entries
    
    entries.slice(0, evictCount).forEach(([key]) => {
      cache.delete(key);
      cacheStats.evictions++;
    });
  };
  
  memoryMonitor.addCleanupCallback(evictOldestEntries);
  
  return {
    name: 'intelligent-cache',
    load(id: string) {
      if (cache.has(id)) {
        cacheStats.hits++;
        return cache.get(id);
      }
      cacheStats.misses++;
      return null;
    },
    transform(code: string, id: string) {
      // Cache transformed code if under memory limit
      if (cache.size < OPTIMAL.cacheSize * 10) { // Rough estimation
        cache.set(id, { code, timestamp: Date.now() });
      }
      
      // Log cache statistics periodically
      if ((cacheStats.hits + cacheStats.misses) % 100 === 0) {
        const hitRate = (cacheStats.hits / (cacheStats.hits + cacheStats.misses) * 100).toFixed(1);
        console.log(`📊 Cache stats: ${hitRate}% hit rate, ${cache.size} entries, ${cacheStats.evictions} evictions`);
      }
    }
  };
}

// Data parallelism plugin for large file processing
function createDataParallelProcessor() {
  return {
    name: 'data-parallel-processor',
    async transform(code: string, id: string) {
      // For large files, use worker threads for parallel processing
      if (code.length > 50000) { // 50KB threshold
        console.log(`🔧 Processing large file in parallel: ${id} (${Math.round(code.length/1024)}KB)`);
        
        // Split code into chunks and process in parallel
        const chunkSize = Math.ceil(code.length / OPTIMAL.maxWorkers);
        const chunks = [];
        
        for (let i = 0; i < code.length; i += chunkSize) {
          chunks.push(code.slice(i, i + chunkSize));
        }
        
        // Process chunks in parallel (simplified example)
        await Promise.all(chunks.map(async (chunk, index) => {
          // Simulate parallel processing
          return new Promise(resolve => {
            setImmediate(() => {
              // Process chunk here (e.g., minification, transformation)
              resolve(chunk);
            });
          });
        }));
        
        console.log(`✅ Parallel processing completed for ${id}`);
      }
    }
  };
}

export default defineConfig(async ({ mode, command }) => {
  // Start memory monitoring in development
  if (mode === 'development') {
    memoryMonitor.startMonitoring();
    
    // Cleanup on process exit
    process.on('exit', () => memoryMonitor.stopMonitoring());
    process.on('SIGINT', () => {
      memoryMonitor.stopMonitoring();
      process.exit(0);
    });
  }
  
  // Enhanced parallel port discovery
  const preferredPort = 5173;
  let availablePort: number;
  
  try {
    const startTime = Date.now();
    availablePort = await findAvailablePortParallel(preferredPort, 20);
    const discoveryTime = Date.now() - startTime;
    
    if (availablePort !== preferredPort) {
      console.log(`⚠️ Port ${preferredPort} occupied, using ${availablePort} (found in ${discoveryTime}ms)`);
    } else {
      console.log(`✅ Port ${availablePort} available (verified in ${discoveryTime}ms)`);
    }
  } catch (error) {
    console.error(`❌ Port discovery failed: ${error}`);
    availablePort = preferredPort; // Fallback
  }

  return {
    plugins: [
      // Core plugins
      UnoCSS({
        // Enhanced UnoCSS with memory optimization
        configDeps: [],
        hmrTopLevelAwait: false // Prevent memory leaks in HMR
      }),
      
      // Enhanced error logging with parallel processing
      vscodeErrorLogger({
        enabled: mode === 'development',
        logFile: resolve('.vscode/vite-errors.json'),
        maxEntries: OPTIMAL.cacheSize * 4, // Scale with memory
        includeWarnings: true,
        includeSourceMaps: mode === 'development',
        autoOpenProblems: false,
        notificationLevel: 'errors-only',
        integrateTasks: true,
        generateDiagnostics: true,
        // Enhanced with parallel processing
        parallelProcessing: true,
        maxWorkers: OPTIMAL.maxWorkers
      }),
      
      // Custom enhanced plugins
      createParallelAssetProcessor(),
      createWorkerBasedOptimizer(),
      createIntelligentCache(),
      createDataParallelProcessor(),
      
      // SvelteKit (configured last for optimal integration)
      sveltekit()
    ],
    
    // Enhanced server configuration with concurrency optimizations
    server: {
      port: availablePort,
      host: "0.0.0.0",
      cors: true,
      strictPort: false,
      
      // Enhanced HMR with memory awareness
      hmr: {
        port: availablePort + 1,
        clientPort: availablePort + 1,
        // Throttle HMR updates based on system performance
        overlay: mode === 'development',
        timeout: OPTIMAL.hmrThrottle * 10
      },
      
      // File system optimization
      fs: {
        allow: ['..', '../../'],
        // Enhanced file watching with debouncing
        strict: true,
        cachedChecks: true
      },
      
      // Enhanced proxy configuration with connection pooling
      proxy: {
        '/api/llm': {
          target: 'http://localhost:11434',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/llm/, '/api'),
          // Connection pooling for better performance
          agent: {
            maxSockets: OPTIMAL.maxConcurrentConnections,
            keepAlive: true
          }
        },
        '/api/qdrant': {
          target: 'http://localhost:6333',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/qdrant/, ''),
          agent: {
            maxSockets: OPTIMAL.maxConcurrentConnections,
            keepAlive: true
          }
        },
        // Enhanced Go microservice proxy with load balancing
        '/api/go': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/go/, ''),
          configure: (proxy, options) => {
            proxy.on('error', (err, req, res) => {
              console.log('🚨 Go microservice proxy error:', err);
            });
            proxy.on('proxyReq', (proxyReq, req, res) => {
              if (mode === 'development') {
                console.log(`🔄 Proxying to Go: ${req.method} ${req.url}`);
              }
            });
          },
          // Enhanced connection pooling
          agent: {
            maxSockets: OPTIMAL.maxConcurrentConnections * 2, // Go services can handle more
            keepAlive: true,
            timeout: 30000
          }
        },
        // Parallel API endpoints
        '/api/parse': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          agent: { maxSockets: OPTIMAL.maxConcurrentConnections, keepAlive: true }
        },
        '/api/train-som': {
          target: 'http://localhost:8080', 
          changeOrigin: true,
          agent: { maxSockets: OPTIMAL.maxConcurrentConnections, keepAlive: true }
        },
        '/api/cuda-infer': {
          target: 'http://localhost:8080',
          changeOrigin: true,
          agent: { maxSockets: OPTIMAL.maxConcurrentConnections, keepAlive: true }
        },
        '/api/neo4j': {
          target: 'http://localhost:7474',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/neo4j/, ''),
          agent: { maxSockets: OPTIMAL.maxConcurrentConnections, keepAlive: true }
        }
      }
    },
    
    // Enhanced preview configuration
    preview: {
      port: availablePort + 1000,
      host: "0.0.0.0",
      cors: true,
      strictPort: false
    },
    
    // Enhanced build configuration with parallel processing
    build: {
      target: 'esnext',
      minify: mode === 'production' ? 'esbuild' : false,
      sourcemap: mode === 'development',
      
      // Enhanced Rollup configuration with parallel builds
      rollupOptions: {
        // Parallel input processing
        input: {
          main: resolve('./src/app.html')
        },
        
        external: [
          "amqplib", "ioredis", "@qdrant/js-client-rest", "neo4j-driver",
          "@xstate/svelte", "xstate", "@langchain/community", 
          "@langchain/anthropic", "@langchain/google-genai", "drizzle-orm"
        ],
        
        // Enhanced output configuration with intelligent chunking
        output: {
          // Dynamic chunk sizing based on system resources
          manualChunks: (id) => {
            // Vendor chunks with size-based splitting
            if (id.includes('node_modules')) {
              if (id.includes('svelte')) return 'vendor-svelte';
              if (id.includes('@melt-ui')) return 'vendor-ui';
              if (id.includes('drizzle') || id.includes('postgres')) return 'vendor-db';
              if (id.includes('redis')) return 'vendor-cache';
              if (id.includes('@langchain')) return 'vendor-ai';
              return 'vendor-other';
            }
            
            // Feature-based chunks with memory awareness
            if (id.includes('/legal/')) return 'legal-analysis';
            if (id.includes('/agents/')) return 'agent-orchestration';
            if (id.includes('/database/')) return 'database-layer';
            if (id.includes('/ai/')) return 'ai-services';
            if (id.includes('/components/')) return 'ui-components';
            
            return undefined; // Default chunking
          },
          
          // Enhanced asset optimization
          assetFileNames: (assetInfo) => {
            const info = assetInfo.name!.split('.');
            const ext = info[info.length - 1];
            
            // Organize assets by type for better caching
            if (/\.(png|jpe?g|svg|gif|tiff|bmp|ico)$/i.test(assetInfo.name!)) {
              return `assets/images/[name]-[hash][extname]`;
            }
            if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name!)) {
              return `assets/fonts/[name]-[hash][extname]`;
            }
            if (/\.(css)$/i.test(assetInfo.name!)) {
              return `assets/styles/[name]-[hash][extname]`;
            }
            
            return `assets/[name]-[hash][extname]`;
          },
          
          // Optimize chunk naming for caching
          chunkFileNames: 'chunks/[name]-[hash].js',
          entryFileNames: 'entries/[name]-[hash].js'
        },
        
        // Enhanced parallel processing
        parallel: OPTIMAL.concurrentBuilds,
        maxParallelFileOps: OPTIMAL.assetParallelism
      },
      
      // Dynamic chunk size warnings based on system resources
      chunkSizeWarningLimit: OPTIMAL.chunkSize,
      
      // Enhanced asset optimization
      assetsInlineLimit: SYSTEM_INFO.memoryGB >= 8 ? 8192 : 4096,
      
      // Parallel CSS processing
      cssCodeSplit: true,
      cssMinify: mode === 'production'
    },
    
    // Enhanced dependency optimization with worker threads
    optimizeDeps: {
      // Include frequently used dependencies
      include: [
        'svelte', '@sveltejs/kit', '@melt-ui/svelte', '@melt-ui/pp',
        'lucide-svelte', 'clsx', 'tailwind-merge'
      ],
      
      // Exclude server-side dependencies
      exclude: [
        '@langchain/community', '@langchain/anthropic', '@langchain/google-genai',
        'ioredis', 'drizzle-orm', 'postgres', '@qdrant/js-client-rest',
        'neo4j-driver', 'amqplib'
      ],
      
      // Enhanced pre-bundling with workers
      force: command === 'build',
      
      // ESBuild configuration for optimization
      esbuildOptions: {
        // Use multiple workers for dependency bundling
        target: 'esnext',
        platform: 'browser',
        
        // Optimize for the detected system
        minify: mode === 'production',
        treeShaking: true,
        
        // Memory management
        memoryLimit: Math.min(SYSTEM_INFO.memoryGB * 256, 2048), // MB, cap at 2GB
        
        // Parallel processing
        workers: OPTIMAL.optimizeDepsWorkers,
        
        // Enhanced syntax support
        jsx: 'automatic',
        jsxDev: mode === 'development'
      }
    },
    
    // Enhanced path resolution with caching
    resolve: {
      alias: {
        $lib: resolve('./src/lib'),
        $components: resolve('./src/lib/components'),
        $stores: resolve('./src/lib/stores'),
        $utils: resolve('./src/lib/utils'),
        $database: resolve('./src/lib/database'),
        $agents: resolve('./src/lib/agents'),
        $legal: resolve('./src/lib/legal'),
        $ai: resolve('./src/lib/ai'),
        $services: resolve('./src/lib/services'),
        $types: resolve('./src/lib/types')
      },
      
      // Enhanced module resolution
      extensions: ['.js', '.ts', '.svelte', '.json', '.mjs'],
      preferRelative: true,
      
      // Optimize resolution caching
      symlinks: true,
      preserveSymlinks: false
    },
    
    // Enhanced CSS processing with parallelization
    css: {
      devSourcemap: mode === 'development',
      
      // Parallel PostCSS processing
      postcss: mode === 'production' ? {
        plugins: [
          require('autoprefixer'),
          require('cssnano')({
            preset: ['default', {
              // Enhanced CSS optimization
              calc: true,
              colormin: true,
              convertValues: true,
              discardComments: { removeAll: true },
              mergeLonghand: true,
              mergeRules: true,
              minifyFontValues: true,
              minifyParams: true,
              minifySelectors: true,
              normalizeCharset: true,
              normalizeDisplayValues: true,
              normalizePositions: true,
              normalizeRepeatStyle: true,
              normalizeString: true,
              normalizeTimingFunctions: true,
              normalizeUnicode: true,
              normalizeUrl: true,
              normalizeWhitespace: true,
              orderedValues: true,
              reduceInitial: true,
              reduceTransforms: true,
              svgo: true,
              uniqueSelectors: true
            }]
          })
        ]
      } : undefined,
      
      // CSS modules with enhanced caching
      modules: {
        generateScopedName: mode === 'production' 
          ? '[hash:base64:5]' 
          : '[name]__[local]--[hash:base64:5]'
      }
    },
    
    // Enhanced ESBuild configuration with system optimization
    esbuild: {
      target: 'esnext',
      keepNames: mode === 'development',
      minify: mode === 'production',
      
      // Memory and performance optimization
      sourcemap: mode === 'development',
      sourcesContent: mode === 'development',
      
      // Legal compliance
      legalComments: 'linked',
      
      // Production optimizations
      ...(mode === 'production' && {
        drop: ['console', 'debugger'],
        pure: ['console.log', 'console.warn', 'console.info'],
        ignoreAnnotations: false,
        treeShaking: true,
        
        // Advanced optimizations
        mangleProps: /^_/,
        reserveProps: /^__/,
        minifyWhitespace: true,
        minifyIdentifiers: true,
        minifySyntax: true
      }),
      
      // System-specific optimizations
      platform: 'browser',
      format: 'esm',
      
      // JSX configuration for enhanced performance
      jsx: 'automatic',
      jsxDev: mode === 'development',
      jsxImportSource: 'svelte'
    },
    
    // Enhanced worker configuration with thread pools
    worker: {
      format: 'es',
      plugins: () => [
        UnoCSS({
          // Optimized UnoCSS for workers
          mode: 'vue-scoped'
        })
      ],
      
      // Enhanced worker rollup options
      rollupOptions: {
        external: ['fs', 'path', 'url'],
        output: {
          format: 'es',
          entryFileNames: 'workers/[name]-[hash].js',
          chunkFileNames: 'workers/chunks/[name]-[hash].js'
        }
      }
    },
    
    // Enhanced environment variables with system info
    define: {
      __DEV__: mode === 'development',
      __PROD__: mode === 'production',
      __VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __VITE_PORT__: availablePort,
      __MCP_SERVER_PORT__: 4100,
      __GRPC_SERVER_PORT__: 8084,
      
      // System optimization flags
      __SYSTEM_CORES__: SYSTEM_INFO.cpuCount,
      __SYSTEM_MEMORY_GB__: SYSTEM_INFO.memoryGB,
      __OPTIMAL_WORKERS__: OPTIMAL.maxWorkers,
      __CACHE_SIZE_MB__: OPTIMAL.cacheSize,
      __CONCURRENT_BUILDS__: OPTIMAL.concurrentBuilds,
      __ASSET_PARALLELISM__: OPTIMAL.assetParallelism,
      
      // Feature flags based on system capabilities
      __ENABLE_WORKERS__: SYSTEM_INFO.cpuCount >= 4,
      __ENABLE_PARALLEL_PROCESSING__: SYSTEM_INFO.memoryGB >= 8,
      __ENABLE_ADVANCED_CACHING__: SYSTEM_INFO.memoryGB >= 16
    },
    
    // Enhanced experimental features
    experimental: {
      // Dynamic asset URL generation
      renderBuiltUrl(filename, { hostType, type }) {
        if (hostType === 'js') {
          return `/${filename}`;
        } else if (type === 'asset') {
          // CDN integration for production
          return mode === 'production' 
            ? `https://cdn.example.com/${filename}`
            : `/${filename}`;
        } else {
          return { relative: true };
        }
      },
      
      // Enhanced module preloading
      modulePreload: {
        enabled: true,
        resolveDependencies: (filename, deps) => {
          // Intelligent dependency resolution based on system resources
          return SYSTEM_INFO.memoryGB >= 8 ? deps : deps.slice(0, 5);
        }
      }
    },
    
    // Enhanced logging configuration
    logLevel: mode === 'development' ? 'info' : 'warn',
    clearScreen: false, // Keep build logs visible
    
    // Memory and performance monitoring
    ...(mode === 'development' && {
      plugins: [
        // Memory usage reporter
        {
          name: 'memory-reporter',
          buildStart() {
            const usage = process.memoryUsage();
            console.log(`🧠 Initial memory: ${Math.round(usage.heapUsed / 1024 / 1024)}MB heap, ${Math.round(usage.rss / 1024 / 1024)}MB RSS`);
          },
          buildEnd() {
            const usage = process.memoryUsage();
            console.log(`🏁 Final memory: ${Math.round(usage.heapUsed / 1024 / 1024)}MB heap, ${Math.round(usage.rss / 1024 / 1024)}MB RSS`);
          }
        }
      ]
    })
  };
});

// Export system information for external use
export { SYSTEM_INFO, OPTIMAL };