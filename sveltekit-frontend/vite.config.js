import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "@unocss/vite";

export default defineConfig({
  plugins: [
    UnoCSS(),
    sveltekit(),
    // Custom plugin for Go server integration
    {
      name: 'vite-plugin-go-integration',
      configureServer(server) {
        // Proxy API requests to Go GPU server
        server.middlewares.use((req, res, next) => {
          if (req.url?.startsWith('/api/go/')) {
            // Rewrite to Go server
            req.url = req.url.replace('/api/go', '');
          }
          next();
        });
      }
    }
  ],
  
  server: {
    port: 5173,
    host: true,
    hmr: {
      port: 5174,
      overlay: true
    },
    // Proxy configuration for backend services
    proxy: {
      '/api/go': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, ''),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Sending Request to Go server:', req.method, req.url);
          });
        }
      },
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
        changeOrigin: true
      },
      '/api/redis': {
        target: 'http://localhost:6379',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/redis/, '')
      },
      '/api/qdrant': {
        target: 'http://localhost:6333',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/qdrant/, '')
      }
    }
  },
  
  preview: {
    port: 4173,
    host: true,
    proxy: {
      '/api/go': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/go/, '')
      }
    }
  },
  
  css: {
    postcss: "./postcss.config.js",
  },
  
  // Enhanced optimization for production builds
  optimizeDeps: {
    include: [
      "lucide-svelte",
      "xstate",
      "@xstate/svelte",
      "bullmq",
      "ioredis",
      "lokijs",
      "fuse.js",
      "bits-ui",
      "@melt-ui/svelte",
      "zod",
      "socket.io-client"
    ],
    exclude: ['@sveltejs/kit'],
    esbuildOptions: {
      // Use multiple CPU cores for faster builds
      target: 'esnext',
      platform: 'browser',
      // Enable optimizations
      minify: process.env.NODE_ENV === 'production',
      treeShaking: true,
      splitting: true
    }
  },
  
  build: {
    // Enable CSS code splitting for better performance
    cssCodeSplit: true,
    
    // Increase chunk size warning limit for AI models
    chunkSizeWarningLimit: 1000,
    
    // Use Terser for better minification in production
    minify: process.env.NODE_ENV === 'production' ? 'terser' : 'esbuild',
    
    terserOptions: {
      compress: {
        drop_console: process.env.NODE_ENV === 'production',
        drop_debugger: true
      }
    },
    
    // Rollup configuration for optimal chunking
    rollupOptions: {
      output: {
        manualChunks: {
          // UI frameworks
          "ui-framework": ["bits-ui", "@melt-ui/svelte"],
          
          // CSS and styling
          "css-engine": ["unocss", "tailwindcss", "tailwind-merge"],
          
          // Icons
          "icons": ["lucide-svelte"],
          
          // State management
          "state-management": ["xstate", "@xstate/svelte", "svelte/store"],
          
          // AI and processing
          "ai-processing": ["bullmq", "ioredis", "socket.io-client"],
          
          // Client-side data
          "client-data": ["lokijs", "fuse.js"],
          
          // Validation
          "validation": ["zod", "sveltekit-superforms"]
        },
        
        // Asset file naming for cache busting
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.');
          const ext = info[info.length - 1];
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return `assets/images/[name]-[hash][extname]`;
          } else if (/woff|woff2|eot|ttf|otf/i.test(ext)) {
            return `assets/fonts/[name]-[hash][extname]`;
          }
          return `assets/[name]-[hash][extname]`;
        },
        
        chunkFileNames: 'chunks/[name]-[hash].js',
        entryFileNames: 'entries/[name]-[hash].js'
      },
      
      // External dependencies (if using CDN)
      external: [],
      
      // Plugins for additional optimizations
      plugins: []
    },
    
    // Source maps for debugging
    sourcemap: process.env.NODE_ENV === 'development',
    
    // Report compressed size
    reportCompressedSize: true,
    
    // Target modern browsers for better performance
    target: 'esnext',
    
    // Asset inlining threshold
    assetsInlineLimit: 4096
  },
  
  // Worker configuration for Web Workers
  worker: {
    format: 'es',
    rollupOptions: {
      output: {
        entryFileNames: 'workers/[name]-[hash].js'
      }
    }
  },
  
  // Environment variables
  define: {
    '__APP_VERSION__': JSON.stringify(process.env.npm_package_version),
    '__BUILD_TIME__': JSON.stringify(new Date().toISOString()),
    '__GO_SERVER_URL__': JSON.stringify(process.env.VITE_GO_SERVER_URL || 'http://localhost:8080'),
    '__REDIS_URL__': JSON.stringify(process.env.VITE_REDIS_URL || 'localhost:6379'),
    '__USE_GPU__': JSON.stringify(process.env.VITE_USE_GPU !== 'false')
  },
  
  // Performance optimizations
  esbuild: {
    // Use Go's esbuild for faster builds
    logLevel: 'info',
    logLimit: 10,
    legalComments: 'none'
  },
  
  // JSON handling optimization
  json: {
    namedExports: true,
    stringify: false
  },
  
  // Resolve configuration
  resolve: {
    alias: {
      '$lib': '/src/lib',
      '$components': '/src/lib/components',
      '$stores': '/src/lib/stores',
      '$machines': '/src/lib/machines',
      '$utils': '/src/lib/utils',
      '$types': '/src/lib/types'
    }
  }
});
