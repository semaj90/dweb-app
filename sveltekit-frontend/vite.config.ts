import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import UnoCSS from "unocss/vite";
import { resolve } from "path";

// Enhanced plugin to fix SuperForms SuperDebug compatibility with Svelte 5
function superFormsCompat() {
  return {
    name: 'superforms-svelte5-compat',
    enforce: 'pre' as const,
    configResolved(config: any) {
      // Add comprehensive aliases to intercept SuperDebug
      config.resolve = config.resolve || {};
      config.resolve.alias = config.resolve.alias || {};

      // Multiple path variations to catch all imports
      config.resolve.alias['sveltekit-superforms/dist/client/SuperDebug.svelte'] = 'virtual:superdebug-compat';
      config.resolve.alias['sveltekit-superforms/client/SuperDebug.svelte'] = 'virtual:superdebug-compat';
      config.resolve.alias['SuperDebug.svelte'] = 'virtual:superdebug-compat';
    },
    resolveId(id: string, importer?: string) {
      // Comprehensive ID resolution for all SuperDebug variations
      const superDebugPaths = [
        'sveltekit-superforms/dist/client/SuperDebug.svelte',
        'sveltekit-superforms/client/SuperDebug.svelte',
        'SuperDebug.svelte'
      ];

      if (superDebugPaths.some(path => id === path || id.endsWith(path) || id.includes('SuperDebug'))) {
        console.log(`🔧 Intercepting SuperDebug import: ${id} from ${importer}`);
        return 'virtual:superdebug-compat';
      }
    },
    load(id: string) {
      if (id === 'virtual:superdebug-compat') {
        console.log('✅ Loading Svelte 5 compatible SuperDebug replacement');
        // Return a completely Svelte 5 compatible SuperDebug component
        return `
<script lang="ts">
  // Svelte 5 runes mode compatible props with comprehensive types
  interface DebugData {
    [key: string]: any;
  }

  interface SuperDebugProps {
    data?: DebugData;
    display?: boolean;
    label?: string;
    status?: string | undefined;
    collapsed?: boolean;
    stringTruncate?: number;
    raw?: boolean;
  }

  let {
    data = {} as DebugData,
    display = true,
    label = 'Form Data',
    status = undefined,
    collapsed = false,
    stringTruncate = 100,
    raw = false
  }: SuperDebugProps = $props();

  // Format data for display
  const formatData = (obj: any): string => {
    if (raw) return String(obj);
    try {
      return JSON.stringify(obj, null, 2);
    } catch (e: any) {
      return String(obj);
    }
  };
</script>

{#if display}
  <div class="superdebug-svelte5-compat" style="margin: 1rem 0; padding: 1rem; background: #f8f9fa; border: 2px solid #6c757d; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 12px;">
    <details {open: !collapsed}>
      <summary style="cursor: pointer; font-weight: bold; margin-bottom: 0.5rem; color: #495057; user-select: none;">
        🐛 {label} {status ? \`(\${status})\` : ''} - Svelte 5 Compatible
      </summary>
      <div style="max-height: 400px; overflow: auto; background: #ffffff; padding: 0.5rem; border: 1px solid #dee2e6; border-radius: 2px;">
        <pre style="margin: 0; white-space: pre-wrap; font-size: 11px; color: #212529;">{formatData(data)}</pre>
      </div>
    </details>
  </div>
{/if}
        `;
      }
    },
    transform(code: string, id: string) {
      // Aggressive transformation to replace any SuperDebug references
      if (code.includes('SuperDebug') && !id.includes('virtual:superdebug-compat')) {
        let transformedCode = code;

        // Replace all possible import patterns
        const importPatterns = [
          /import\s+.*SuperDebug.*from\s+['"]['"]sveltekit-superforms[^'"]*['"]['"];?/gi,
          /import\s+.*SuperDebug.*from\s+['"]['"]sveltekit-superforms\/dist\/client[^'"]*['"]['"];?/gi,
          /import\s+.*SuperDebug.*from\s+['"]['"]sveltekit-superforms\/client[^'"]*['"]['"];?/gi,
          /import\s+\{\s*SuperDebug\s*\}\s+from\s+['"]['"]sveltekit-superforms[^'"]*['"]['"];?/gi,
        ];

        importPatterns.forEach(pattern => {
          transformedCode = transformedCode.replace(pattern, '// SuperDebug import replaced for Svelte 5 compatibility');
        });

        // Replace SuperDebug component usage
        transformedCode = transformedCode.replace(/<SuperDebug\s+([^>]*)\/>/g, '<!-- SuperDebug removed for Svelte 5 compatibility -->');
        transformedCode = transformedCode.replace(/<SuperDebug\s+([^>]*)>.*?<\/SuperDebug>/gs, '<!-- SuperDebug removed for Svelte 5 compatibility -->');

        if (transformedCode !== code) {
          console.log(`🔄 Transformed SuperDebug usage in: ${id}`);
          return transformedCode;
        }
      }
    }
  };
}

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
    } catch (error: any) {
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
  } catch (error: any) {
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
  } catch (error: any) {
    console.error(`❌ Failed to find available port: ${error}`);
    availablePort = preferredPort; // Fallback to default
  }

  // Enhanced logging configuration
  const logLevel = process.env.VITE_LOG_LEVEL || 'info';
  const isVerbose = process.env.VITE_DEBUG === 'true';

  if (isVerbose) {
    console.log(`🔍 Debug mode enabled - comprehensive error logging active`);
    console.log(`📊 Log level: ${logLevel}`);
  }

  // Load dynamic port configuration for proxy
  const dynamicPorts = await loadDynamicPorts();

  return {
    logLevel: logLevel as 'info' | 'warn' | 'error' | 'silent',
    clearScreen: false,
  plugins: [
    superFormsCompat(),
    // Alias Drizzle node-postgres imports to a local shim that re-exports
    // the postgres-js adapter. This prevents runtime adapter mismatches
    // when some modules import 'drizzle-orm/node-postgres' while the
    // application uses the 'postgres' (postgres-js) client.
    {
      name: 'drizzle-node-postgres-alias',
      configResolved(config: any) {
        config.resolve = config.resolve || {};
        config.resolve.alias = config.resolve.alias || {};
        config.resolve.alias['drizzle-orm/node-postgres'] = resolve(__dirname, 'src/lib/shims/drizzle-node-postgres.ts');
      }
    },
    {
      name: 'superforms-transform',
      transform(code: string, id: string) {
        // Transform any remaining SuperDebug imports during build
        if (code.includes('SuperDebug') && code.includes('sveltekit-superforms')) {
          return code.replace(
            /import\s+.*SuperDebug.*from\s+['"]sveltekit-superforms.*['"];?/g,
            "// SuperDebug import replaced for Svelte 5 compatibility"
          );
        }
      }
    },
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
      'melt-ui',
      'bits-ui',
      'camelcase',
      'decamelize'
    ],
    exclude: [
      '@langchain/community',
      '@langchain/anthropic',
      '@langchain/google-genai',
      '@langchain/core',
      'ioredis',
      'drizzle-orm',
      'postgres',
      '@qdrant/js-client-rest',
      'sveltekit-superforms/dist/client/SuperDebug.svelte'
    ],

    // ESBuild options for dependency pre-bundling
    esbuildOptions: {
      plugins: [
        {
          name: 'fix-camelcase-import',
          setup(build) {
            // Fix camelcase default import issue for LangChain compatibility
            build.onResolve({ filter: /^camelcase$/ }, (args) => {
              return {
                path: args.path,
                namespace: 'fix-camelcase'
              };
            });

            // Also handle camelcase imports from nested node_modules
            build.onResolve({ filter: /\/camelcase\/index\.js$/ }, (args) => {
              return {
                path: args.path,
                namespace: 'fix-camelcase'
              };
            });

            build.onLoad({ filter: /.*/, namespace: 'fix-camelcase' }, async () => {
              return {
                contents: `
                  // Enhanced camelcase compatibility for LangChain
                  import camelCase from 'camelcase';
                  export default camelCase;
                  export { camelCase };
                  export * from 'camelcase';
                `,
                loader: 'js',
              };
            });
          }
        },
        {
          name: 'fix-langchain-camelcase',
          setup(build) {
            // Specifically handle LangChain's camelcase imports
            build.onResolve({
              filter: /node_modules\/@langchain\/core\/node_modules\/camelcase/
            }, (args) => {
              console.log(`🔧 Fixing LangChain camelcase import: ${args.path}`);
              return {
                path: require.resolve('camelcase'),
                external: false
              };
            });
          }
        }
      ]
    },

    // Force pre-bundling for better performance
    force: true
  },

  // Path resolution
  resolve: {
    alias: {
      // Comprehensive SuperDebug compatibility with Svelte 5
      'sveltekit-superforms/dist/client/SuperDebug.svelte': 'virtual:superdebug-compat',
      'sveltekit-superforms/client/SuperDebug.svelte': 'virtual:superdebug-compat',
      'SuperDebug.svelte': 'virtual:superdebug-compat',

      $lib: resolve('./src/lib'),
      $components: resolve('./src/lib/components'),
      $stores: resolve('./src/lib/stores'),
      $utils: resolve('./src/utils'),
      $database: resolve('./src/lib/database'),
      $agents: resolve('./src/lib/agents'),
  $legal: resolve('./src/lib/legal'),
  '@shared': resolve('../shared'),
  '@text': resolve('../shared/text'),
        // Fix camelcase version conflict for LangChain compatibility by redirecting
        // LangChain's nested camelcase import to our ESM shim which provides a default export.
        'camelcase': resolve('./src/shims/camelcase-compat.mjs'),
        // Additional camelcase path resolutions used by @langchain/core
        '/node_modules/@langchain/core/node_modules/camelcase': resolve('./src/shims/camelcase-compat.mjs'),
        '/node_modules/@langchain/core/node_modules/camelcase/index.js': resolve('./src/shims/camelcase-compat.mjs')
      },
      // Fix ESM module compatibility issues
      conditions: ['import', 'module', 'browser', 'default']
    },

    // Fix for LangChain ESM module compatibility
    ssr: {
      noExternal: [
        '@langchain/core',
        '@langchain/community',
        '@langchain/anthropic',
        'camelcase'
      ],
      // Ensure camelcase is treated as ESM in SSR context
      external: []
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

    // Environment variables and global fixes
  define: {
    global: 'globalThis', // Fix for Node.js global in browser
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
