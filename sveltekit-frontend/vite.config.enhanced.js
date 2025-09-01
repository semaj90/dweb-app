import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import UnoCSS from '@unocss/vite';
import { nodePolyfills } from 'vite-plugin-node-polyfills';
import path from 'path';
import fs from 'fs';

// Custom error logging plugin
function errorLoggingPlugin() {
	const logFile = path.resolve('./logs/vite-errors.log');
	const logDir = path.dirname(logFile);
	
	// Ensure log directory exists
	if (!fs.existsSync(logDir)) {
		fs.mkdirSync(logDir, { recursive: true });
	}
	
	// Initialize log file
	fs.writeFileSync(logFile, `=== Vite Error Log Started: ${new Date().toISOString()} ===\n\n`);
	
	return {
		name: 'vite-error-logger',
		
		// Log configuration errors
		configResolved(config) {
			console.log('[Vite] Configuration resolved successfully');
			fs.appendFileSync(logFile, `[CONFIG] Resolved at ${new Date().toISOString()}\n`);
		},
		
		// Log build errors
		buildStart() {
			console.log('[Vite] Build started');
			fs.appendFileSync(logFile, `[BUILD] Started at ${new Date().toISOString()}\n`);
		},
		
		// Log module resolution errors
		resolveId(source, importer) {
			if (source.includes('error') || source.includes('fail')) {
				const msg = `[RESOLVE] Potential issue: ${source} from ${importer}\n`;
				fs.appendFileSync(logFile, msg);
			}
		},
		
		// Log transformation errors
		transform(code, id) {
			if (id.includes('.svelte') || id.includes('.ts')) {
				try {
					// Check for common errors
					if (code.includes('$$props') && !code.includes('// @migration-skip')) {
						const msg = `[TRANSFORM] Warning: $$props usage in ${id}\n`;
						console.warn(msg);
						fs.appendFileSync(logFile, msg);
					}
					if (code.includes('$$restProps') && !code.includes('// @migration-skip')) {
						const msg = `[TRANSFORM] Warning: $$restProps usage in ${id}\n`;
						console.warn(msg);
						fs.appendFileSync(logFile, msg);
					}
				} catch (e) {
					const msg = `[TRANSFORM] Error in ${id}: ${e.message}\n`;
					console.error(msg);
					fs.appendFileSync(logFile, msg);
				}
			}
		},
		
		// Log HMR errors
		handleHotUpdate({ file, server, modules }) {
			if (modules.length === 0) {
				const msg = `[HMR] Warning: No modules to update for ${file}\n`;
				console.warn(msg);
				fs.appendFileSync(logFile, msg);
			}
		},
		
		// Log server errors
		configureServer(server) {
			// Enhanced error handling
			server.middlewares.use((err, req, res, next) => {
				const msg = `[SERVER] Error: ${err.message}\n  URL: ${req.url}\n  Stack: ${err.stack}\n`;
				console.error(msg);
				fs.appendFileSync(logFile, msg);
				next(err);
			});
			
			// Log WebSocket errors
			server.ws.on('error', (error) => {
				const msg = `[WS] Error: ${error.message}\n`;
				console.error(msg);
				fs.appendFileSync(logFile, msg);
			});
			
			// Log server startup
			server.httpServer?.once('listening', () => {
				const address = server.httpServer.address();
				const msg = `[SERVER] Listening on http://localhost:${address.port}\n`;
				console.log(msg);
				fs.appendFileSync(logFile, msg);
			});
		}
	};
}

// Service health check plugin
function serviceHealthPlugin() {
	return {
		name: 'service-health-check',
		
		async configureServer(server) {
			// Add health check endpoint
			server.middlewares.use('/api/vite-health', (req, res) => {
				const services = {
					vite: 'running',
					hmr: server.ws ? 'active' : 'inactive',
					timestamp: new Date().toISOString()
				};
				res.setHeader('Content-Type', 'application/json');
				res.end(JSON.stringify(services));
			});
			
			// Check external services on startup
			const checkServices = async () => {
				const services = [
					{ name: 'PostgreSQL', port: 5432 },
					{ name: 'MinIO', port: 9000 },
					{ name: 'Ollama', port: 11434 },
					{ name: 'Qdrant', port: 6333 },
					{ name: 'Enhanced RAG', port: 8094 },
					{ name: 'GPU Orchestrator', port: 8231 }
				];
				
				console.log('\n=== Service Status Check ===');
				for (const service of services) {
					try {
						const response = await fetch(`http://localhost:${service.port}/health`, {
							signal: AbortSignal.timeout(1000)
						}).catch(() => null);
						
						if (response?.ok) {
							console.log(`✅ ${service.name} (${service.port}): Running`);
						} else {
							console.log(`⚠️  ${service.name} (${service.port}): Not responding`);
						}
					} catch (error) {
						console.log(`❌ ${service.name} (${service.port}): Failed`);
					}
				}
				console.log('===========================\n');
			};
			
			// Check services after server starts
			setTimeout(checkServices, 2000);
		}
	};
}

export default defineConfig({
	plugins: [
		sveltekit(),
		UnoCSS(),
		nodePolyfills({
			// Enable polyfills for Node.js globals and modules
			include: ['process', 'buffer', 'util', 'stream', 'events', 'crypto'],
			exclude: ['fs', 'dns', 'os', 'os-browserify'],
			globals: {
				Buffer: true,
				global: true,
				process: true,
			},
			protocolImports: true
		}),
		errorLoggingPlugin(),
		serviceHealthPlugin()
	],

	// Maximum logging
	logLevel: 'info',
	
	// Custom logger for enhanced error tracking
	customLogger: {
		info(msg) {
			console.log(`[INFO] ${msg}`);
		},
		warn(msg) {
			console.warn(`[WARN] ${msg}`);
			fs.appendFileSync('./logs/vite-errors.log', `[WARN] ${msg}\n`);
		},
		error(msg) {
			console.error(`[ERROR] ${msg}`);
			fs.appendFileSync('./logs/vite-errors.log', `[ERROR] ${msg}\n`);
		}
	},

	resolve: {
		alias: {
			$lib: path.resolve('./src/lib'),
			$components: path.resolve('./src/lib/components'),
			$services: path.resolve('./src/lib/services'),
			$types: path.resolve('./src/lib/types'),
			'lucide-svelte': path.resolve('./src/lib/shims/lucide-shim'),
			'sveltekit-superforms/dist/client/SuperDebug.svelte': path.resolve('./src/lib/shims/superforms/SuperDebug.svelte'),
			'fabric': path.resolve('./node_modules/fabric/dist/fabric.js'),
			'fs': path.resolve('./src/lib/shims/fs-browser-shim.js'),
			'dns': path.resolve('./src/lib/shims/dns-browser-shim.js'),
			'ioredis': path.resolve('./src/lib/shims/ioredis-browser-shim.js'),
			'os': path.resolve('./src/lib/shims/os-browser-shim.js'),
			'os-browserify': path.resolve('./src/lib/shims/os-browser-shim.js')
		}
	},

	// Define global constants for browser compatibility
	define: {
		global: 'globalThis',
		'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
		__DEV__: JSON.stringify(process.env.NODE_ENV !== 'production')
	},
	
	server: {
		port: 5173,
		strictPort: false,
		host: 'localhost',
		hmr: {
			overlay: true,
			port: 5173
		},
		
		// Watch for errors
		watch: {
			usePolling: true,
			interval: 100
		},
		
		// Enhanced proxy with error handling
		proxy: {
			// Production upload endpoint
			'/api/production-upload': {
				target: 'http://localhost:5173',
				changeOrigin: false,
				configure: (proxy, options) => {
					proxy.on('error', (err, req, res) => {
						console.error(`[PROXY ERROR] ${req.url}:`, err.message);
						fs.appendFileSync('./logs/vite-errors.log', `[PROXY ERROR] ${req.url}: ${err.message}\n`);
					});
				}
			},
			
			// MinIO proxy
			'/minio': {
				target: 'http://localhost:9000',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/minio/, ''),
				configure: (proxy) => {
					proxy.on('error', (err) => {
						console.warn('[MinIO Proxy] Error:', err.message);
					});
				}
			},
			
			// Ollama proxy
			'/ollama': {
				target: 'http://localhost:11434',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/ollama/, ''),
				configure: (proxy) => {
					proxy.on('error', (err) => {
						console.warn('[Ollama Proxy] Error:', err.message);
					});
				}
			},
			
			// Enhanced RAG proxy
			'/rag': {
				target: 'http://localhost:8094',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/rag/, ''),
				configure: (proxy) => {
					proxy.on('error', (err) => {
						console.warn('[RAG Proxy] Error:', err.message);
					});
				}
			},
			
			// GPU service proxy
			'/gpu': {
				target: 'http://localhost:8231',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/gpu/, ''),
				configure: (proxy) => {
					proxy.on('error', (err) => {
						console.warn('[GPU Proxy] Error:', err.message);
					});
				}
			}
		}
	},
	
	optimizeDeps: {
		include: [
			'socket.io-client',
			'bits-ui',
			'melt',
			'fuse.js',
			'zod',
			'class-variance-authority',
			'clsx',
			'tailwind-merge',
			'@xenova/transformers',
			'camelcase'
		],
		exclude: [
			'@tauri-apps/api',
			'pdf-lib',
			'@xenova/transformers'
		],
		
		// Force re-optimization on error
		force: process.env.VITE_FORCE_OPTIMIZE === 'true'
	},

	ssr: {
		noExternal: ['bits-ui', 'melt'],
		external: [
			'fabric', 
			'canvas', 
			'fs', 
			'dns', 
			'lokijs',
			'ioredis',
			'crypto-browserify',
			'os-browserify',
			'os'
		]
	},

	build: {
		target: ['es2020', 'chrome80', 'firefox78', 'safari14'],
		modulePreload: { polyfill: true },
		sourcemap: true,
		minify: 'terser',
		
		// Better error reporting
		terserOptions: {
			compress: {
				drop_console: false,
				drop_debugger: false
			}
		},
		
		rollupOptions: {
			output: {
				manualChunks: {
					'bits-ui': ['bits-ui'],
					'melt-ui': ['melt'],
					'search': ['fuse.js'],
					'vector': ['@xenova/transformers'],
					'ai': ['@langchain/core', '@langchain/community'],
					'utils': ['zod', 'clsx', 'tailwind-merge', 'class-variance-authority']
				}
			},
			
			// Enhanced error handling
			onwarn(warning, warn) {
				// Log all warnings to file
				const msg = `[ROLLUP WARN] ${warning.code}: ${warning.message}\n`;
				fs.appendFileSync('./logs/vite-errors.log', msg);
				
				// Skip certain warnings
				if (warning.code === 'MODULE_LEVEL_DIRECTIVE') return;
				if (warning.code === 'CIRCULAR_DEPENDENCY') return;
				
				// Default warning handler
				warn(warning);
			}
		},
		
		reportCompressedSize: true,
		chunkSizeWarningLimit: 1000
	},
	
	// Preview server configuration
	preview: {
		port: 4173,
		strictPort: false,
		host: 'localhost'
	}
});
