import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import UnoCSS from '@unocss/vite';
import { nodePolyfills } from 'vite-plugin-node-polyfills';
import path from 'path';

export default defineConfig({
	plugins: [
		sveltekit(),
		UnoCSS(),
		nodePolyfills({
			// Enable polyfills for Node.js globals and modules
			include: ['process', 'buffer', 'util', 'stream', 'events', 'crypto'],
			exclude: ['fs', 'dns', 'os', 'os-browserify'], // Explicitly exclude problematic modules
			globals: {
				Buffer: true,
				global: true,
				process: true,
			},
			protocolImports: true
		})
	],

	// Enhanced logging configuration
	logLevel: 'info', // 'error' | 'warn' | 'info' | 'silent'

	resolve: {
		alias: {
			$lib: path.resolve('./src/lib'),
			$components: path.resolve('./src/lib/components'),
			$services: path.resolve('./src/lib/services'),
			$types: path.resolve('./src/lib/types'),
			// Shim lucide-svelte during Svelte 5 migration (avoids $$props usage in package)
			'lucide-svelte': path.resolve('./src/lib/shims/lucide-shim'),
				// Shim sveltekit-superforms SuperDebug (package ships a duplicate top-level <script> during build)
				'sveltekit-superforms/dist/client/SuperDebug.svelte': path.resolve('./src/lib/shims/superforms/SuperDebug.svelte'),
			// Force fabric to use the browser-specific build
			'fabric': path.resolve('./node_modules/fabric/dist/fabric.js'),
			// Browser-compatible shims for Node.js modules
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
		hmr: true,
		// Enhanced proxy logging
		proxy: {
			'/health': {
				target: 'http://localhost:8080',
				changeOrigin: true,
				configure: (proxy, options) => {
					proxy.on('proxyReq', (proxyReq, req, res) => {
						console.log(`[PROXY] ${req.method} ${req.url} -> ${options.target}`);
					});
					proxy.on('proxyRes', (proxyRes, req, res) => {
						console.log(`[PROXY] ${req.method} ${req.url} <- ${proxyRes.statusCode}`);
					});
				}
			},
			'/api/v1': {
				target: 'http://localhost:8080',
				changeOrigin: true,
				configure: (proxy, options) => {
					proxy.on('proxyReq', (proxyReq, req, res) => {
						console.log(`[PROXY] ${req.method} ${req.url} -> ${options.target}`);
					});
					proxy.on('proxyRes', (proxyRes, req, res) => {
						console.log(`[PROXY] ${req.method} ${req.url} <- ${proxyRes.statusCode}`);
					});
				}
			}
		}
	},
	optimizeDeps: {
		include: [
			'socket.io-client',
			// Bits UI and Melt UI dependencies
			'bits-ui',
			'melt',
			// Browser polyfills for Node.js modules
			'fuse.js',
			'zod',
			'class-variance-authority',
			'clsx',
			'tailwind-merge',
			// Vector/AI dependencies
			'@xenova/transformers',
			// Force camelcase to be optimized to fix import issues
			'camelcase'
		],
		exclude: [
			'@tauri-apps/api', // Tauri should not be optimized
			'pdf-lib', // Problematic module
			'@xenova/transformers' // Uses Node.js fs module
		]
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
		] // Exclude problematic modules from SSR
	},

	// Enhanced build configuration for browser compatibility
	build: {
		target: ['es2020', 'chrome80', 'firefox78', 'safari14'],
		modulePreload: { polyfill: true },
		rollupOptions: {
			output: {
				// Separate chunks for better caching
				manualChunks: {
					'bits-ui': ['bits-ui'],
					'melt-ui': ['melt'],
					'search': ['fuse.js'],
					'vector': ['@xenova/transformers'],
					'ai': ['@langchain/core', '@langchain/community'],
					'utils': ['zod', 'clsx', 'tailwind-merge', 'class-variance-authority']
				}
			},
			// Fix CommonJS/ESM compatibility for camelcase
			external: (id) => {
				// Don't externalize camelcase - let Vite handle the conversion
				if (id === 'camelcase') return false;
				return false;
			}
		}
		,
		reportCompressedSize: true,
		chunkSizeWarningLimit: 1000
	}
});