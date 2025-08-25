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
			include: ['process', 'buffer', 'util', 'stream', 'events'],
			globals: {
				Buffer: true,
				global: true,
				process: true,
			},
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
			// Force fabric to use the browser-specific build
			'fabric': path.resolve('./node_modules/fabric/dist/fabric.js')
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
		host: '0.0.0.0',
		hmr: { 
			port: 24678, 
			clientPort: 24678 
		},
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
			'@xenova/transformers'
		],
		exclude: [
			'@tauri-apps/api', // Tauri should not be optimized
			'pdf-lib', // Problematic module
			'@langchain/core', // Missing exports
			'@langchain/community',
			'canvas' // Native module issues (fabric.js now properly configured)
		]
	},
	
	ssr: {
		noExternal: ['bits-ui', 'melt'],
		external: ['fabric', 'canvas'] // Exclude problematic canvas modules from SSR
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
			}
		}
		,
		reportCompressedSize: true,
		chunkSizeWarningLimit: 1000
	}
});