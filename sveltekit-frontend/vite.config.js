import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
	plugins: [sveltekit()],
	
	// Enhanced logging configuration
	logLevel: 'info', // 'error' | 'warn' | 'info' | 'silent'
	
	resolve: {
		alias: {
			$lib: path.resolve('./src/lib'),
			$components: path.resolve('./src/lib/components'),
			$services: path.resolve('./src/lib/services'),
			$types: path.resolve('./src/lib/types')
		}
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
		include: ['fabric', 'pdf-lib', 'socket.io-client']
	},
	
	// Build logging
	build: {
		reportCompressedSize: true,
		chunkSizeWarningLimit: 1000
	}
});