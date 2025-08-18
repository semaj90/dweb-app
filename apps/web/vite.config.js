import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: true,
		port: 5173,
		strictPort: true,
		hmr: {
			clientPort: 5173
		}
	},
	optimizeDeps: {
		exclude: ['@grpc/grpc-js', '@grpc/proto-loader']
	},
	build: {
		target: 'esnext',
		minify: 'esbuild',
		rollupOptions: {
			external: ['@grpc/grpc-js', '@grpc/proto-loader']
		}
	},
	define: {
		global: 'globalThis'
	}
});
