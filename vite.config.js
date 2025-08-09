import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
	plugins: [sveltekit()],
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
		strictPort: false
	},
	optimizeDeps: {
		include: ['fabric', 'pdf-lib', 'socket.io-client']
	}
});