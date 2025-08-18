import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5174,
		host: true,
		cors: true,
		hmr: {
			port: 5174
		}
	},
	preview: {
		port: 5174,
		host: true
	},
	optimizeDeps: {
		include: [
			'@xenova/transformers',
			'onnxruntime-web',
			'@tensorflow/tfjs',
			'ws'
		]
	},
	ssr: {
		noExternal: ['ws', 'ioredis']
	}
});
