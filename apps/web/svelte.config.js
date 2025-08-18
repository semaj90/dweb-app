import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			// GPU-optimized production build
			precompress: true,
			envPrefix: 'DEEDS_'
		}),
		alias: {
			'$lib': 'src/lib',
			'$components': 'src/lib/components',
			'$stores': 'src/lib/stores',
			'$utils': 'src/lib/utils',
			'$types': 'src/lib/types',
			'$api': 'src/lib/api'
		},
		// Enhanced for real-time features
		csp: {
			directives: {
				'script-src': ['self', 'wasm-unsafe-eval'],
				'worker-src': ['self', 'blob:'],
				'connect-src': ['self', 'ws:', 'wss:']
			}
		},
		serviceWorker: {
			register: false // Disable for real-time features
		}
	}
};

export default config;
