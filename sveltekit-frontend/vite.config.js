import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	build: {
		rollupOptions: {
			// Removed external: ['@prisma/client', '$/app.css'] as they are not needed and cause issues.
			// @prisma/client is not used, and $/app.css should be bundled.
		},
	},
});
