import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';

// Consolidated vitest config (reduced for initial auth & embedding tests)
export default defineConfig({
  plugins: [sveltekit()],
  test: {
    globals: true,
    environment: 'node',
    include: [
      'src/lib/**/__tests__/**/*.{test,spec}.ts',
      // Keep only lightweight integration smoke for now
      'src/lib/tests/integration/register-login-embed-search.test.ts',
      'src/lib/tests/integration/system-metrics.test.ts',
      'src/lib/tests/integration/comprehensive-summary.test.ts'
    ],
    exclude: [
      'src/lib/tests/integration/service-coordinator-integration.test.ts',
      'src/lib/tests/integration/ssr-session-integrity.test.ts',
      'src/lib/tests/integration/xstate-machine*.test.ts'
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary'],
      reportsDirectory: './coverage'
    }
  }
});
