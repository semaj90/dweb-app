import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';
import { resolve } from 'path';

export default defineConfig({
  plugins: [sveltekit()],
  
  test: {
    // Integration test configuration
    name: 'integration',
    include: ['src/lib/tests/integration/**/*.test.ts'],
    exclude: ['node_modules', 'build', 'dist'],
    
    // Test environment setup
    environment: 'jsdom',
    globals: true,
    setupFiles: ['src/lib/tests/setup/integration-setup.ts'],
    
    // Timeouts for async operations
    testTimeout: 10000,
    hookTimeout: 10000,
    
    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: [
        'src/lib/state/**/*.ts',
        'src/lib/components/**/*.svelte',
        'src/routes/**/*.ts',
        'src/routes/**/*.svelte',
        'src/hooks.server.ts'
      ],
      exclude: [
        'src/lib/tests/**',
        'src/lib/types/**',
        '**/*.d.ts',
        'build/**',
        'dist/**'
      ],
      thresholds: {
        global: {
          branches: 70,
          functions: 70,
          lines: 70,
          statements: 70
        }
      }
    },
    
    // Mock configuration
    deps: {
      inline: [
        '@testing-library/svelte',
        '@xstate/svelte'
      ]
    },
    
    // Browser-like environment setup
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: true
      }
    }
  },
  
  resolve: {
    alias: {
      $lib: resolve('./src/lib'),
      $app: resolve('./node_modules/@sveltejs/kit/src/runtime/app'),
      $routes: resolve('./src/routes'),
      $hooks: resolve('./src'),
      $types: resolve('./src/lib/types')
    }
  },
  
  define: {
    // Test environment variables
    'import.meta.env.DEV': true,
    'import.meta.env.PROD': false,
    'import.meta.env.MODE': '"test"',
    'import.meta.vitest': true
  }
});