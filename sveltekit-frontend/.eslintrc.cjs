/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:svelte/recommended'
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  overrides: [
    {
      files: ['*.svelte'],
      parser: 'svelte-eslint-parser',
      parserOptions: {
        parser: '@typescript-eslint/parser',
      },
    },
  ],
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
  },
  ignorePatterns: [
    '*.md', 
    'dist', 
    'build', 
    '.svelte-kit', 
    'node_modules',
    'js_tests/**/*',
    'playwright-report/**/*',
    '*.js',
    'drizzle.config.ts',
    'src/lib/**/*',
    'src/hooks.client.ts',
    'src/routes/**/*',
    'src/service-worker.ts',
    'src/test-setup.ts',
    'tests/**/*',
    'scripts/**/*',
    '*.config.ts',
    'vite.config.ts',
    'vitest.config.ts',
    'uno.config.ts',
    'unocss.config.ts'
  ],
};