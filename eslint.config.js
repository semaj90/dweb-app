// Root ESLint flat config for SvelteKit frontend and Node.js backend.
// (Original explanatory text removed for valid JavaScript.)

import js from "@eslint/js";
import js from "@eslint/js";
import svelte from "eslint-plugin-svelte";
import ts from "typescript-eslint";
import globals from "globals";
import svelteParser from "svelte-eslint-parser";

export default [
  {
    files: ['sveltekit-frontend/**/*.ts'],
    extends: [ts.configs.recommended],
    languageOptions: {
      parser: ts.parser,
      parserOptions: {
        project: './sveltekit-frontend/tsconfig.json',
        ecmaVersion: 2022,
        sourceType: 'module',
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      ...ts.configs.recommended.rules,
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'no-undef': 'off',
    },
  },
  {
    files: ['sveltekit-frontend/**/*.svelte'],
    extends: [
      svelte.configs['flat/recommended'],
      svelte.configs['flat/prettier'],
      ts.configs.recommended,
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        parser: ts.parser,
        project: './sveltekit-frontend/tsconfig.json',
        ecmaVersion: 2022,
        sourceType: 'module',
        extraFileExtensions: ['.svelte'],
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      ...svelte.configs.recommended.rules,
      ...ts.configs.recommended.rules,
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'no-undef': 'off',
      'svelte/no-at-html-tags': 'off',
      'svelte/no-unused-export-let': 'off',
      'svelte/valid-compile': 'off',
    },
  },
  {
    files: ['sveltekit-frontend/**/*.js', 'sveltekit-frontend/**/*.mjs'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      'no-unused-vars': 'off',
      'no-undef': 'off',
    },
  },
  {
    files: ['mcp-servers/**/*.js'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        ...globals.node,
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      'no-unused-vars': 'off',
      'no-undef': 'off',
    },
  },

  // Ignored directories
  {
    ignores: ['.vscode/']
  }
];