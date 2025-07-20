#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, 'sveltekit-frontend');

console.log('🔧 Starting comprehensive project fixes and setup...');

// Helper functions
function readFile(filePath) {
    try {
        return fs.readFileSync(filePath, 'utf8');
    } catch (error) {
        console.warn(`⚠️  Could not read ${filePath}: ${error.message}`);
        return null;
    }
}

function writeFile(filePath, content) {
    try {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`✅ Updated: ${path.relative(projectRoot, filePath)}`);
        return true;
    } catch (error) {
        console.error(`❌ Error writing ${filePath}: ${error.message}`);
        return false;
    }
}

function ensureDir(dirPath) {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`📁 Created directory: ${path.relative(projectRoot, dirPath)}`);
    }
}

// Fix 1: Ensure proper type definitions
function createGlobalTypes() {
    const globalTypesPath = path.join(projectRoot, 'src/lib/types/global.d.ts');
    ensureDir(path.dirname(globalTypesPath));

    const globalTypes = `
// Global type definitions for the project

declare global {
  namespace App {
    interface Error {
      message: string;
      code?: string;
    }
    
    interface Locals {
      user: User | null;
      session: Session | null;
    }
    
    interface PageData {}
    
    interface Platform {}
  }
}

// User type definition
export interface User {
  id: string;
  email: string;
  name: string;
  firstName?: string;
  lastName?: string;
  avatarUrl?: string;
  role: 'user' | 'admin' | 'prosecutor' | 'investigator';
  isActive: boolean;
  emailVerified?: boolean;
  createdAt: Date;
  updatedAt: Date;
}

// Session type definition
export interface Session {
  id: string;
  userId: string;
  fresh: boolean;
  expiresAt: Date;
}

// Component props for UI components
export interface SelectContext {
  selected: import('svelte/store').Writable<any>;
  open: import('svelte/store').Writable<boolean>;
  onSelect: (value: any) => void;
  onToggle: () => void;
}

// AI Service types
export interface AIResponse {
  content: string;
  confidence?: number;
  tokens?: number;
  model?: string;
}

export interface EmbeddingResult {
  embedding: number[];
  tokens: number;
  model: string;
}

// Vector search types
export interface VectorSearchResult {
  id: string;
  content: string;
  similarity: number;
  metadata?: Record<string, any>;
}

export interface SearchOptions {
  limit?: number;
  threshold?: number;
  caseId?: string;
  contentType?: string;
}

export {};
`;

    writeFile(globalTypesPath, globalTypes);
}

// Fix 2: Update app.d.ts
function updateAppDts() {
    const appDtsPath = path.join(projectRoot, 'src/app.d.ts');
    const appDtsContent = `
// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
  namespace App {
    interface Error {
      message: string;
      code?: string;
    }
    
    interface Locals {
      user: import('./lib/types/global').User | null;
      session: import('./lib/types/global').Session | null;
    }
    
    interface PageData {}
    
    interface Platform {}
  }
}

export {};
`;

    writeFile(appDtsPath, appDtsContent);
}

// Fix 3: Create missing component types
function createSelectTypes() {
    const selectTypesPath = path.join(projectRoot, 'src/lib/components/ui/select/types.ts');
    ensureDir(path.dirname(selectTypesPath));

    const selectTypes = `
import type { Writable } from 'svelte/store';

export interface SelectContext {
  selected: Writable<any>;
  open: Writable<boolean>;
  onSelect: (value: any) => void;
  onToggle: () => void;
}

export interface SelectItemProps {
  value: any;
  class_?: string;
  selected?: boolean;
}

export interface SelectProps {
  value?: any;
  onValueChange?: (value: any) => void;
  disabled?: boolean;
  class_?: string;
}
`;

    writeFile(selectTypesPath, selectTypes);
}

// Fix 4: Update tsconfig.json for better type checking
function updateTsConfig() {
    const tsconfigPath = path.join(projectRoot, 'tsconfig.json');
    let content = readFile(tsconfigPath);
    if (!content) return;

    const tsconfig = JSON.parse(content);
    
    // Update compiler options for better type safety
    tsconfig.compilerOptions = {
      ...tsconfig.compilerOptions,
      strict: true,
      noImplicitAny: false,
      noImplicitReturns: true,
      noImplicitThis: true,
      noUnusedLocals: false,
      noUnusedParameters: false,
      exactOptionalPropertyTypes: false,
      noUncheckedIndexedAccess: false,
      skipLibCheck: true,
    };

    // Add paths for better imports
    tsconfig.compilerOptions.paths = {
      "$lib": ["./src/lib"],
      "$lib/*": ["./src/lib/*"],
      "$components": ["./src/lib/components"],
      "$components/*": ["./src/lib/components/*"],
      "$types": ["./src/lib/types"],
      "$types/*": ["./src/lib/types/*"],
    };

    writeFile(tsconfigPath, JSON.stringify(tsconfig, null, 2));
}

// Fix 5: Create/update vite.config.ts
function updateViteConfig() {
    const viteConfigPath = path.join(projectRoot, 'vite.config.ts');
    
    const viteConfig = `
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import UnoCSS from 'unocss/vite';

export default defineConfig({
  plugins: [
    UnoCSS(),
    sveltekit()
  ],
  
  define: {
    // Add any global defines here
  },
  
  server: {
    fs: {
      allow: ['..']
    }
  },
  
  optimizeDeps: {
    include: [
      'lucide-svelte',
      '@tiptap/core',
      '@tiptap/starter-kit',
      'fabric'
    ]
  },
  
  build: {
    target: 'esnext',
    minify: 'terser'
  }
});
`;

    writeFile(viteConfigPath, viteConfig);
}

// Fix 6: Update svelte.config.js
function updateSvelteConfig() {
    const svelteConfigPath = path.join(projectRoot, 'svelte.config.js');
    
    const svelteConfig = `
import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://kit.svelte.dev/docs/integrations#preprocessors
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter(),
    
    alias: {
      $lib: 'src/lib',
      $components: 'src/lib/components',
      $types: 'src/lib/types'
    },
    
    files: {
      assets: 'static',
      hooks: {
        client: 'src/hooks.client.ts',
        server: 'src/hooks.server.ts'
      },
      lib: 'src/lib',
      params: 'src/params',
      routes: 'src/routes',
      serviceWorker: 'src/service-worker.ts',
      appTemplate: 'src/app.html',
      errorTemplate: 'src/error.html'
    }
  },

  vitePlugin: {
    experimental: {
      inspector: {
        holdMode: true
      }
    }
  }
};

export default config;
`;

    writeFile(svelteConfigPath, svelteConfig);
}

// Fix 7: Create environment setup
function createEnvSetup() {
    const envExamplePath = path.join(projectRoot, '.env.example');
    
    const envExample = `
# Database Configuration
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/prosecutor_db"
DATABASE_AUTH_TOKEN=""

# AI Service Configuration
OPENAI_API_KEY=""
OLLAMA_BASE_URL="http://localhost:11434"
TAURI_LLM_ENABLED=true

# Authentication
AUTH_SECRET="your-super-secret-auth-key-here"
JWT_SECRET="your-jwt-secret-here"

# Application Configuration
PUBLIC_APP_NAME="Legal AI Assistant"
PUBLIC_APP_VERSION="2.0.0"
PUBLIC_ENABLE_REGISTRATION=true

# Vector Search Configuration
QDRANT_URL="http://localhost:6333"
ENABLE_VECTOR_SEARCH=true

# Development
NODE_ENV=development
PUBLIC_DEBUG=false
`;

    if (!fs.existsSync(envExamplePath)) {
        writeFile(envExamplePath, envExample);
    }

    const envPath = path.join(projectRoot, '.env');
    if (!fs.existsSync(envPath)) {
        writeFile(envPath, envExample);
        console.log('📝 Created .env file - please update with your actual values');
    }
}

// Fix 8: Create basic error boundary
function createErrorBoundary() {
    const errorBoundaryPath = path.join(projectRoot, 'src/routes/+error.svelte');
    ensureDir(path.dirname(errorBoundaryPath));
    
    const errorBoundary = `
<script lang="ts">
  import { page } from '$app/stores';
  import { dev } from '$app/environment';
  
  $: error = $page.error;
  $: status = $page.status;
</script>

<svelte:head>
  <title>Error {status}</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
  <div class="sm:mx-auto sm:w-full sm:max-w-md">
    <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
      <div class="text-center">
        <h1 class="text-4xl font-bold text-gray-900 mb-4">{status}</h1>
        <h2 class="text-lg font-medium text-gray-700 mb-6">
          {#if status === 404}
            Page not found
          {:else if status === 500}
            Internal server error
          {:else}
            Something went wrong
          {/if}
        </h2>
        
        {#if error?.message}
          <p class="text-sm text-gray-600 mb-6">{error.message}</p>
        {/if}
        
        {#if dev && error}
          <details class="text-left mt-6">
            <summary class="cursor-pointer text-sm font-medium text-gray-700">
              Error Details (Development)
            </summary>
            <pre class="mt-2 text-xs bg-gray-100 p-4 rounded overflow-auto">
              {JSON.stringify(error, null, 2)}
            </pre>
          </details>
        {/if}
        
        <div class="mt-6">
          <a 
            href="/" 
            class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition-colors"
          >
            Go Home
          </a>
        </div>
      </div>
    </div>
  </div>
</div>
`;

    if (!fs.existsSync(errorBoundaryPath)) {
        writeFile(errorBoundaryPath, errorBoundary);
    }
}

// Fix 9: Update package.json scripts
function updatePackageScripts() {
    const packageJsonPath = path.join(projectRoot, 'package.json');
    let content = readFile(packageJsonPath);
    if (!content) return;

    const packageJson = JSON.parse(content);
    
    // Add useful scripts
    packageJson.scripts = {
      ...packageJson.scripts,
      "type-check": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json",
      "type-check:watch": "svelte-kit sync && svelte-check --tsconfig ./tsconfig.json --watch",
      "fix-types": "node ../fix-all-errors-comprehensive.mjs",
      "clean": "rimraf .svelte-kit build dist",
      "clean:all": "rimraf .svelte-kit build dist node_modules",
      "reinstall": "npm run clean:all && npm install",
      "dev:clean": "npm run clean && npm run dev",
      "build:clean": "npm run clean && npm run build"
    };

    writeFile(packageJsonPath, JSON.stringify(packageJson, null, 2));
}

// Main execution function
async function main() {
    console.log('🚀 Starting comprehensive project setup...\n');

    try {
        console.log('📚 Setting up type definitions...');
        createGlobalTypes();
        updateAppDts();
        createSelectTypes();

        console.log('⚙️ Updating configuration files...');
        updateTsConfig();
        updateViteConfig();
        updateSvelteConfig();

        console.log('🔧 Setting up environment...');
        createEnvSetup();
        createErrorBoundary();

        console.log('📝 Updating package scripts...');
        updatePackageScripts();

        console.log('\n✨ All fixes and setup completed!');
        console.log('\n📋 Summary of changes:');
        console.log('  • Created comprehensive type definitions');
        console.log('  • Updated TypeScript configuration');
        console.log('  • Enhanced Vite and Svelte configurations');
        console.log('  • Set up environment configuration');
        console.log('  • Created error boundary component');
        console.log('  • Updated package.json scripts');

        console.log('\n🎯 Next steps to fix remaining issues:');
        console.log('  1. Run "npm install" to ensure all dependencies are installed');
        console.log('  2. Run "npm run type-check" to verify type issues are resolved');
        console.log('  3. Run "npm run dev:clean" to start development server');
        console.log('  4. Check the .env file and update with your actual values');
        
        console.log('\n📝 Available scripts:');
        console.log('  • npm run type-check       - Check TypeScript types');
        console.log('  • npm run type-check:watch - Watch for type changes');
        console.log('  • npm run fix-types        - Run this fix script again');
        console.log('  • npm run clean            - Clean build artifacts');
        console.log('  • npm run clean:all        - Clean everything including node_modules');
        console.log('  • npm run reinstall        - Clean and reinstall dependencies');
        console.log('  • npm run dev:clean        - Clean and start development');

    } catch (error) {
        console.error('❌ Error during setup:', error);
        process.exit(1);
    }
}

// Run the setup
main().catch(console.error);
