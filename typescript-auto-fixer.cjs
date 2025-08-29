// typescript-auto-fixer.cjs
// Run with: node typescript-auto-fixer.cjs

const fs = require('fs');
const path = require('path');
// lightweight recursive file collector to avoid external glob dependency
const glob = null;

// Configuration
const ROOT_DIR = './src';
const BACKUP_DIR = './typescript-backup';

// Statistics tracking
let stats = {
  filesProcessed: 0,
  fixesApplied: 0,
  errors: [],
  backedUp: 0
};

// Create backup directory
if (!fs.existsSync(BACKUP_DIR)) {
  fs.mkdirSync(BACKUP_DIR, { recursive: true });
}

/**
 * Common TypeScript fixes
 */
const fixes = {
  // Fix 1: Replace Node.js worker imports with Web Worker APIs
  fixWorkerImports(content) {
    let fixed = content;
    let changes = 0;

    // Remove Node.js worker_threads imports
    if (fixed.includes('import { parentPort, workerData } from "worker_threads"')) {
      fixed = fixed.replace(
        'import { parentPort, workerData } from "worker_threads";',
        '// Web Worker context - no imports needed\ndeclare const self: DedicatedWorkerGlobalScope;'
      );
      changes++;
    }

    // Remove node-fetch imports in worker files
    if (fixed.includes('import fetch from "node-fetch"')) {
      fixed = fixed.replace(
        'import fetch from "node-fetch";',
        '// fetch is available globally in Web Workers'
      );
      changes++;
    }

    // Replace parentPort with self
    fixed = fixed.replace(/parentPort\?\.postMessage/g, 'self.postMessage');
    fixed = fixed.replace(/parentPort\?\.on\("message"/g, 'self.addEventListener("message"');

    return { content: fixed, changes };
  },

  // Fix 2: Add proper return types to async functions
  fixAsyncReturnTypes(content) {
    let fixed = content;
    let changes = 0;

    // Pattern: async function without return type
    const asyncFunctionPattern = /async\s+function\s+(\w+)\s*\([^)]*\)\s*(?!:)/g;
    fixed = fixed.replace(asyncFunctionPattern, (match, name) => {
      changes++;
      return match.replace(/\)(?!:)/, '): Promise<any>');
    });

    // Pattern: async arrow functions without return type
    const asyncArrowPattern = /async\s*\([^)]*\)\s*=>/g;
    fixed = fixed.replace(asyncArrowPattern, (match) => {
      if (!match.includes(':')) {
        changes++;
        return match.replace(/\)\s*=>/, '): Promise<any> =>');
      }
      return match;
    });

    // Pattern: async methods without return type
    const asyncMethodPattern = /async\s+(\w+)\s*\([^)]*\)\s*{/g;
    fixed = fixed.replace(asyncMethodPattern, (match, name) => {
      if (!match.includes(':')) {
        changes++;
        return match.replace(/\)\s*{/, '): Promise<any> {');
      }
      return match;
    });

    return { content: fixed, changes };
  },

  // Fix 3: Fix import paths
  fixImportPaths(content, filePath) {
    let fixed = content;
    let changes = 0;

    // Fix relative imports to use $lib alias
    const relativeImportPattern = /from\s+['\"](\.\.\/)+lib\//g;
    fixed = fixed.replace(relativeImportPattern, (match) => {
      changes++;
      return "from '$lib/";
    });

    // Fix .js extensions to .ts
    fixed = fixed.replace(/from\s+['\"]([^'\"]+)\.js['\"]/g, (match, path) => {
      if (!path.startsWith('http')) {
        changes++;
        return `from '${path}'`;
      }
      return match;
    });

    return { content: fixed, changes };
  },

  // Fix 4: Add type annotations for common patterns
  fixMissingTypes(content) {
    let fixed = content;
    let changes = 0;

    // Add type to event handlers
    fixed = fixed.replace(/\(e\)\s*=>/g, '(e: any) =>');
    fixed = fixed.replace(/\(event\)\s*=>/g, '(event: any) =>');

    // Add type to error handlers
    fixed = fixed.replace(/catch\s*\(error\)/g, 'catch (error: any)');
    fixed = fixed.replace(/catch\s*\(e\)/g, 'catch (e: any)');

    // Add type to common parameters
    fixed = fixed.replace(/function\s+\w+\(data\)/g, (match) => {
      changes++;
      return match.replace('(data)', '(data: any)');
    });

    return { content: fixed, changes };
  },

  // Fix 5: Fix vector dimension types
  fixVectorDimensions(content) {
    let fixed = content;
    let changes = 0;

    // Update vector dimensions from 768 to 384
    fixed = fixed.replace(/vector\(768\)/g, 'vector(384)');
    fixed = fixed.replace(/VECTOR\(768\)/g, 'VECTOR(384)');
    fixed = fixed.replace(/dimensions:\s*768/g, 'dimensions: 384');
    fixed = fixed.replace(/vectorDimensions:\s*768/g, 'vectorDimensions: 384');

    if (fixed !== content) changes++;

    return { content: fixed, changes };
  },

  // Fix 6: Add missing interface exports
  fixMissingExports(content) {
    let fixed = content;
    let changes = 0;

    // Add export to interfaces that are likely used elsewhere
    const interfacePattern = /^interface\s+([A-Z]\w+)/gm;
    fixed = fixed.replace(interfacePattern, (match, name) => {
      // Skip if already exported
      if (!content.includes(`export interface ${name}`) && !content.includes(`export { ${name}`)) {
        changes++;
        return `export ${match}`;
      }
      return match;
    });

    return { content: fixed, changes };
  },

  // Fix 7: Fix Promise type issues
  fixPromiseTypes(content) {
    let fixed = content;
    let changes = 0;

    // Fix void Promise returns
    fixed = fixed.replace(/Promise<void>/g, 'Promise<any>');

    // Add Promise wrapper to async function returns
    fixed = fixed.replace(/:\s*(\w+)\s*{\s*\/\/\s*async/g, ': Promise<$1> { // async');

    if (fixed !== content) changes++;

    return { content: fixed, changes };
  },

  // Fix 8: Fix common library import issues
  fixLibraryImports(content) {
    let fixed = content;
    let changes = 0;

    // Fix Drizzle ORM imports
    if (fixed.includes('drizzle-orm/')) {
      // Replace postgres-js import path with node-postgres package path
      // Use split/join to avoid regex escaping pitfalls
      fixed = fixed.split('drizzle-orm/postgres-js').join('drizzle-orm/node-postgres');
      fixed = fixed.split('"drizzle-orm/postgres-js"').join('"drizzle-orm/node-postgres"');
      fixed = fixed.split("'drizzle-orm/postgres-js'").join("'drizzle-orm/node-postgres'");
      changes++;
    }

    // Fix missing type imports
    if (fixed.includes('// @ts-nocheck')) {
      fixed = fixed.replace('// @ts-nocheck\n', '');
      changes++;
    }

    return { content: fixed, changes };
  }
};

/**
 * Process a single file
 */
function processFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;

    // Create backup
    const backupPath = path.join(BACKUP_DIR, path.relative('.', filePath));
    const backupDir = path.dirname(backupPath);
    if (!fs.existsSync(backupDir)) {
      fs.mkdirSync(backupDir, { recursive: true });
    }
    fs.writeFileSync(backupPath, originalContent);
    stats.backedUp++;

    let modifiedContent = content;
    let totalChanges = 0;

    // Apply fixes based on file type
    const isWorkerFile = filePath.includes('worker');
    const isServerFile = filePath.includes('+page.server.ts') || filePath.includes('+server.ts');
    const isComponentFile = filePath.includes('.svelte');

    // Apply appropriate fixes
    if (isWorkerFile) {
      const result = fixes.fixWorkerImports(modifiedContent);
      modifiedContent = result.content;
      totalChanges += result.changes;
    }

    // Apply common fixes to all TypeScript files
    if (filePath.endsWith('.ts')) {
      for (const fixName of ['fixAsyncReturnTypes', 'fixImportPaths', 'fixMissingTypes',
                             'fixVectorDimensions', 'fixMissingExports', 'fixPromiseTypes',
                             'fixLibraryImports']) {
        const result = fixes[fixName](modifiedContent, filePath);
        modifiedContent = result.content;
        totalChanges += result.changes;
      }
    }

    // Write fixed content if changes were made
    if (modifiedContent !== originalContent) {
      fs.writeFileSync(filePath, modifiedContent);
      stats.fixesApplied += totalChanges;
      console.log(`✅ Fixed ${totalChanges} issues in: ${filePath}`);
    }

    stats.filesProcessed++;

  } catch (error) {
    stats.errors.push({ file: filePath, error: error.message });
    console.error(`❌ Error processing ${filePath}:`, error.message);
  }
}

/**
 * Main execution
 */
function collectFiles(dir, out = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (['node_modules', 'dist', '.svelte-kit'].includes(entry.name)) continue;
      collectFiles(full, out);
    } else if (entry.isFile()) {
      if (full.endsWith('.ts') || full.endsWith('.tsx')) {
        out.push(full.replace(/\\/g, '/'));
      }
    }
  }
  return out;
}

async function main() {
  console.log('🔧 TypeScript Auto-Fixer Starting...\n');
  console.log(`📁 Backup directory: ${BACKUP_DIR}\n`);

  // Find all TypeScript files
  const files = fs.existsSync(ROOT_DIR) ? collectFiles(ROOT_DIR) : [];

  console.log(`Found ${files.length} TypeScript files to process\n`);

  // Process priority files first
  const priorityFiles = [
    'src/lib/workers/enhanced-analysis-worker.ts',
    'src/lib/workers/streaming-worker.ts',
    'src/lib/websockets/cache-monitoring-service.ts',
    'src/routes/api/caching/advanced/+server.ts',
    'src/routes/api/documents/+server.ts',
    'src/routes/api/scaling/horizontal/+server.ts',
    'src/routes/api/gpu/acceleration/+server.ts'
  ];

  for (const file of priorityFiles) {
    if (fs.existsSync(file)) {
      console.log(`🎯 Processing priority file: ${file}`);
      processFile(file);
    }
  }

  // Process remaining files
  for (const file of files) {
    if (!priorityFiles.includes(file)) {
      processFile(file);
    }
  }

  // Print summary
  console.log('\n📊 Auto-Fix Summary:');
  console.log('='.repeat(50));
  console.log(`Files processed: ${stats.filesProcessed}`);
  console.log(`Fixes applied: ${stats.fixesApplied}`);
  console.log(`Files backed up: ${stats.backedUp}`);
  console.log(`Errors encountered: ${stats.errors.length}`);

  if (stats.errors.length > 0) {
    console.log('\n⚠️ Files with errors:');
    stats.errors.forEach(({ file, error }) => {
      console.log(`  - ${file}: ${error}`);
    });
  }

  console.log('\n✨ Auto-fix complete!');
  console.log(`💡 Run 'npm run check' to see remaining TypeScript errors`);
  console.log(`💾 Original files backed up to: ${BACKUP_DIR}`);
}

// Run the script
main().catch(console.error);
