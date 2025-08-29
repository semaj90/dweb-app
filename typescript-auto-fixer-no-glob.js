// typescript-auto-fixer-no-glob.js
// Run with: node typescript-auto-fixer-no-glob.js

const fs = require('fs');
const path = require('path');

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
 * Recursively find all TypeScript files
 */
function findTypeScriptFiles(dir, files = []) {
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    
    for (const item of items) {
      const fullPath = path.join(dir, item.name);
      
      if (item.isDirectory() && !item.name.includes('node_modules') && !item.name.includes('.svelte-kit')) {
        findTypeScriptFiles(fullPath, files);
      } else if (item.isFile() && (item.name.endsWith('.ts') || item.name.endsWith('.tsx'))) {
        files.push(fullPath);
      }
    }
  } catch (err) {
    console.error(`Error reading directory ${dir}:`, err.message);
  }
  
  return files;
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
    
    if (fixed !== content) changes++;
    
    return { content: fixed, changes };
  },

  // Fix 2: Add proper return types to async functions
  fixAsyncReturnTypes(content) {
    let fixed = content;
    let changes = 0;

    // Pattern: async function without return type
    const asyncFunctionPattern = /async\s+function\s+(\w+)\s*\([^)]*\)\s*(?!:)/g;
    let matches = [...fixed.matchAll(asyncFunctionPattern)];
    if (matches.length > 0) {
      matches.forEach(() => {
        fixed = fixed.replace(asyncFunctionPattern, (match) => {
          if (!match.includes(':')) {
            return match.replace(/\)\s*(?={)/, '): Promise<any> ');
          }
          return match;
        });
      });
      changes += matches.length;
    }

    // Pattern: async arrow functions without return type
    const asyncArrowPattern = /async\s*\([^)]*\)\s*=>/g;
    matches = [...fixed.matchAll(asyncArrowPattern)];
    if (matches.length > 0) {
      fixed = fixed.replace(asyncArrowPattern, (match) => {
        if (!match.includes(':')) {
          changes++;
          return match.replace(/\)\s*=>/, '): Promise<any> =>');
        }
        return match;
      });
    }

    return { content: fixed, changes };
  },

  // Fix 3: Fix import paths
  fixImportPaths(content, filePath) {
    let fixed = content;
    let changes = 0;

    // Fix relative imports to use $lib alias
    const relativeImportPattern = /from\s+['"](\.\.\/)+lib\//g;
    const matches = fixed.match(relativeImportPattern);
    if (matches) {
      fixed = fixed.replace(relativeImportPattern, 'from \'$lib/');
      changes += matches.length;
    }

    // Fix .js extensions
    const jsImportPattern = /from\s+['"]([^'"]+)\.js['"]/g;
    const jsMatches = [...fixed.matchAll(jsImportPattern)];
    jsMatches.forEach(match => {
      if (!match[1].startsWith('http')) {
        fixed = fixed.replace(match[0], `from '${match[1]}'`);
        changes++;
      }
    });

    return { content: fixed, changes };
  },

  // Fix 4: Add type annotations for common patterns
  fixMissingTypes(content) {
    let fixed = content;
    let changes = 0;

    // Add type to event handlers
    const eventPatterns = [
      { from: /\(e\)\s*=>/g, to: '(e: any) =>' },
      { from: /\(event\)\s*=>/g, to: '(event: any) =>' },
      { from: /catch\s*\(error\)/g, to: 'catch (error: any)' },
      { from: /catch\s*\(e\)/g, to: 'catch (e: any)' }
    ];

    eventPatterns.forEach(pattern => {
      const matches = fixed.match(pattern.from);
      if (matches) {
        fixed = fixed.replace(pattern.from, pattern.to);
        changes += matches.length;
      }
    });

    return { content: fixed, changes };
  },

  // Fix 5: Fix vector dimension types
  fixVectorDimensions(content) {
    let fixed = content;
    let changes = 0;

    const patterns = [
      { from: /vector\(768\)/g, to: 'vector(384)' },
      { from: /VECTOR\(768\)/g, to: 'VECTOR(384)' },
      { from: /dimensions:\s*768/g, to: 'dimensions: 384' },
      { from: /vectorDimensions:\s*768/g, to: 'vectorDimensions: 384' }
    ];

    patterns.forEach(pattern => {
      if (fixed.includes(pattern.from.source.replace(/\\/g, ''))) {
        fixed = fixed.replace(pattern.from, pattern.to);
        changes++;
      }
    });

    return { content: fixed, changes };
  },

  // Fix 6: Add missing interface exports
  fixMissingExports(content) {
    let fixed = content;
    let changes = 0;

    // Add export to interfaces
    const interfacePattern = /^interface\s+([A-Z]\w+)/gm;
    const matches = [...fixed.matchAll(interfacePattern)];
    
    matches.forEach(match => {
      const interfaceName = match[1];
      if (!content.includes(`export interface ${interfaceName}`) && 
          !content.includes(`export { ${interfaceName}`)) {
        fixed = fixed.replace(match[0], `export ${match[0]}`);
        changes++;
      }
    });

    return { content: fixed, changes };
  },

  // Fix 7: Fix common library import issues
  fixLibraryImports(content) {
    let fixed = content;
    let changes = 0;

    // Remove @ts-nocheck
    if (fixed.includes('// @ts-nocheck')) {
      fixed = fixed.replace(/\/\/\s*@ts-nocheck\n?/g, '');
      changes++;
    }

    // Fix Drizzle imports
    if (fixed.includes('drizzle-orm/postgres-js')) {
      fixed = fixed.replace(/from ['"]drizzle-orm\/postgres-js['"]/g, 'from \'drizzle-orm/node-postgres\'');
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
    const relativePath = path.relative('.', filePath);
    const backupPath = path.join(BACKUP_DIR, relativePath);
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

    // Apply appropriate fixes
    if (isWorkerFile) {
      const result = fixes.fixWorkerImports(modifiedContent);
      modifiedContent = result.content;
      totalChanges += result.changes;
    }

    // Apply common fixes to all TypeScript files
    const fixFunctions = [
      'fixAsyncReturnTypes',
      'fixImportPaths',
      'fixMissingTypes',
      'fixVectorDimensions',
      'fixMissingExports',
      'fixLibraryImports'
    ];

    for (const fixName of fixFunctions) {
      const result = fixes[fixName](modifiedContent, filePath);
      modifiedContent = result.content;
      totalChanges += result.changes;
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
async function main() {
  console.log('🔧 TypeScript Auto-Fixer Starting...\n');
  console.log(`📁 Backup directory: ${BACKUP_DIR}\n`);
  console.log('Scanning for TypeScript files...\n');

  // Find all TypeScript files
  const files = findTypeScriptFiles(ROOT_DIR);
  console.log(`Found ${files.length} TypeScript files to process\n`);

  // Process files with highest error counts first
  const priorityFiles = [
    'src/lib/workers/enhanced-analysis-worker.ts',
    'src/lib/workers/streaming-worker.ts',
    'src/lib/websockets/cache-monitoring-service.ts',
    'src/routes/api/caching/advanced/+server.ts',
    'src/routes/api/documents/+server.ts',
    'src/routes/api/scaling/horizontal/+server.ts',
    'src/routes/api/gpu/acceleration/+server.ts'
  ];

  // Process priority files first
  console.log('Processing high-priority files first...\n');
  for (const file of priorityFiles) {
    const fullPath = path.join('.', file);
    if (fs.existsSync(fullPath)) {
      console.log(`🎯 Processing priority file: ${file}`);
      processFile(fullPath);
    }
  }

  // Process remaining files
  console.log('\nProcessing remaining files...\n');
  for (const file of files) {
    const normalizedFile = file.replace(/\\/g, '/');
    const isPriority = priorityFiles.some(pf => normalizedFile.includes(pf.replace(/\\/g, '/')));
    if (!isPriority) {
      processFile(file);
    }
  }

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 Auto-Fix Summary');
  console.log('='.repeat(60));
  console.log(`Files processed: ${stats.filesProcessed}`);
  console.log(`Total fixes applied: ${stats.fixesApplied}`);
  console.log(`Files backed up: ${stats.backedUp}`);
  console.log(`Errors encountered: ${stats.errors.length}`);

  if (stats.errors.length > 0) {
    console.log('\n⚠️ Files with errors:');
    stats.errors.forEach(({ file, error }) => {
      console.log(`  - ${file}: ${error}`);
    });
  }

  console.log('\n✨ Auto-fix complete!');
  console.log(`💾 Original files backed up to: ${BACKUP_DIR}`);
  console.log(`\n💡 Next steps:`);
  console.log(`   1. Run 'npm run check' to see remaining TypeScript errors`);
  console.log(`   2. Test your application with 'npm run dev'`);
  console.log(`   3. If issues arise, restore from ${BACKUP_DIR}`);
}

// Run the script
main().catch(console.error);