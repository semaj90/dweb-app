#!/usr/bin/env node
/*
  Clean CommonJS TypeScript auto-fixer (no-glob)
  - Recursively finds .ts/.tsx files (skips node_modules/.svelte-kit/.git)
  - Creates backups under ./typescript-backup
  - Applies a small set of conservative, mechanical fixes
*/
const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve('.');
const BACKUP_DIR = path.join(ROOT_DIR, 'typescript-backup');
if (!fs.existsSync(BACKUP_DIR)) fs.mkdirSync(BACKUP_DIR, { recursive: true });

const stats = { filesProcessed: 0, fixesApplied: 0, backedUp: 0, errors: [] };

function findTypeScriptFiles(dir, files = []) {
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    for (const item of items) {
      const fullPath = path.join(dir, item.name);
      if (item.isDirectory()) {
        if (item.name === 'node_modules' || item.name === '.svelte-kit' || item.name === '.git') continue;
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

const fixes = {
  fixWorkerImports(content) {
    let fixed = content;
    let changes = 0;
    if (fixed.includes('worker_threads')) {
      fixed = fixed.replace(/import\s+\{[^}]+\}\s+from\s+['\"]worker_threads['\"];?/g, '// Worker threads import removed for browser worker compatibility\n// declare const self: DedicatedWorkerGlobalScope;');
      changes++;
    }
    fixed = fixed.replace(/parentPort\?\.postMessage/g, 'self.postMessage');
    fixed = fixed.replace(/parentPort\?\.on\("message"/g, 'self.addEventListener("message"');
    if (fixed !== content) changes++;
    return { content: fixed, changes };
  },
  fixAsyncReturnTypes(content) {
    let fixed = content; let changes = 0;
    fixed = fixed.replace(/async\s+function\s+(\w+)\s*\(([^)]*)\)\s*(?=\{)/g, (m) => {
      if (/\):\s*Promise</.test(m) || /\):\s*any/.test(m)) return m;
      changes++;
      return m.replace(/\)\s*$/, '): Promise<any> ');
    });
    fixed = fixed.replace(/(const|let|var)\s+(\w+)\s*=\s*async\s*\(([^)]*)\)\s*=>/g, (m) => {
      if (/:\s*Promise/.test(m)) return m;
      changes++;
      return m.replace(/\)\s*=>/, '): Promise<any> =>');
    });
    return { content: fixed, changes };
  },
  fixImportPaths(content) {
    let fixed = content; let changes = 0;
    fixed = fixed.replace(/from\s+['\"]((?:\.\/|\.\.\/)[^'\"]+)\.js['\"]/g, (m, p1) => { changes++; return `from '${p1}'`; });
    fixed = fixed.replace(/from\s+['\"](\.\.\/)+src\/lib\//g, (m) => { changes++; return "from '$lib/"; });
    return { content: fixed, changes };
  },
  fixMissingTypes(content) {
    let fixed = content; let changes = 0;
    fixed = fixed.replace(/\(e\)\s*=>/g, '(e: any) =>');
    fixed = fixed.replace(/\(event\)\s*=>/g, '(event: any) =>');
    fixed = fixed.replace(/catch\s*\((?:e|err|error)\)/g, (m) => { changes++; return m.replace(/\)/, ': any)'); });
    return { content: fixed, changes };
  },
  fixVectorDimensions(content) {
    let fixed = content; let changes = 0;
    const patterns = [/dimensions:\s*768/g, /vectorDimensions:\s*768/g, /VECTOR\(768\)/g, /vector\(768\)/g];
    patterns.forEach(p => { if (p.test(fixed)) { fixed = fixed.replace(p, (m) => { changes++; return m.replace(/768/g, '384'); }); } });
    return { content: fixed, changes };
  },
  fixMissingExports(content) {
    let fixed = content; let changes = 0;
    fixed = fixed.replace(/^interface\s+([A-Z]\w+)/gm, (m) => { changes++; return `export ${m}`; });
    return { content: fixed, changes };
  },
  fixLibraryImports(content) {
    let fixed = content; let changes = 0;
    if (fixed.includes('drizzle-orm/postgres-js')) { fixed = fixed.split('drizzle-orm/postgres-js').join('drizzle-orm/node-postgres'); changes++; }
    if (fixed.includes('@ts-nocheck')) { fixed = fixed.replace(/\/\/\s*@ts-nocheck\n?/g, ''); changes++; }
    return { content: fixed, changes };
  }
};

function processFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    const relativePath = path.relative(ROOT_DIR, filePath);
    const backupPath = path.join(BACKUP_DIR, relativePath);
    const backupDir = path.dirname(backupPath);
    if (!fs.existsSync(backupDir)) fs.mkdirSync(backupDir, { recursive: true });
    fs.writeFileSync(backupPath, originalContent, 'utf8');
    stats.backedUp++;
    let modifiedContent = content; let totalChanges = 0;
    const isWorker = filePath.toLowerCase().includes('worker');
    if (isWorker) { const r = fixes.fixWorkerImports(modifiedContent); modifiedContent = r.content; totalChanges += r.changes; }
    const order = ['fixAsyncReturnTypes','fixImportPaths','fixMissingTypes','fixVectorDimensions','fixMissingExports','fixLibraryImports'];
    for (const fn of order) { const { content: c, changes } = fixes[fn](modifiedContent); modifiedContent = c; totalChanges += (changes || 0); }
    if (modifiedContent !== originalContent) { fs.writeFileSync(filePath, modifiedContent, 'utf8'); stats.fixesApplied += totalChanges; console.log(`✅ Fixed ${totalChanges} issues in: ${filePath}`); }
    stats.filesProcessed++;
  } catch (error) { stats.errors.push({ file: filePath, error: error.message }); console.error(`❌ Error processing ${filePath}:`, error.message); }
}

function main() {
  console.log('🔧 TypeScript Auto-Fixer (no-glob, CJS) Starting...\n');
  const allFiles = findTypeScriptFiles(ROOT_DIR);
  console.log(`Found ${allFiles.length} TypeScript files to process`);
  const priorityFiles = [ 'src/lib/workers/enhanced-analysis-worker.ts', 'src/lib/workers/streaming-worker.ts' ];
  console.log('Processing priority files...');
  for (const pf of priorityFiles) { const full = path.join(ROOT_DIR, pf); if (fs.existsSync(full)) processFile(full); }
  console.log('Processing remaining files...');
  for (const f of allFiles) { const rel = path.relative(ROOT_DIR, f); if (priorityFiles.some(p => rel === p)) continue; processFile(f); }
  console.log('\n' + '='.repeat(60));
  console.log('📊 Auto-Fix Summary');
  console.log('Files processed:', stats.filesProcessed);
  console.log('Total fixes applied:', stats.fixesApplied);
  console.log('Files backed up:', stats.backedUp);
  console.log('Errors:', stats.errors.length);
  if (stats.errors.length) { console.log('Error details:'); stats.errors.forEach(e => console.log(e)); }
  console.log('\nDone. Backups are in ./typescript-backup');
}

main();