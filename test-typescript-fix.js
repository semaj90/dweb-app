// test-typescript-fix.js
// Simple test to show the fixes working

const fs = require('fs');
const path = require('path');

console.log('TypeScript Auto-Fixer Test Run\n');
console.log('=' .repeat(60));

// Test on the enhanced-analysis-worker.ts file
const testFile = 'src/lib/workers/streaming-worker.ts';
const fullPath = path.join('C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app', testFile);

console.log(`\nReading: ${testFile}`);

try {
  const content = fs.readFileSync(fullPath, 'utf8');
  console.log(`File size: ${content.length} characters`);
  
  // Count current issues
  let issues = [];
  
  if (content.includes('import { parentPort')) {
    issues.push('Node.js worker imports detected');
  }
  if (content.includes('node-fetch')) {
    issues.push('node-fetch import detected');
  }
  
  // Count async functions without return types
  const asyncWithoutReturn = (content.match(/async\s+function\s+\w+\s*\([^)]*\)\s*{/g) || []).length;
  if (asyncWithoutReturn > 0) {
    issues.push(`${asyncWithoutReturn} async functions without return types`);
  }
  
  // Count relative imports
  const relativeImports = (content.match(/from\s+['"](\.\.\/)+lib\//g) || []).length;
  if (relativeImports > 0) {
    issues.push(`${relativeImports} relative imports that should use $lib`);
  }
  
  console.log(`\nIssues found:`);
  issues.forEach(issue => console.log(`  - ${issue}`));
  
  // Apply fixes
  let fixed = content;
  let totalFixes = 0;
  
  // Fix worker imports
  if (fixed.includes('import { parentPort, workerData } from "worker_threads"')) {
    fixed = fixed.replace(
      'import { parentPort, workerData } from "worker_threads";',
      '// Web Worker context - no imports needed\ndeclare const self: DedicatedWorkerGlobalScope;'
    );
    totalFixes++;
    console.log('\n✅ Fixed: Worker imports replaced with Web Worker API');
  }
  
  // Fix node-fetch
  if (fixed.includes('import fetch from "node-fetch"')) {
    fixed = fixed.replace(
      'import fetch from "node-fetch";',
      '// fetch is available globally in Web Workers'
    );
    totalFixes++;
    console.log('✅ Fixed: Removed node-fetch import');
  }
  
  // Fix parentPort references
  const parentPortCount = (fixed.match(/parentPort\??\./g) || []).length;
  if (parentPortCount > 0) {
    fixed = fixed.replace(/parentPort\?\.postMessage/g, 'self.postMessage');
    fixed = fixed.replace(/parentPort\?\.on\("message"/g, 'self.addEventListener("message"');
    totalFixes += parentPortCount;
    console.log(`✅ Fixed: ${parentPortCount} parentPort references`);
  }
  
  // Save fixed file
  const backupPath = fullPath + '.backup';
  fs.writeFileSync(backupPath, content);
  fs.writeFileSync(fullPath, fixed);
  
  console.log(`\n📊 Results:`);
  console.log(`  Total fixes applied: ${totalFixes}`);
  console.log(`  Backup saved to: ${path.basename(backupPath)}`);
  console.log(`  File updated: ${testFile}`);
  
} catch (error) {
  console.error(`Error: ${error.message}`);
}

console.log('\n' + '='.repeat(60));
console.log('Test complete!');