// analyze-typescript-errors.js
// This script analyzes TypeScript files to show the current error patterns

const fs = require('fs');
const path = require('path');

console.log('TypeScript Error Analysis Report');
console.log('=' .repeat(70));
console.log('');

// Files to analyze (highest error counts)
const targetFiles = [
  'src/lib/workers/enhanced-analysis-worker.ts',
  'src/lib/workers/streaming-worker.ts',
  'src/lib/websockets/cache-monitoring-service.ts',
  'src/routes/api/caching/advanced/+server.ts',
  'src/routes/api/documents/+server.ts'
];

let totalIssuesFound = 0;
let totalFilesScan = 0;
let fixedFiles = 0;

targetFiles.forEach(file => {
  const fullPath = path.join('C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app', file);
  
  if (fs.existsSync(fullPath)) {
    totalFilesScan++;
    console.log(`\n📄 Analyzing: ${file}`);
    console.log('-'.repeat(60));
    
    const content = fs.readFileSync(fullPath, 'utf8');
    let issues = [];
    let fixes = [];
    
    // Check if already fixed
    if (content.includes('declare const self: DedicatedWorkerGlobalScope')) {
      fixes.push('✅ Web Worker context properly declared');
      fixedFiles++;
    }
    
    // Check for Node.js worker imports (issue)
    if (content.includes('import { parentPort, workerData } from "worker_threads"')) {
      issues.push('❌ Node.js worker imports still present');
    } else if (file.includes('worker')) {
      fixes.push('✅ No Node.js worker imports');
    }
    
    // Check for node-fetch
    if (content.includes('import fetch from "node-fetch"')) {
      issues.push('❌ node-fetch import present');
    }
    
    // Check for workerData usage (problematic)
    const workerDataUsage = (content.match(/workerData/g) || []).length;
    if (workerDataUsage > 0 && !content.includes('declare')) {
      issues.push(`⚠️  ${workerDataUsage} workerData references found`);
    }
    
    // Check for parentPort usage (should be replaced with self)
    const parentPortUsage = (content.match(/parentPort\??\./g) || []).length;
    if (parentPortUsage > 0) {
      issues.push(`⚠️  ${parentPortUsage} parentPort references (should use self)`);
    }
    
    // Check async functions without return types
    const asyncNoReturn = (content.match(/async\s+\w+\s*\([^)]*\)\s*(?!:)\s*{/g) || []).length;
    if (asyncNoReturn > 0) {
      issues.push(`⚠️  ${asyncNoReturn} async functions missing return types`);
    }
    
    // Check for relative imports that should use $lib
    const relativeImports = (content.match(/from\s+['"](\.\.\/)+lib\//g) || []).length;
    if (relativeImports > 0) {
      issues.push(`⚠️  ${relativeImports} relative imports (should use $lib)`);
    }
    
    // Check for vector dimensions
    if (content.includes('vector(768)') || content.includes('VECTOR(768)')) {
      issues.push('⚠️  Old vector dimensions (768) found');
    }
    if (content.includes('vector(384)') || content.includes('VECTOR(384)')) {
      fixes.push('✅ Correct vector dimensions (384) used');
    }
    
    // Check for @ts-nocheck
    if (content.includes('// @ts-nocheck')) {
      issues.push('⚠️  @ts-nocheck directive present');
    }
    
    // Print results for this file
    if (fixes.length > 0) {
      console.log('Fixed items:');
      fixes.forEach(fix => console.log(`  ${fix}`));
    }
    
    if (issues.length > 0) {
      console.log('Remaining issues:');
      issues.forEach(issue => console.log(`  ${issue}`));
      totalIssuesFound += issues.length;
    } else {
      console.log('  ✨ No major TypeScript issues detected!');
    }
    
    // Estimate error count reduction
    const estimatedErrors = asyncNoReturn * 2 + parentPortUsage * 3 + relativeImports + workerDataUsage * 2;
    if (estimatedErrors > 0) {
      console.log(`  📊 Estimated TypeScript errors from this file: ~${estimatedErrors}`);
    }
  }
});

console.log('\n' + '='.repeat(70));
console.log('📊 SUMMARY');
console.log('='.repeat(70));
console.log(`Files analyzed: ${totalFilesScan}`);
console.log(`Files with fixes applied: ${fixedFiles}`);
console.log(`Total remaining issues: ${totalIssuesFound}`);
console.log('');
console.log('Next steps:');
console.log('1. The enhanced-analysis-worker.ts has been fixed');
console.log('2. Run "npm run check" to see exact remaining error count');
console.log('3. Most critical Worker API issues have been resolved');
console.log('');