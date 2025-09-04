#!/usr/bin/env zx
// Fix Core TypeScript Errors - Targeted Approach
// Focus on the most critical syntax issues first

import { $, glob } from 'zx';
import fs from 'fs/promises';
import path from 'path';

console.log('🔧 Starting targeted core error fixes...');

// Critical files that need immediate fixing
const criticalFiles = [
  'src/lib/optimization/advanced-memory-optimizer.ts',
  'src/lib/server/services/QdrantService.ts', 
  'src/lib/services/ai-worker-manager.ts',
  'src/lib/services/autogen-service.ts',
  'src/lib/services/crewai-service.ts',
  'src/lib/services/context7-autosolve-integration.ts'
];

let totalFixes = 0;

for (const file of criticalFiles) {
  try {
    console.log(`🔄 Processing ${path.basename(file)}...`);
    
    const content = await fs.readFile(file, 'utf-8');
    let modified = content;
    let fileFixes = 0;

    // Fix: Remove orphaned content comments and fix imports
    modified = modified.replace(/\/\/ Orphaned content: [^\n]*\n/g, '');
    
    // Fix: Restore proper import statements
    modified = modified.replace(/^([a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+"([^"]+)";$/gm, 'import $1 from "$2";');
    
    // Fix: Restore import { } statements  
    modified = modified.replace(/^\{\s*([^}]+)\s*\}\s+from\s+"([^"]+)";$/gm, 'import { $1 } from "$2";');
    
    // Fix: Add missing 'import' keywords
    modified = modified.replace(/^(\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+["']([^"']+)["'];$/gm, '$1import $2 from "$3";');
    
    // Fix: Clean up malformed type imports
    modified = modified.replace(/^(\s*)(type\s+)?\{\s*([^}]+)\s*\}(\s*from\s+["'][^"']+["'];?)$/gm, '$1import $2{ $3 }$4');

    // Fix specific issues
    if (file.includes('advanced-memory-optimizer.ts')) {
      modified = modified.replace(
        /SelfOrganizingMapRAG\n.*?crypto from "crypto";\n.*?SIMDJSONParser,/s,
        'import { SelfOrganizingMapRAG } from "../ai/som-rag-system.js";\nimport crypto from "crypto";\nimport {\n  SIMDJSONParser,'
      );
    }

    if (content !== modified) {
      await fs.writeFile(file, modified);
      fileFixes = 1;
      console.log(`✅ Fixed critical errors in ${path.basename(file)}`);
      totalFixes++;
    } else {
      console.log(`✓ No issues found in ${path.basename(file)}`);
    }

  } catch (error) {
    console.log(`❌ Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Core error fixes complete!`);
console.log(`🔧 Fixed: ${totalFixes} critical files`);

// Quick validation
console.log('\n🔍 Running quick validation...');
try {
  await $`npm run check:ultra-fast`;
  console.log('✅ Quick validation passed');
} catch (error) {
  console.log('⚠️ Some errors remain - continuing with comprehensive fix');
}