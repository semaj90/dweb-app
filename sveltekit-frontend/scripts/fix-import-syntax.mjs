#!/usr/bin/env zx
// Fix Import Syntax Issues
// Fixes malformed import statements where imports got inserted incorrectly

import { $, glob } from 'zx';
import fs from 'fs/promises';
import path from 'path';

console.log('🔧 Starting import syntax fix...');

// Find all TypeScript files with potential import issues
const targetFiles = await glob([
  'src/**/*.ts',
  'src/**/*.js'
], { ignore: ['node_modules/**', 'dist/**', '.svelte-kit/**'] });

console.log(`📁 Found ${targetFiles.length} files to check`);

let totalFixes = 0;
let processedFiles = 0;

for (const file of targetFiles) {
  try {
    const content = await fs.readFile(file, 'utf-8');
    let modified = content;
    let fileFixes = 0;

    // Fix malformed import statements like:
    // import {
    // import crypto from "crypto";
    //   SomeType,
    // } from "module";
    
    // Pattern 1: import statement split incorrectly
    modified = modified.replace(
      /import\s*\{\s*\nimport\s+([^;]+);\s*\n([^}]*)\}\s*from\s*["']([^"']+)["'];/g,
      (match, firstImport, remainingImports, modulePath) => {
        return `${firstImport};\nimport {\n${remainingImports}} from "${modulePath}";`;
      }
    );

    // Pattern 2: More complex malformed imports
    modified = modified.replace(
      /import\s+(.*?)\s+from\s+["']([^"']+)["'];\s*\nimport\s*\{\s*([^}]*)\}\s*from\s*["']([^"']+)["'];/g,
      (match, firstImport, firstModule, typeImports, secondModule) => {
        if (firstModule === secondModule) {
          // Combine imports from same module
          return `import ${firstImport}, {\n${typeImports}} from "${firstModule}";`;
        } else {
          // Keep separate
          return `import ${firstImport} from "${firstModule}";\nimport {\n${typeImports}} from "${secondModule}";`;
        }
      }
    );

    // Pattern 3: Fix type import mixed with regular import
    modified = modified.replace(
      /import\s+type\s*\{\s*\nimport\s+([^;]+);\s*\n([^}]*)\}\s*from\s*["']([^"']+)["'];/g,
      (match, firstImport, remainingTypes, modulePath) => {
        return `${firstImport};\nimport type {\n${remainingTypes}} from "${modulePath}";`;
      }
    );

    // Pattern 4: Fix orphaned closing braces after imports
    modified = modified.replace(
      /import\s+([^;]+);\s*\n\s*([^}]*)\}\s*from\s*["']([^"']+)["'];/g,
      (match, firstImport, orphanedContent) => {
        if (orphanedContent.trim()) {
          return `${firstImport};\n// Orphaned content: ${orphanedContent.trim()}`;
        } else {
          return `${firstImport};`;
        }
      }
    );

    // Fix duplicate import statements for same module
    const lines = modified.split('\n');
    const importMap = new Map();
    const newLines = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const importMatch = line.match(/^import\s+(.+?)\s+from\s+["']([^"']+)["'];?$/);
      
      if (importMatch) {
        const [, imports, module] = importMatch;
        if (importMap.has(module)) {
          // Combine with existing import
          const existing = importMap.get(module);
          importMap.set(module, `${existing}, ${imports}`);
        } else {
          importMap.set(module, imports);
        }
      } else {
        newLines.push(line);
      }
    }
    
    // Reconstruct file with fixed imports
    const importLines = Array.from(importMap.entries()).map(([module, imports]) => {
      return `import ${imports} from "${module}";`;
    });
    
    if (importLines.length > 0) {
      modified = importLines.join('\n') + '\n' + newLines.join('\n');
    }

    // Check if changes were made
    if (content !== modified) {
      await fs.writeFile(file, modified);
      fileFixes = 1;
      console.log(`✅ Fixed import syntax in ${path.basename(file)}`);
      totalFixes += fileFixes;
      processedFiles++;
    }

  } catch (error) {
    console.log(`❌ Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Import syntax fix complete!`);
console.log(`📊 Processed: ${processedFiles} files`);
console.log(`🔧 Total fixes: ${totalFixes}`);
console.log(`✅ Ready for recompilation`);