#!/usr/bin/env node

/**
 * TypeScript Error Fixer for Legal AI Platform
 * Addresses common TypeScript errors systematically
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🔧 TypeScript Error Fixer Starting...\n');

// Common TypeScript error patterns and fixes
const errorFixes = [
  {
    pattern: /\/\/ @ts-nocheck/g,
    replacement: '',
    description: 'Remove @ts-nocheck directives'
  },
  {
    pattern: /\/\/ @ts-ignore/g,
    replacement: '',
    description: 'Remove @ts-ignore directives'
  },
  {
    pattern: /any\[\]/g,
    replacement: 'unknown[]',
    description: 'Replace any[] with unknown[]'
  },
  {
    pattern: /: any/g,
    replacement: ': unknown',
    description: 'Replace : any with : unknown'
  },
  {
    pattern: /const \w+: any =/g,
    replacement: (match) => match.replace(': any', ': unknown'),
    description: 'Replace const declarations with any type'
  },
  {
    pattern: /let \w+: any =/g,
    replacement: (match) => match.replace(': any', ': unknown'),
    description: 'Replace let declarations with any type'
  },
  {
    pattern: /function \w+\(.*\): any/g,
    replacement: (match) => match.replace(': any', ': unknown'),
    description: 'Replace function return types'
  }
];

// Files to skip (already processed or problematic)
const skipFiles = [
  'node_modules',
  '.svelte-kit',
  'dist',
  'build',
  '.git'
];

// Recursively find TypeScript files
function findTsFiles(dir, files = []) {
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (skipFiles.includes(item)) continue;
    
    if (stat.isDirectory()) {
      findTsFiles(fullPath, files);
    } else if (item.endsWith('.ts') || item.endsWith('.svelte')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

// Apply fixes to a file
function fixFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let originalContent = content;
    let fixesApplied = 0;
    
    for (const fix of errorFixes) {
      if (typeof fix.replacement === 'function') {
        const newContent = content.replace(fix.pattern, fix.replacement);
        if (newContent !== content) {
          content = newContent;
          fixesApplied++;
        }
      } else {
        const newContent = content.replace(fix.pattern, fix.replacement);
        if (newContent !== content) {
          content = newContent;
          fixesApplied++;
        }
      }
    }
    
    if (fixesApplied > 0) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`✅ Fixed ${fixesApplied} issues in ${path.relative(process.cwd(), filePath)}`);
      return fixesApplied;
    }
    
    return 0;
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
    return 0;
  }
}

// Main execution
function main() {
  const startDir = process.cwd();
  console.log(`📁 Scanning directory: ${startDir}\n`);
  
  try {
    const tsFiles = findTsFiles(startDir);
    console.log(`📊 Found ${tsFiles.length} TypeScript/Svelte files\n`);
    
    let totalFixes = 0;
    let processedFiles = 0;
    
    for (const file of tsFiles) {
      const fixes = fixFile(file);
      if (fixes > 0) {
        totalFixes += fixes;
        processedFiles++;
      }
    }
    
    console.log(`\n🎯 Summary:`);
    console.log(`   Files processed: ${processedFiles}`);
    console.log(`   Total fixes applied: ${totalFixes}`);
    
    if (totalFixes > 0) {
      console.log(`\n🔄 Next steps:`);
      console.log(`   1. Run: npm run check:typescript`);
      console.log(`   2. Address remaining specific errors`);
      console.log(`   3. Test compilation: npm run build`);
    }
    
  } catch (error) {
    console.error('❌ Error during processing:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { findTsFiles, fixFile, errorFixes };
