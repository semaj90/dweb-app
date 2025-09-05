#!/usr/bin/env node

import { execSync } from 'child_process';
import { readdirSync, readFileSync, writeFileSync, statSync } from 'fs';
import { join } from 'path';

const ROOT_DIR = process.cwd();

console.log('🔧 Starting quick Svelte syntax fixes...');

// Common syntax issues to fix
const fixes = [
  // Fix $derived syntax - remove trailing semicolons
  { pattern: /\$derived\([^)]+\);?\)\;/g, replacement: (match) => match.replace(/;\)$/, ')') },

  // Fix duplicate prop destructuring
  { pattern: /let \{ ([^}]+) = \$bindable\(\) \} = \$props\(\); \/\/ ([^;]+);?\s*let \{ \1 = \$bindable\(\) \} = \$props\(\);/g, replacement: 'let { $1 = $bindable() } = $props(); // $2' },

  // Fix malformed derived expressions
  { pattern: /\$derived\([^)]*;?\);/g, replacement: (match) => match.replace(/;?\);$/, ')') },

  // Fix missing closing parentheses in $derived
  { pattern: /\$derived\([^)]+;/g, replacement: (match) => match.replace(/;$/, ')') },

  // Remove orphaned import statements
  { pattern: /^import\s+['"][^'"]*['"]\s*$/gm, replacement: '' },

  // Fix incomplete prop destructuring
  { pattern: /let \{\s*\} = \$props\(\);/g, replacement: '// Props destructured above' },

  // Fix incomplete derived expressions
  { pattern: /\$derived\(\[\s*\);/g, replacement: '$derived([])' },
  { pattern: /\$derived\([^)]*\(\s*\);/g, replacement: (match) => match.replace(/\(\s*\);$/, '())') }
];

function fixFile(filePath) {
  try {
    let content = readFileSync(filePath, 'utf-8');
    let changed = false;

    fixes.forEach(fix => {
      const newContent = content.replace(fix.pattern, fix.replacement);
      if (newContent !== content) {
        content = newContent;
        changed = true;
      }
    });

    if (changed) {
      writeFileSync(filePath, content, 'utf-8');
      console.log(`✅ Fixed: ${filePath}`);
      return true;
    }
  } catch (error) {
    console.warn(`⚠️ Could not fix ${filePath}: ${error.message}`);
  }

  return false;
}

function processDirectory(dir) {
  let fixedCount = 0;

  try {
    const entries = readdirSync(dir);

    for (const entry of entries) {
      const fullPath = join(dir, entry);
      const stat = statSync(fullPath);

      if (stat.isDirectory() && !['node_modules', '.svelte-kit', '.git', 'dist'].includes(entry)) {
        fixedCount += processDirectory(fullPath);
      } else if (entry.endsWith('.svelte') && stat.isFile()) {
        if (fixFile(fullPath)) {
          fixedCount++;
        }
      }
    }
  } catch (error) {
    console.warn(`⚠️ Could not process directory ${dir}: ${error.message}`);
  }

  return fixedCount;
}

// Process src directory
const srcDir = join(ROOT_DIR, 'src');
const fixedFiles = processDirectory(srcDir);

console.log(`🎯 Quick fixes completed: ${fixedFiles} files updated`);

// Also fix the duplicate schema export
try {
  const schemaPath = join(ROOT_DIR, 'src/lib/server/db/schema-postgres.ts');
  let schemaContent = readFileSync(schemaPath, 'utf-8');

  // Remove duplicate usersRelations export
  const lines = schemaContent.split('\n');
  const seenExports = new Set();
  const cleanedLines = lines.filter(line => {
    const exportMatch = line.match(/^export const (\w+Relations)/);
    if (exportMatch) {
      const exportName = exportMatch[1];
      if (seenExports.has(exportName)) {
        console.log(`📋 Removing duplicate export: ${exportName}`);
        return false;
      }
      seenExports.add(exportName);
    }
    return true;
  });

  writeFileSync(schemaPath, cleanedLines.join('\n'), 'utf-8');
  console.log('✅ Fixed duplicate schema exports');
} catch (error) {
  console.warn('⚠️ Could not fix schema exports:', error.message);
}

console.log('🚀 Quick syntax fixes complete. Try running npm run dev again.');
