#!/usr/bin/env node

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

console.log('🚀 Starting Comprehensive FlashAttention2 Error Fixing System...');

// Error patterns and their fixes
const errorPatterns = [
  {
    pattern: /error TS1005: ',' expected/,
    category: 'syntax',
    fix: (line, file) => {
      // Fix malformed import statements
      if (line.includes('import {') && !line.includes('}')) {
        return line.replace(/import {[^}]*$/, 'import { /* fix import */ } from');
      }
      return line;
    }
  },
  {
    pattern: /error TS1003: Identifier expected/,
    category: 'syntax',
    fix: (line, file) => {
      // Fix orphaned content patterns
      if (line.includes('// Orphaned content:')) {
        return ''; // Remove orphaned content lines
      }
      return line;
    }
  },
  {
    pattern: /error TS1136: Property assignment expected/,
    category: 'syntax',
    fix: (line, file) => {
      // Fix malformed object properties
      if (line.includes(',') && !line.includes(':')) {
        return line.replace(/,\s*([^:,]+)$/, ', $1: undefined');
      }
      return line;
    }
  },
  {
    pattern: /error TS2307: Cannot find module/,
    category: 'import',
    fix: (line, file) => {
      // Fix import paths
      if (line.includes('import') && line.includes('.js')) {
        return line.replace('.js', '.ts');
      }
      return line;
    }
  },
  {
    pattern: /export let/,
    category: 'svelte5',
    fix: (line, file) => {
      if (file.endsWith('.svelte')) {
        // Convert to Svelte 5 $props() syntax
        const match = line.match(/export let (\w+)(?:\s*=\s*(.+?))?;?/);
        if (match) {
          const [, propName, defaultValue] = match;
          return `let { ${propName}${defaultValue ? ` = ${defaultValue}` : ''} } = $props();`;
        }
      }
      return line;
    }
  }
];

// Get all TypeScript files
async function getAllTSFiles() {
  const { stdout } = await execAsync('find src -name "*.ts" -o -name "*.svelte" | head -100');
  return stdout.trim().split('\n').filter(Boolean);
}

// Parse TypeScript errors
async function getTypeScriptErrors() {
  try {
    const { stdout } = await execAsync('npx tsc --noEmit --skipLibCheck 2>&1 || true');
    const errorLines = stdout.split('\n').filter(line => 
      line.includes('TS') && (line.includes('error') || line.includes('warning'))
    );
    
    return errorLines.map(line => {
      const match = line.match(/(.+?)\((\d+),(\d+)\):\s*error\s+(TS\d+):\s*(.+)/);
      if (match) {
        const [, file, lineNum, col, code, message] = match;
        return {
          file: file.trim(),
          line: parseInt(lineNum),
          column: parseInt(col),
          code,
          message: message.trim(),
          fullLine: line
        };
      }
      return null;
    }).filter(Boolean);
  } catch (error) {
    console.error('Error getting TypeScript errors:', error);
    return [];
  }
}

// Fix file content
async function fixFileContent(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n');
    let fixed = false;
    
    const newLines = lines.map((line, index) => {
      let newLine = line;
      
      // Apply error pattern fixes
      for (const pattern of errorPatterns) {
        if (pattern.pattern.test(line) || 
            (pattern.category === 'svelte5' && line.includes('export let')) ||
            line.includes('// Orphaned content:')) {
          const fixedLine = pattern.fix(line, filePath);
          if (fixedLine !== line) {
            console.log(`  Fixed line ${index + 1}: ${pattern.category}`);
            newLine = fixedLine;
            fixed = true;
          }
        }
      }
      
      // Additional common fixes
      if (newLine.includes('import {') && newLine.includes(', ,')) {
        newLine = newLine.replace(', ,', ',');
        fixed = true;
      }
      
      if (newLine.includes('} from""') || newLine.includes('} from "')) {
        if (!newLine.includes('"$lib/') && !newLine.includes('"@sveltejs/')) {
          newLine = newLine.replace('} from""', '} from "$lib/utils"');
          newLine = newLine.replace('} from "', '} from "$lib/utils/');
          fixed = true;
        }
      }
      
      return newLine;
    });
    
    if (fixed) {
      await fs.writeFile(filePath, newLines.join('\n'));
      console.log(`✅ Fixed file: ${filePath}`);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`Error fixing file ${filePath}:`, error.message);
    return false;
  }
}

// Main error fixing process
async function runComprehensiveErrorFix() {
  console.log('📊 Analyzing current errors...');
  
  let currentErrors = await getTypeScriptErrors();
  let iteration = 0;
  const maxIterations = 10;
  let fixedFiles = 0;
  
  console.log(`🎯 Found ${currentErrors.length} TypeScript errors`);
  
  while (currentErrors.length > 0 && iteration < maxIterations) {
    iteration++;
    console.log(`\n🔄 Iteration ${iteration}/${maxIterations} - Fixing errors...`);
    
    // Group errors by file
    const errorsByFile = currentErrors.reduce((acc, error) => {
      if (!acc[error.file]) acc[error.file] = [];
      acc[error.file].push(error);
      return acc;
    }, {});
    
    // Fix files with most errors first
    const sortedFiles = Object.entries(errorsByFile)
      .sort(([,a], [,b]) => b.length - a.length)
      .slice(0, 20); // Fix top 20 problematic files per iteration
    
    let iterationFixes = 0;
    
    for (const [filePath, fileErrors] of sortedFiles) {
      if (await fixFileContent(filePath)) {
        iterationFixes++;
        fixedFiles++;
      }
    }
    
    console.log(`✅ Fixed ${iterationFixes} files in iteration ${iteration}`);
    
    // Re-analyze errors
    const newErrors = await getTypeScriptErrors();
    const errorReduction = currentErrors.length - newErrors.length;
    
    console.log(`📉 Error reduction: ${errorReduction} (${currentErrors.length} → ${newErrors.length})`);
    
    if (errorReduction <= 0) {
      console.log('⚠️ No more automatic fixes available');
      break;
    }
    
    currentErrors = newErrors;
    
    // Small delay to avoid overwhelming the system
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  console.log(`\n🎉 Comprehensive error fixing complete!`);
  console.log(`📊 Final results:`);
  console.log(`   - Files fixed: ${fixedFiles}`);
  console.log(`   - Remaining errors: ${currentErrors.length}`);
  console.log(`   - Iterations: ${iteration}`);
  
  if (currentErrors.length > 0) {
    console.log(`\n📋 Remaining error categories:`);
    const categories = {};
    currentErrors.forEach(error => {
      const category = error.code || 'unknown';
      categories[category] = (categories[category] || 0) + 1;
    });
    
    Object.entries(categories)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .forEach(([category, count]) => {
        console.log(`   - ${category}: ${count} errors`);
      });
  }
}

// Run the comprehensive error fixer
runComprehensiveErrorFix().catch(console.error);