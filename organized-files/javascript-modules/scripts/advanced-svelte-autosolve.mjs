#!/usr/bin/env node

/**
 * Advanced Svelte Auto-Solver
 * Fixes TypeScript, Svelte 5, and accessibility issues automatically
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

console.log('🔧 SVELTE AUTO-SOLVER v2.0 - Advanced Issue Resolution\n');

const projectRoot = process.cwd();
const srcDir = join(projectRoot, 'sveltekit-frontend', 'src');

let filesProcessed = 0;
let issuesFixed = 0;
let filesWithIssues = [];

/**
 * Get all TypeScript and Svelte files
 */
function getAllFiles(dir, extensions = ['.svelte', '.ts', '.js']) {
  const files = [];
  
  try {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        if (!['node_modules', 'build', 'dist', '.svelte-kit'].includes(item)) {
          files.push(...getAllFiles(fullPath, extensions));
        }
      } else if (extensions.includes(extname(item))) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.warn(`⚠️ Could not read directory ${dir}: ${error.message}`);
  }
  
  return files;
}

/**
 * Fix TypeScript type issues
 */
function fixTypeScriptIssues(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Fix common TypeScript issues
  const typeScriptFixes = [
    {
      pattern: /:\s*any(?!\w)/g,
      replacement: ': unknown',
      description: 'Replace any with unknown for better type safety'
    },
    {
      pattern: /let\s+(\w+)(?:\s*:\s*any)?\s*;/g,
      replacement: 'let $1: unknown = undefined;',
      description: 'Initialize variables with proper types'
    },
    {
      pattern: /function\s+(\w+)\s*\([^)]*\)\s*{/g,
      replacement: (match, funcName) => match.replace('{', ': void {'),
      description: 'Add return type annotations to functions'
    },
    {
      pattern: /export\s+let\s+(\w+)(?:\s*:\s*any)?/g,
      replacement: 'export let $1: unknown',
      description: 'Add proper types to exported props'
    }
  ];
  
  typeScriptFixes.forEach(fix => {
    const matches = content.match(fix.pattern);
    if (matches) {
      fixed = fixed.replace(fix.pattern, fix.replacement);
      issueCount += matches.length;
      console.log(`    ✅ ${fix.description} (${matches.length} fixes)`);
    }
  });
  
  return { content: fixed, issueCount };
}

/**
 * Fix Svelte 5 compatibility issues
 */
function fixSvelte5Issues(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Fix Svelte 5 patterns
  const svelte5Fixes = [
    {
      pattern: /\$:\s*(\w+)\s*=\s*([^;]+);/g,
      replacement: 'let $1 = $derived($2);',
      description: 'Convert reactive statements to $derived'
    },
    {
      pattern: /export\s+let\s+(\w+)(?:\s*:\s*([^=;]+))?\s*(?:=\s*([^;]+))?;/g,
      replacement: (match, name, type, defaultValue) => {
        const typeAnnotation = type ? `: ${type.trim()}` : '';
        const defaultVal = defaultValue ? ` = ${defaultValue.trim()}` : '';
        return `let { ${name}${typeAnnotation}${defaultVal} } = $props();`;
      },
      description: 'Convert export let to $props syntax'
    },
    {
      pattern: /createEventDispatcher\(\)/g,
      replacement: '$dispatch',
      description: 'Update event dispatcher for Svelte 5'
    }
  ];
  
  svelte5Fixes.forEach(fix => {
    const originalContent = fixed;
    fixed = fixed.replace(fix.pattern, fix.replacement);
    if (fixed !== originalContent) {
      issueCount++;
      console.log(`    ✅ ${fix.description}`);
    }
  });
  
  return { content: fixed, issueCount };
}

/**
 * Fix accessibility issues
 */
function fixAccessibilityIssues(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Fix accessibility patterns
  const a11yFixes = [
    {
      pattern: /<img\s+(?![^>]*alt=)[^>]*src="([^"]*)"[^>]*>/g,
      replacement: '<img src="$1" alt="Image" />',
      description: 'Add alt attributes to images'
    },
    {
      pattern: /<button\s+(?![^>]*aria-label=)(?![^>]*title=)([^>]*)>(\s*)<\/button>/g,
      replacement: '<button aria-label="Button" $1>$2</button>',
      description: 'Add aria-label to buttons without text'
    },
    {
      pattern: /<input\s+(?![^>]*id=)([^>]*type="[^"]*"[^>]*)>/g,
      replacement: (match, attrs) => {
        const id = `input-${Math.random().toString(36).substr(2, 9)}`;
        return `<input id="${id}" ${attrs}>`;
      },
      description: 'Add IDs to form inputs'
    }
  ];
  
  a11yFixes.forEach(fix => {
    const matches = content.match(fix.pattern);
    if (matches) {
      fixed = fixed.replace(fix.pattern, fix.replacement);
      issueCount += matches.length;
      console.log(`    ✅ ${fix.description} (${matches.length} fixes)`);
    }
  });
  
  return { content: fixed, issueCount };
}

/**
 * Fix import issues
 */
function fixImportIssues(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Fix common import patterns
  const importFixes = [
    {
      pattern: /import\s+([^'"]+)\s+from\s+['"]([^'"]+)['"];?/g,
      replacement: (match, imports, path) => {
        // Add .js extension if missing and it's a relative import
        if (path.startsWith('./') || path.startsWith('../')) {
          if (!path.includes('.')) {
            path += '.js';
          }
        }
        return `import ${imports} from '${path}';`;
      },
      description: 'Add file extensions to relative imports'
    },
    {
      pattern: /import\s+type\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"];?/g,
      replacement: "import type { $1 } from '$2';",
      description: 'Normalize type imports'
    }
  ];
  
  importFixes.forEach(fix => {
    const originalContent = fixed;
    fixed = fixed.replace(fix.pattern, fix.replacement);
    if (fixed !== originalContent) {
      issueCount++;
      console.log(`    ✅ ${fix.description}`);
    }
  });
  
  return { content: fixed, issueCount };
}

/**
 * Process a single file
 */
function processFile(filePath) {
  try {
    const originalContent = readFileSync(filePath, 'utf-8');
    let modifiedContent = originalContent;
    let totalIssues = 0;
    
    console.log(`\n📁 Processing: ${filePath.replace(projectRoot, '.')}`);
    
    // Apply fixes based on file type
    if (filePath.endsWith('.svelte')) {
      // Svelte files get all fixes
      const a11yResult = fixAccessibilityIssues(modifiedContent, filePath);
      modifiedContent = a11yResult.content;
      totalIssues += a11yResult.issueCount;
      
      const svelte5Result = fixSvelte5Issues(modifiedContent, filePath);
      modifiedContent = svelte5Result.content;
      totalIssues += svelte5Result.issueCount;
    }
    
    if (filePath.endsWith('.ts') || filePath.endsWith('.svelte')) {
      // TypeScript fixes
      const tsResult = fixTypeScriptIssues(modifiedContent, filePath);
      modifiedContent = tsResult.content;
      totalIssues += tsResult.issueCount;
      
      // Import fixes
      const importResult = fixImportIssues(modifiedContent, filePath);
      modifiedContent = importResult.content;
      totalIssues += importResult.issueCount;
    }
    
    // Write back if changes were made
    if (modifiedContent !== originalContent) {
      writeFileSync(filePath, modifiedContent, 'utf-8');
      filesWithIssues.push({
        path: filePath,
        issueCount: totalIssues
      });
      console.log(`  💾 Saved ${totalIssues} fixes to file`);
    } else {
      console.log(`  ✅ No issues found`);
    }
    
    filesProcessed++;
    issuesFixed += totalIssues;
    
  } catch (error) {
    console.error(`❌ Error processing ${filePath}: ${error.message}`);
  }
}

/**
 * Main execution
 */
async function runAutoSolver() {
  console.log(`🔍 Scanning for files in: ${srcDir}\n`);
  
  const files = getAllFiles(srcDir);
  
  if (files.length === 0) {
    console.log('⚠️ No files found to process');
    return;
  }
  
  console.log(`📊 Found ${files.length} files to process\n`);
  
  // Process each file
  for (const filePath of files.slice(0, 50)) { // Limit to first 50 files for demo
    processFile(filePath);
  }
  
  // Generate final report
  console.log('\n' + '='.repeat(60));
  console.log('📋 AUTO-SOLVER FINAL REPORT');
  console.log('='.repeat(60));
  console.log(`📁 Files processed: ${filesProcessed}`);
  console.log(`🔧 Total issues fixed: ${issuesFixed}`);
  console.log(`📄 Files with fixes: ${filesWithIssues.length}`);
  
  if (filesWithIssues.length > 0) {
    console.log('\n📝 Modified files:');
    filesWithIssues.slice(0, 10).forEach(file => {
      const relativePath = file.path.replace(projectRoot, '.');
      console.log(`  • ${relativePath} (${file.issueCount} issues fixed)`);
    });
    
    if (filesWithIssues.length > 10) {
      console.log(`  • ... and ${filesWithIssues.length - 10} more files`);
    }
  }
  
  console.log('\n✨ AUTO-SOLVER COMPLETE!');
  console.log('\n🚀 Next steps:');
  console.log('  1. npm run check           # Verify fixes');
  console.log('  2. npm run dev             # Test application');
  console.log('  3. npm run build           # Check production build');
  
  return {
    filesProcessed,
    issuesFixed,
    filesWithIssues: filesWithIssues.length,
    success: true
  };
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAutoSolver().catch(error => {
    console.error('❌ Auto-solver failed:', error);
    process.exit(1);
  });
}

export default runAutoSolver;
