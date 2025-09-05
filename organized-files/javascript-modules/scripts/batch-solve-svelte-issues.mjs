#!/usr/bin/env node

// Automated Svelte Issues Batch Solver
// Fixes accessibility warnings and unused CSS selectors across the entire project

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

console.log('🔧 Starting batch Svelte issues solver...\n');

const projectRoot = process.cwd();
const srcDir = join(projectRoot, 'sveltekit-frontend', 'src');

// Counters for reporting
let filesProcessed = 0;
let issuesFixed = 0;
let filesWithIssues = [];

/**
 * Get all Svelte files recursively
 */
function getAllSvelteFiles(dir) {
  const files = [];
  
  try {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Skip node_modules and build directories
        if (!['node_modules', 'build', 'dist', '.svelte-kit'].includes(item)) {
          files.push(...getAllSvelteFiles(fullPath));
        }
      } else if (extname(item) === '.svelte') {
        files.push(fullPath);
      }
    }
  } catch (error) {
    console.warn(`⚠️ Could not read directory ${dir}: ${error.message}`);
  }
  
  return files;
}

/**
 * Fix accessibility issues in Svelte files
 */
function fixAccessibilityIssues(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Pattern to match unassociated labels
  const labelWithoutForPattern = /<label\s+class="[^"]*"\s*>\s*([^<]+)\s*<\/label>\s*<(input|select|textarea)/gs;
  
  // Fix: Add proper for/id associations
  fixed = fixed.replace(labelWithoutForPattern, (match, labelText, inputType) => {
    const controlId = labelText
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '-')
      .replace(/^-+|-+$/g, '');
    
    const newLabel = match.replace(
      /<label\s+class="([^"]*)"\s*>/,
      `<label for="${controlId}" class="$1">`
    );
    
    const newControl = newLabel.replace(
      /<(input|select|textarea)(\s[^>]*)?>/,
      `<$1 id="${controlId}"$2>`
    );
    
    issueCount++;
    console.log(`  ✅ Fixed label association: "${labelText.trim()}" -> "${controlId}"`);
    
    return newControl;
  });
  
  // Pattern to match labels followed by form controls (more flexible)
  const labelBeforeControlPattern = /<label\s+(?!for=)[^>]*class="[^"]*"[^>]*>\s*([^<]+)\s*<\/label>\s*<(input|select|textarea)(?!\s+id=)([^>]*)>/gs;
  
  fixed = fixed.replace(labelBeforeControlPattern, (match, labelText, controlType, controlAttrs) => {
    const controlId = labelText
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '-')
      .replace(/^-+|-+$/g, '') + '-' + controlType;
    
    const newMatch = match.replace(
      /<label\s+([^>]*class="[^"]*"[^>]*)>/,
      `<label for="${controlId}" $1>`
    ).replace(
      `<${controlType}${controlAttrs}>`,
      `<${controlType} id="${controlId}"${controlAttrs}>`
    );
    
    issueCount++;
    console.log(`  ✅ Fixed label-control association: "${labelText.trim()}" -> "${controlId}"`);
    
    return newMatch;
  });
  
  return { content: fixed, issueCount };
}

/**
 * Fix unused CSS selectors
 */
function fixUnusedCSSSelectors(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Extract style block
  const styleMatch = fixed.match(/<style[^>]*>([\s\S]*?)<\/style>/);
  if (!styleMatch) return { content: fixed, issueCount: 0 };
  
  const styleContent = styleMatch[1];
  const htmlContent = fixed.replace(/<style[^>]*>[\s\S]*?<\/style>/, '');
  
  // Find CSS selectors
  const selectorPattern = /\.([a-zA-Z][a-zA-Z0-9_-]*)\s*\{[^}]*\}/g;
  const selectors = [];
  let match;
  
  while ((match = selectorPattern.exec(styleContent)) !== null) {
    selectors.push({
      fullMatch: match[0],
      className: match[1],
      index: match.index
    });
  }
  
  // Check which selectors are unused
  const unusedSelectors = selectors.filter(selector => {
    const classUsagePattern = new RegExp(`class="[^"]*\\b${selector.className}\\b[^"]*"`, 'g');
    const classUsagePattern2 = new RegExp(`class:${selector.className}`, 'g');
    const bindClassPattern = new RegExp(`class:${selector.className}=`, 'g');
    
    return !classUsagePattern.test(htmlContent) && 
           !classUsagePattern2.test(htmlContent) && 
           !bindClassPattern.test(htmlContent);
  });
  
  // Remove unused selectors
  if (unusedSelectors.length > 0) {
    let newStyleContent = styleContent;
    
    // Sort by index in reverse order to avoid index shifting
    unusedSelectors.sort((a, b) => b.index - a.index);
    
    for (const selector of unusedSelectors) {
      const beforeStyle = newStyleContent.substring(0, selector.index);
      const afterStyle = newStyleContent.substring(selector.index + selector.fullMatch.length);
      newStyleContent = beforeStyle + afterStyle;
      
      issueCount++;
      console.log(`  ✅ Removed unused CSS selector: .${selector.className}`);
    }
    
    // Replace style block
    fixed = fixed.replace(
      /<style[^>]*>[\s\S]*?<\/style>/,
      `<style>\n  /* @unocss-include */\n${newStyleContent}\n</style>`
    );
  }
  
  return { content: fixed, issueCount };
}

/**
 * Process a single Svelte file
 */
function processSvelteFile(filePath) {
  try {
    const originalContent = readFileSync(filePath, 'utf-8');
    let modifiedContent = originalContent;
    let totalIssues = 0;
    
    console.log(`\n📁 Processing: ${filePath.replace(projectRoot, '.')}`);
    
    // Fix accessibility issues
    const accessibilityResult = fixAccessibilityIssues(modifiedContent, filePath);
    modifiedContent = accessibilityResult.content;
    totalIssues += accessibilityResult.issueCount;
    
    // Fix unused CSS selectors
    const cssResult = fixUnusedCSSSelectors(modifiedContent, filePath);
    modifiedContent = cssResult.content;
    totalIssues += cssResult.issueCount;
    
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
 * Main batch processing function
 */
async function batchSolveSvelteIssues() {
  console.log(`🔍 Scanning for Svelte files in: ${srcDir}\n`);
  
  if (!statSync(srcDir).isDirectory()) {
    console.error(`❌ Source directory not found: ${srcDir}`);
    console.log('💡 Make sure you\'re running this from the project root');
    process.exit(1);
  }
  
  const svelteFiles = getAllSvelteFiles(srcDir);
  
  if (svelteFiles.length === 0) {
    console.log('⚠️ No Svelte files found to process');
    return;
  }
  
  console.log(`📊 Found ${svelteFiles.length} Svelte files to process\n`);
  
  // Process each file
  for (const filePath of svelteFiles) {
    processSvelteFile(filePath);
  }
  
  // Generate report
  console.log('\n' + '='.repeat(60));
  console.log('📋 BATCH PROCESSING REPORT');
  console.log('='.repeat(60));
  console.log(`📁 Files processed: ${filesProcessed}`);
  console.log(`🔧 Total issues fixed: ${issuesFixed}`);
  console.log(`📄 Files with issues: ${filesWithIssues.length}`);
  
  if (filesWithIssues.length > 0) {
    console.log('\n📝 Files that were modified:');
    filesWithIssues.forEach(file => {
      const relativePath = file.path.replace(projectRoot, '.');
      console.log(`  • ${relativePath} (${file.issueCount} issues fixed)`);
    });
  }
  
  console.log('\n✨ Batch processing complete!');
  
  // Suggest next steps
  console.log('\n💡 Next steps:');
  console.log('  1. Run svelte-check to verify all issues are resolved');
  console.log('  2. Test your application to ensure functionality is preserved');
  console.log('  3. Commit the changes if everything looks good');
  
  if (issuesFixed > 0) {
    console.log('\n🚀 Suggested commands:');
    console.log('  npm run check           # Verify Svelte compilation');
    console.log('  npm run dev             # Start development server');
    console.log('  npm run test            # Run test suite');
  }
}

/**
 * Create a more comprehensive fix for common Svelte patterns
 */
function createComprehensiveFixes(content, filePath) {
  let fixed = content;
  let issueCount = 0;
  
  // Fix pattern: Labels without proper association
  const labelPatterns = [
    // Pattern 1: Label with class but no for attribute
    {
      pattern: /<label\s+class="([^"]*)"[^>]*>\s*([^<]+)\s*<\/label>\s*<(input|select|textarea)([^>]*)>/gs,
      fix: (match, className, labelText, controlType, controlAttrs) => {
        const id = labelText.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
        const hasId = /\bid\s*=/.test(controlAttrs);
        
        if (!hasId) {
          issueCount++;
          console.log(`  ✅ Fixed label: "${labelText.trim()}" -> ${id}`);
          return `<label for="${id}" class="${className}">${labelText}</label><${controlType} id="${id}"${controlAttrs}>`;
        }
        return match;
      }
    },
    
    // Pattern 2: Label inside div with form control
    {
      pattern: /<div[^>]*>\s*<label\s+class="([^"]*)"[^>]*>\s*([^<]+)\s*<\/label>\s*<(input|select|textarea)([^>]*)>\s*<\/div>/gs,
      fix: (match, className, labelText, controlType, controlAttrs) => {
        const id = labelText.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
        const hasId = /\bid\s*=/.test(controlAttrs);
        
        if (!hasId) {
          issueCount++;
          console.log(`  ✅ Fixed wrapped label: "${labelText.trim()}" -> ${id}`);
          return match.replace(
            `<label class="${className}"`,
            `<label for="${id}" class="${className}"`
          ).replace(
            `<${controlType}${controlAttrs}>`,
            `<${controlType} id="${id}"${controlAttrs}>`
          );
        }
        return match;
      }
    }
  ];
  
  // Apply all patterns
  for (const { pattern, fix } of labelPatterns) {
    fixed = fixed.replace(pattern, fix);
  }
  
  return { content: fixed, issueCount };
}

// Run the batch solver
if (import.meta.url === `file://${process.argv[1]}`) {
  batchSolveSvelteIssues().catch(error => {
    console.error('❌ Batch processing failed:', error);
    process.exit(1);
  });
}

export { batchSolveSvelteIssues, processSvelteFile };
