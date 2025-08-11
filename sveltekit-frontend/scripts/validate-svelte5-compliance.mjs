#!/usr/bin/env node
/**
 * Svelte 5 Compliance Validator
 * Based on Context7 MCP Best Practices
 */

import { readFileSync, readdirSync, statSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = join(__dirname, '../src');

console.log('🔍 Svelte 5 Compliance Validator');
console.log('================================');

let validationStats = {
  filesScanned: 0,
  compliantFiles: 0,
  issuesFound: 0,
  issuesFixed: 0,
  errors: []
};

// Compliance checks
const COMPLIANCE_CHECKS = [
  {
    name: 'No createEventDispatcher',
    severity: 'error',
    detect: /createEventDispatcher/g,
    message: 'Component still uses createEventDispatcher - should use callback props',
    autofix: false
  },
  {
    name: 'No export let',
    severity: 'warning', 
    detect: /export\s+let\s+\w+/g,
    message: 'Component uses export let - should use $props() destructuring',
    autofix: false
  },
  {
    name: 'Unsafe event handlers',
    severity: 'error',
    detect: /on\w+={(.*?)e\.target\.(?:value|checked)(?!\s+as\s+HTML)/g,
    message: 'Event handler lacks TypeScript casting - should cast e.target',
    autofix: false
  },
  {
    name: 'Legacy reactive statements',
    severity: 'warning',
    detect: /\$:\s+(?!if|while|for)/g,
    message: 'Legacy reactive statement found - consider using $derived or $effect',
    autofix: false
  },
  {
    name: 'Placeholder CSS classes',
    severity: 'info',
    detect: /class="container mx-auto px-4"/g,
    message: 'Placeholder CSS class found - should use proper Tailwind classes',
    autofix: true,
    fix: (content) => content.replace(/class="container mx-auto px-4"/g, 'class="space-y-4"')
  },
  {
    name: 'Modern Svelte 5 patterns',
    severity: 'success',
    detect: /\$(?:state|derived|effect|props|bindable)\(/g,
    message: 'Uses modern Svelte 5 runes',
    autofix: false,
    isGoodPattern: true
  }
];

function findSvelteFiles(dir) {
  let files = [];
  
  try {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
        files = files.concat(findSvelteFiles(fullPath));
      } else if (stat.isFile() && item.endsWith('.svelte')) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    validationStats.errors.push(`Error reading directory ${dir}: ${error.message}`);
  }
  
  return files;
}

function validateComponent(filePath, autofix = false) {
  try {
    const content = readFileSync(filePath, 'utf8');
    const relativePath = filePath.replace(srcDir, '').replace(/\\/g, '/');
    let modified = content;
    let hasChanges = false;
    let componentIssues = [];
    let hasGoodPatterns = false;
    
    validationStats.filesScanned++;
    
    for (const check of COMPLIANCE_CHECKS) {
      const matches = content.match(check.detect);
      
      if (matches) {
        if (check.isGoodPattern) {
          hasGoodPatterns = true;
          componentIssues.push({
            type: check.severity,
            rule: check.name,
            message: `✅ ${check.message} (${matches.length} instances)`,
            count: matches.length
          });
        } else {
          validationStats.issuesFound++;
          componentIssues.push({
            type: check.severity,
            rule: check.name,
            message: check.message,
            count: matches.length
          });
          
          // Apply autofix if enabled
          if (autofix && check.autofix && check.fix) {
            modified = check.fix(modified);
            hasChanges = true;
            validationStats.issuesFixed++;
          }
        }
      }
    }
    
    // Write fixes if any
    if (hasChanges) {
      writeFileSync(filePath, modified);
      console.log(`🔧 Applied autofixes to ${relativePath}`);
    }
    
    // Report component status
    if (componentIssues.length === 0 || (componentIssues.length === 1 && hasGoodPatterns)) {
      validationStats.compliantFiles++;
      console.log(`✅ ${relativePath} - Fully compliant`);
    } else {
      console.log(`\n📁 ${relativePath}:`);
      componentIssues.forEach(issue => {
        const icon = {
          error: '❌',
          warning: '⚠️ ',
          info: 'ℹ️ ',
          success: '✅'
        }[issue.type] || '•';
        
        console.log(`  ${icon} ${issue.message}${issue.count > 1 ? ` (${issue.count} instances)` : ''}`);
      });
    }
    
    return componentIssues;
    
  } catch (error) {
    validationStats.errors.push(`Error processing ${filePath}: ${error.message}`);
    return [];
  }
}

function generateReport(results, outputPath) {
  const timestamp = new Date().toISOString();
  
  const report = {
    timestamp,
    summary: {
      filesScanned: validationStats.filesScanned,
      compliantFiles: validationStats.compliantFiles,
      compliancePercentage: Math.round((validationStats.compliantFiles / validationStats.filesScanned) * 100),
      issuesFound: validationStats.issuesFound,
      issuesFixed: validationStats.issuesFixed,
      errors: validationStats.errors.length
    },
    details: results.filter(r => r.issues.length > 0).map(r => ({
      file: r.file,
      issues: r.issues
    })),
    errors: validationStats.errors
  };
  
  writeFileSync(outputPath, JSON.stringify(report, null, 2));
  console.log(`📊 Report saved to: ${outputPath}`);
  
  return report;
}

async function main() {
  const args = process.argv.slice(2);
  const autofix = args.includes('--fix');
  const generateReportFlag = args.includes('--report');
  
  if (autofix) {
    console.log('🔧 Auto-fix mode enabled');
  }
  
  console.log(`📁 Scanning for Svelte components in: ${srcDir}`);
  
  const svelteFiles = findSvelteFiles(srcDir);
  console.log(`📊 Found ${svelteFiles.length} Svelte components\n`);
  
  const results = [];
  
  for (const file of svelteFiles) {
    const issues = validateComponent(file, autofix);
    results.push({
      file: file.replace(srcDir, '').replace(/\\/g, '/'),
      issues
    });
  }
  
  // Generate summary
  console.log('\n📋 Validation Summary:');
  console.log('=====================');
  console.log(`📄 Files scanned: ${validationStats.filesScanned}`);
  console.log(`✅ Compliant files: ${validationStats.compliantFiles}`);
  console.log(`📊 Compliance rate: ${Math.round((validationStats.compliantFiles / validationStats.filesScanned) * 100)}%`);
  console.log(`🐛 Issues found: ${validationStats.issuesFound}`);
  
  if (autofix) {
    console.log(`🔧 Issues fixed: ${validationStats.issuesFixed}`);
  }
  
  if (validationStats.errors.length > 0) {
    console.log(`\n❌ Errors encountered: ${validationStats.errors.length}`);
    validationStats.errors.forEach(error => console.log(`   ${error}`));
  }
  
  // Generate report if requested
  if (generateReportFlag) {
    const reportPath = join(__dirname, `../svelte5-compliance-report-${Date.now()}.json`);
    const report = generateReport(results, reportPath);
    
    console.log('\n📈 Compliance Breakdown:');
    console.log(`   ${report.summary.compliancePercentage}% of components are fully compliant`);
    console.log(`   ${validationStats.issuesFound} issues need attention`);
  }
  
  console.log('\n🎯 Next Steps:');
  if (validationStats.issuesFound > 0 && !autofix) {
    console.log('1. Run with --fix flag to apply automatic fixes');
    console.log('2. Manually address remaining issues');
  }
  console.log('3. Run: npm run check');
  console.log('4. Run: npm run dev');
  console.log('5. Test functionality');
  
  // Exit with appropriate code
  process.exit(validationStats.errors.length > 0 ? 1 : 0);
}

main().catch(console.error);