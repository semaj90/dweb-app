#!/usr/bin/env node
/**
 * Critical Svelte Syntax Error Fixer
 * Fixes the most critical syntax errors preventing compilation
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Critical syntax fixes for compilation errors
const criticalFixes = [
  // Fix missing script end tags
  {
    pattern: /(\s+});\s*$/m,
    replacement: '$1;\n</script>',
    description: 'Add missing script end tags',
    files: [
      'src/lib/components/EnhancedRAGInterface.svelte',
      'src/lib/components/ai/AIChat.svelte',
      'src/lib/components/cases/CaseFilters.svelte',
      'src/lib/components/LegalCaseManager.svelte',
      'src/lib/components/layout/MasonryGrid.svelte'
    ]
  },
  
  // Fix duplicate prop declarations in AIDropdown
  {
    file: 'src/lib/components/ui/AIDropdown.svelte',
    pattern: /let \{ disabled = \$bindable\(\) \} = \$props\(\); \/\/ false;\s*let \{ onReportGenerate = \$bindable\(\) \} = \$props\(\); \/\/ \(reportType: string\) => void = \(\) => \{\};\s*let \{ onSummarize = \$bindable\(\) \} = \$props\(\); \/\/ \(\) => void = \(\) => \{\};\s*let \{ onAnalyze = \$bindable\(\) \} = \$props\(\); \/\/ \(\) => void = \(\) => \{\};\s*let \{ hasContent = \$bindable\(\) \} = \$props\(\); \/\/ false;\s*let \{ isGenerating = \$bindable\(\) \} = \$props\(\); \/\/ false;/g,
    replacement: `let {
      disabled = false,
      onReportGenerate = (reportType: string) => {},
      onSummarize = () => {},
      onAnalyze = () => {},
      hasContent = false,
      isGenerating = false
    } = $props();`,
    description: 'Fix duplicate prop declarations in AIDropdown'
  },

  // Fix Grid component ternary operator
  {
    file: 'src/lib/components/ui/grid/Grid.svelte',
    pattern: /let gridClass = \$derived\(responsive\);\s*\? `grid-cols-1 sm:grid-cols-2 md:grid-cols-\$\{Math\.min\(columns, 6\)\} lg:grid-cols-\$\{Math\.min\(columns, 8\)\} xl:grid-cols-\$\{columns\}`\s*: `grid-cols-\$\{columns\}`;/g,
    replacement: `let gridClass = $derived(responsive 
      ? \`grid-cols-1 sm:grid-cols-2 md:grid-cols-\${Math.min(columns, 6)} lg:grid-cols-\${Math.min(columns, 8)} xl:grid-cols-\${columns}\`
      : \`grid-cols-\${columns}\`);`,
    description: 'Fix Grid component ternary operator syntax'
  },

  // Fix DialogTitle semicolon issue
  {
    file: 'src/lib/components/ui/dialog/DialogTitle.svelte',
    pattern: /title: any;\r?;/g,
    replacement: 'title: any;',
    description: 'Fix DialogTitle semicolon issue'
  },

  // Fix context menu content script structure
  {
    file: 'src/lib/components/ui/context-menu/context-menu-content.svelte',
    pattern: /(\s+)(\s+)(const \{ isOpen, position, close \} = getContext)/g,
    replacement: '$1$3',
    description: 'Fix context menu content script structure'
  },

  // Fix enhanced bits components semicolon issues
  {
    pattern: /interface Props \{\s*[^}]*;\r?;\s*let \{/g,
    replacement: function(match) {
      return match.replace(/;\r?;/, ';\n    }\n    let {');
    },
    description: 'Fix enhanced bits components interface structure',
    files: [
      'src/lib/components/ui/enhanced-bits/AIChatMessage.svelte',
      'src/lib/components/ui/enhanced-bits/ChatMessage.svelte'
    ]
  },

  // Fix context menu trigger interface placement
  {
    file: 'src/lib/components/ui/context-menu/context-menu-trigger.svelte',
    pattern: /import type \{ Writable \} from 'svelte\/store'; children,\s*interface BuilderAction/g,
    replacement: `import type { Writable } from 'svelte/store';
    
    interface BuilderAction`,
    description: 'Fix context menu trigger interface placement'
  },

  // Fix legal components semicolon issues  
  {
    pattern: /evidenceFile: File \| null;\r?;/g,
    replacement: 'evidenceFile: File | null;',
    description: 'Fix legal components semicolon issues',
    files: [
      'src/lib/components/legal/IntegrityVerification.svelte'
    ]
  }
];

function fixFile(filePath, fixes) {
  if (!fs.existsSync(filePath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return { fixed: false, count: 0, fixes: [] };
  }

  let content = fs.readFileSync(filePath, 'utf8');
  const originalContent = content;
  let fixCount = 0;
  const appliedFixes = [];

  for (const fix of fixes) {
    if (fix.file && !filePath.endsWith(fix.file)) continue;
    if (fix.files && !fix.files.some(f => filePath.endsWith(f))) continue;

    const beforeCount = (content.match(fix.pattern) || []).length;
    if (beforeCount > 0) {
      if (typeof fix.replacement === 'function') {
        content = content.replace(fix.pattern, fix.replacement);
      } else {
        content = content.replace(fix.pattern, fix.replacement);
      }
      
      const afterCount = (content.match(fix.pattern) || []).length;
      const actualFixed = beforeCount - afterCount;
      
      if (actualFixed > 0) {
        fixCount += actualFixed;
        appliedFixes.push(`${fix.description} (${actualFixed} times)`);
      }
    }
  }

  // Write back if changes were made
  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    return { fixed: true, count: fixCount, fixes: appliedFixes };
  }

  return { fixed: false, count: 0, fixes: [] };
}

function getAllSvelteFiles(dir, files = []) {
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
      getAllSvelteFiles(fullPath, files);
    } else if (item.endsWith('.svelte')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

function main() {
  console.log('🔧 Starting Critical Svelte Syntax Fix...\n');

  const srcDir = path.join(__dirname, 'src');
  const files = getAllSvelteFiles(srcDir);
  let totalFixed = 0;
  let totalChanges = 0;
  const fileResults = [];

  for (const file of files) {
    const result = fixFile(file, criticalFixes);
    if (result.fixed) {
      totalFixed++;
      totalChanges += result.count;
      fileResults.push({
        file: path.relative(__dirname, file),
        changes: result.count,
        fixes: result.fixes
      });
    }
  }

  // Summary
  console.log('📊 Critical Syntax Fix Summary:');
  console.log(`   Files processed: ${files.length}`);
  console.log(`   Files fixed: ${totalFixed}`);
  console.log(`   Total changes: ${totalChanges}\n`);

  if (fileResults.length > 0) {
    console.log('📝 Detailed Results:');
    for (const result of fileResults) {
      console.log(`\n   ${result.file} (${result.changes} changes):`);
      for (const fix of result.fixes) {
        console.log(`     • ${fix}`);
      }
    }
  }

  console.log('\n✅ Critical syntax errors fixed!');
  console.log('\n🚀 Next steps:');
  console.log('   1. Run: npm run check:svelte');
  console.log('   2. Review and test components');
  console.log('   3. Run complete check: npm run check');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}