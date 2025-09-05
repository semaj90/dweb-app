#!/usr/bin/env zx
// Fix Critical Svelte Component Errors
// Target specific files with compilation-blocking errors

import { $, glob } from 'zx';
import fs from 'fs/promises';
import path from 'path';

console.log('🚨 Fixing critical Svelte compilation errors...');

// Specific files with critical errors from the log
const criticalFiles = [
  'src/lib/components/ui/GoldenLayout.svelte',
  'src/lib/components/LegalCaseManager.svelte', 
  'src/lib/components/ai/EnhancedFileUpload.svelte',
  'src/lib/components/EnhancedRAGInterface.svelte',
  'src/lib/components/ai/EnhancedMCPIntegration.svelte',
  'src/lib/components/detective/ContextMenu.svelte',
  'src/lib/components/ui/enhanced-bits/Card.svelte',
  'src/lib/components/ui/enhanced-bits/Input.svelte',
  'src/lib/components/ai/LLMProviderSelector.svelte',
  'src/lib/components/legal/IntegrityVerification.svelte',
  'src/lib/components/cases/CaseFilters.svelte',
  'src/lib/components/ui/select/SelectContent.svelte'
];

let totalFixes = 0;

for (const file of criticalFiles) {
  try {
    console.log(`🔄 Processing ${path.basename(file)}...`);
    
    const content = await fs.readFile(file, 'utf-8');
    let modified = content;
    let fileFixes = 0;

    // Fix 1: Duplicate prop declarations (remove export let, keep $props)
    if (file.includes('GoldenLayout.svelte')) {
      // Remove duplicate export declarations
      modified = modified.replace(/export let ratio[^;]*;.*$/gm, '');
      modified = modified.replace(/export let mainFlex[^;]*;.*$/gm, '');
      modified = modified.replace(/export let sidebarFlex[^;]*;.*$/gm, '');
      modified = modified.replace(/export let sidebarPosition[^;]*;.*$/gm, '');
      modified = modified.replace(/export let collapsible[^;]*;.*$/gm, '');
      modified = modified.replace(/export let collapsed[^;]*;.*$/gm, '');
      modified = modified.replace(/export let minSidebarWidth[^;]*;.*$/gm, '');
      modified = modified.replace(/export let maxSidebarWidth[^;]*;.*$/gm, '');
      modified = modified.replace(/export let gap[^;]*;.*$/gm, '');
      fileFixes++;
    }

    // Fix 2: WebSocket event syntax (on:close -> onclose)
    modified = modified.replace(/\.on:close\s*=/g, '.onclose =');
    modified = modified.replace(/\.on:open\s*=/g, '.onopen =');
    modified = modified.replace(/\.on:message\s*=/g, '.onmessage =');
    modified = modified.replace(/\.on:error\s*=/g, '.onerror =');
    if (content.match(/\.on:(close|open|message|error)/)) {
      fileFixes++;
    }

    // Fix 3: Unexpected semicolon in object types
    modified = modified.replace(/verificationResults:\s*\{\s*;/g, 'verificationResults: {');
    modified = modified.replace(/\}\s*;\s*$/gm, '}');
    if (content.includes('verificationResults: {;')) {
      fileFixes++;
    }

    // Fix 4: Malformed object syntax  
    modified = modified.replace(/\(\s*\{\s*([^}]+),\s*class\s*\}\s*\)\)/g, '({ $1, className })');
    modified = modified.replace(/\(\s*\{\s*([^}]+)\s*,\s*class\s*\)\)/g, '({ $1, className })');
    if (content.match(/\}\s*\)\)/)) {
      fileFixes++;
    }

    // Fix 5: Incomplete files - add missing closing tags
    if (file.includes('CaseFilters.svelte') || file.includes('LegalCaseManager.svelte') || 
        file.includes('EnhancedRAGInterface.svelte') || file.includes('LLMProviderSelector.svelte')) {
      
      // If file doesn't end with </script> or </div> or </style>, add proper closing
      if (!modified.match(/(<\/script>|<\/div>|<\/style>)\s*$/)) {
        if (modified.includes('<script')) {
          modified += '\n</script>';
          fileFixes++;
        }
        if (modified.includes('<div') && !modified.includes('</div>')) {
          modified += '\n</div>';
          fileFixes++;
        }
        if (modified.includes('<style') && !modified.includes('</style>')) {
          modified += '\n</style>';
          fileFixes++;
        }
      }
    }

    // Fix 6: Clean up extra semicolons and malformed interfaces
    modified = modified.replace(/\;\s*\r?;\s*$/gm, ';');
    modified = modified.replace(/\}\s*\r?;\s*let\s*\{/g, '}\n\nlet {');

    // Write file if modified
    if (fileFixes > 0) {
      await fs.writeFile(file, modified);
      console.log(`✅ Fixed ${fileFixes} critical issues in ${path.basename(file)}`);
      totalFixes += fileFixes;
    } else {
      console.log(`✓ No critical issues in ${path.basename(file)}`);
    }

  } catch (error) {
    console.log(`❌ Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Critical Svelte fixes complete!`);
console.log(`🔧 Total fixes: ${totalFixes}`);
console.log(`✅ Ready for recompilation`);