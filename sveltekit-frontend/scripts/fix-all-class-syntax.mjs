#!/usr/bin/env zx
// Comprehensive Class Syntax Fix for Svelte 5 Components
// Fixes all "class: class" destructuring issues across the entire codebase

import { $, glob } from 'zx';
import fs from 'fs/promises';
import path from 'path';

console.log('🔧 Starting comprehensive class syntax fix...');

// Find all Svelte files with class syntax issues
const svelteFiles = await glob('src/**/*.svelte');
console.log(`📁 Found ${svelteFiles.length} Svelte files to process`);

let totalFixes = 0;
let processedFiles = 0;

for (const file of svelteFiles) {
  try {
    const content = await fs.readFile(file, 'utf-8');
    let modified = content;
    let fileFixes = 0;

    // Fix patterns
    const fixes = [
      // Fix "class: class = ''" destructuring
      [/class:\s*class\s*=/g, 'class: className ='],
      
      // Fix "let class = ..." declarations
      [/let\s+class\s*:/g, 'let className:'],
      [/let\s+class\s*=/g, 'let className ='],
      
      // Fix class usage in templates
      [/\{class\}/g, '{className}'],
      [/\{class\s*\|\|\s*['""]['""]?\}/g, '{className || ""}'],
      [/\{class\s*\?\}/g, '{className}'],
      
      // Fix $derived(class)
      [/\$derived\(class\)/g, '$derived(className)'],
      
      // Fix function parameters
      [/, class\)/g, ', className)'],
      [/\(class\)/g, '(className)'],
      
      // Fix template class references
      [/class="\{[^}]*class[^}]*\}"/g, (match) => match.replace(/\bclass\b/g, 'className')],
      [/class='\{[^}]*class[^}]*\}'/g, (match) => match.replace(/\bclass\b/g, 'className')],
      
      // Fix on:open syntax errors
      [/\.on:open=/g, '.onopen='],
      [/statusSocket\.on:open=/g, 'statusSocket.onopen='],
      [/wsConnection\.on:open=/g, 'wsConnection.onopen='],
      
      // Fix extra semicolons in interfaces
      [/timestamp\?\s*:\s*string\s*\}\s*;+/g, 'timestamp?: string }'],
      
      // Fix analytics function syntax
      [/analyticsLog\s*=\s*>\s*void\s*=\s*\(\)\s*=>\s*\{\}/g, 'analyticsLog = () => {}'],
      
      // Fix incomplete prop destructuring
      [/\.\.\.\s*restProps\s*\}\s*=\s*\$props\(\)\s*;/g, '...restProps } = $props();']
    ];

    // Apply all fixes
    for (const [pattern, replacement] of fixes) {
      const before = modified;
      modified = modified.replace(pattern, replacement);
      if (before !== modified) {
        fileFixes++;
      }
    }

    // Additional context-specific fixes
    if (file.includes('enhanced-bits') || file.includes('Card') || file.includes('alert')) {
      // Fix missing Props interface
      if (!modified.includes('interface Props')) {
        modified = modified.replace(
          /<script lang="ts">/,
          `<script lang="ts">
  interface Props {
    class?: string;
    children?: import('svelte').Snippet;
  }`
        );
        fileFixes++;
      }
    }

    // Write file if modified
    if (fileFixes > 0) {
      await fs.writeFile(file, modified);
      console.log(`✅ Fixed ${fileFixes} issues in ${path.basename(file)}`);
      totalFixes += fileFixes;
      processedFiles++;
    }

  } catch (error) {
    console.log(`❌ Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Class syntax fix complete!`);
console.log(`📊 Processed: ${processedFiles} files`);
console.log(`🔧 Total fixes: ${totalFixes}`);
console.log(`✅ Ready for compilation`);