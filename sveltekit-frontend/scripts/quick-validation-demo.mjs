#!/usr/bin/env node
/**
 * Quick Validation Demo - Native Node.js
 * Demonstrates Svelte 5 compliance checking without external dependencies
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Simplified glob function using native Node.js
async function findSvelteFiles(dir = 'src') {
  const files = [];
  
  async function scan(currentDir) {
    try {
      const entries = await fs.readdir(currentDir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(currentDir, entry.name);
        
        if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
          await scan(fullPath);
        } else if (entry.isFile() && entry.name.endsWith('.svelte')) {
          files.push(fullPath);
        }
      }
    } catch (error) {
      console.warn(`Warning: Could not scan ${currentDir}: ${error.message}`);
    }
  }
  
  await scan(dir);
  return files;
}

async function quickValidationDemo() {
  console.log('🚀 Quick Svelte 5 Compliance Demo');
  console.log('═'.repeat(50));
  
  const svelteFiles = await findSvelteFiles();
  console.log(`📁 Found ${svelteFiles.length} Svelte components`);
  
  const issues = {
    createEventDispatcher: 0,
    exportLet: 0,
    oldEventSyntax: 0,
    modernComponents: 0
  };
  
  const sampleFiles = [];
  
  for (const file of svelteFiles.slice(0, 10)) { // Check first 10 files for demo
    try {
      const content = await fs.readFile(file, 'utf-8');
      const relativePath = path.relative(process.cwd(), file);
      
      const analysis = {
        file: relativePath,
        hasCreateEventDispatcher: /createEventDispatcher/.test(content),
        hasExportLet: /export\s+let\s+/.test(content),
        hasOldEventSyntax: /on:\w+/.test(content),
        hasModernRunes: /\$state\(|\$props\(|\$derived/.test(content),
        linesOfCode: content.split('\n').length
      };
      
      if (analysis.hasCreateEventDispatcher) issues.createEventDispatcher++;
      if (analysis.hasExportLet) issues.exportLet++;
      if (analysis.hasOldEventSyntax) issues.oldEventSyntax++;
      if (analysis.hasModernRunes) issues.modernComponents++;
      
      sampleFiles.push(analysis);
    } catch (error) {
      console.warn(`Could not analyze ${file}: ${error.message}`);
    }
  }
  
  console.log('\n📊 Migration Status (Sample):');
  console.log(`🔧 Components with createEventDispatcher: ${issues.createEventDispatcher}`);
  console.log(`📝 Components with export let: ${issues.exportLet}`);  
  console.log(`⚡ Components with old event syntax: ${issues.oldEventSyntax}`);
  console.log(`✨ Components with modern runes: ${issues.modernComponents}`);
  
  console.log('\n📋 Sample Component Analysis:');
  sampleFiles.forEach(analysis => {
    const flags = [];
    if (analysis.hasCreateEventDispatcher) flags.push('📤 dispatcher');
    if (analysis.hasExportLet) flags.push('📝 export let');
    if (analysis.hasOldEventSyntax) flags.push('⚡ old events');
    if (analysis.hasModernRunes) flags.push('✨ runes');
    
    console.log(`  📄 ${analysis.file} (${analysis.linesOfCode} lines) ${flags.join(', ') || '✅ clean'}`);
  });
  
  // Migration recommendations
  console.log('\n🎯 Migration Recommendations:');
  if (issues.createEventDispatcher > 0) {
    console.log(`  🔧 ${issues.createEventDispatcher} components need createEventDispatcher migration`);
  }
  if (issues.exportLet > 0) {
    console.log(`  📝 ${issues.exportLet} components need export let → $props() migration`);
  }
  if (issues.oldEventSyntax > 0) {
    console.log(`  ⚡ ${issues.oldEventSyntax} components need modern event syntax`);
  }
  
  console.log(`\n✨ ${issues.modernComponents} components already use modern Svelte 5 patterns!`);
  
  console.log('\n🚀 Ready for automated migration:');
  console.log('  node scripts/migrate-components-phase9.mjs');
  console.log('  node scripts/validate-svelte5-compliance.mjs --fix');
}

quickValidationDemo().catch(console.error);