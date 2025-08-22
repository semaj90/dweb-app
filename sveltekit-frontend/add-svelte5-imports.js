#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { glob } from 'glob';

const SRC_DIR = './src';

// Check if file needs Svelte 5 imports
function needsSvelte5Imports(content) {
  const needs = {
    props: content.includes('$props()'),
    state: content.includes('$state('),
    derived: content.includes('$derived('),
    effect: content.includes('$effect(')
  };
  
  return Object.values(needs).some(Boolean) ? needs : null;
}

// Get existing import line
function getImportLine(content) {
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("import") && line.includes("svelte")) {
      return { line, index: i };
    }
    if (line.startsWith('<script')) {
      continue;
    }
    if (line.length > 0 && !line.startsWith('//') && !line.startsWith('/*')) {
      break;
    }
  }
  return null;
}

// Add missing Svelte 5 imports
function addSvelte5Imports(content) {
  const needs = needsSvelte5Imports(content);
  if (!needs) return content;
  
  const lines = content.split('\n');
  const existingImport = getImportLine(content);
  
  const requiredImports = [];
  if (needs.props) requiredImports.push('$props');
  if (needs.state) requiredImports.push('$state');
  if (needs.derived) requiredImports.push('$derived');
  if (needs.effect) requiredImports.push('$effect');
  
  if (requiredImports.length === 0) return content;
  
  const importStatement = `  import { ${requiredImports.join(', ')} } from 'svelte';`;
  
  if (existingImport) {
    // Insert after existing svelte import
    lines.splice(existingImport.index + 1, 0, importStatement);
  } else {
    // Find script tag and add after it
    const scriptIndex = lines.findIndex(line => line.trim().startsWith('<script'));
    if (scriptIndex !== -1) {
      lines.splice(scriptIndex + 1, 0, importStatement);
    }
  }
  
  return lines.join('\n');
}

async function fixFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const needs = needsSvelte5Imports(content);
    
    if (!needs) {
      return false;
    }
    
    const newContent = addSvelte5Imports(content);
    
    if (content !== newContent) {
      fs.writeFileSync(filePath, newContent);
      const imports = Object.entries(needs).filter(([k, v]) => v).map(([k]) => k).join(', ');
      console.log(`✅ ${filePath}: Added ${imports} imports`);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`❌ ${filePath}: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log('🔧 Adding missing Svelte 5 imports...\n');
  
  const files = await glob(`${SRC_DIR}/**/*.svelte`);
  console.log(`Found ${files.length} Svelte files to process\n`);
  
  let fixedCount = 0;
  
  for (const file of files) {
    const wasFixed = await fixFile(file);
    if (wasFixed) fixedCount++;
  }
  
  console.log(`\n✨ Summary: Fixed ${fixedCount} out of ${files.length} files`);
}

main().catch(console.error);