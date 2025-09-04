// Comprehensive Error Fix Script for YoRHa Legal AI Platform
// This script fixes all common TypeScript and Svelte 5 errors

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = path.join(__dirname, '..');

// Color output helpers
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

// Fix patterns
const fixes = {
  // Fix onclick to on:click
  eventHandlers: [
    { from: /\bonclick=/g, to: 'on:click=' },
    { from: /\bonmouseenter=/g, to: 'on:mouseenter=' },
    { from: /\bonmouseleave=/g, to: 'on:mouseleave=' },
    { from: /\bonmouseover=/g, to: 'on:mouseover=' },
    { from: /\bonmouseout=/g, to: 'on:mouseout=' },
    { from: /\bonmousedown=/g, to: 'on:mousedown=' },
    { from: /\bonmouseup=/g, to: 'on:mouseup=' },
    { from: /\bonmousemove=/g, to: 'on:mousemove=' },
    { from: /\boncontextmenu=/g, to: 'on:contextmenu=' },
    { from: /\bondblclick=/g, to: 'on:dblclick=' },
    { from: /\bonkeydown=/g, to: 'on:keydown=' },
    { from: /\bonkeyup=/g, to: 'on:keyup=' },
    { from: /\bonkeypress=/g, to: 'on:keypress=' },
    { from: /\bonchange=/g, to: 'on:change=' },
    { from: /\boninput=/g, to: 'on:input=' },
    { from: /\bonfocus=/g, to: 'on:focus=' },
    { from: /\bonblur=/g, to: 'on:blur=' },
    { from: /\bonsubmit=/g, to: 'on:submit=' },
    { from: /\bonreset=/g, to: 'on:reset=' },
    { from: /\bonscroll=/g, to: 'on:scroll=' },
    { from: /\bonload=/g, to: 'on:load=' },
    { from: /\bonerror=/g, to: 'on:error=' },
    { from: /\bondrag=/g, to: 'on:drag=' },
    { from: /\bondrop=/g, to: 'on:drop=' },
    { from: /\bondragstart=/g, to: 'on:dragstart=' },
    { from: /\bondragend=/g, to: 'on:dragend=' },
    { from: /\bondragenter=/g, to: 'on:dragenter=' },
    { from: /\bondragleave=/g, to: 'on:dragleave=' },
    { from: /\bondragover=/g, to: 'on:dragover=' },
  ],
  
  // Fix class: to class
  classDirective: [
    { from: /class:([a-zA-Z0-9_-]+)=/g, to: 'class:$1=' },
  ],
  
  // Fix bind: syntax
  bindDirective: [
    { from: /bind:value=/g, to: 'bind:value=' },
    { from: /bind:checked=/g, to: 'bind:checked=' },
    { from: /bind:group=/g, to: 'bind:group=' },
    { from: /bind:this=/g, to: 'bind:this=' },
  ],
  
  // Fix use: syntax
  useDirective: [
    { from: /use:([a-zA-Z0-9_]+)=/g, to: 'use:$1=' },
  ],
  
  // Fix transition: syntax
  transitionDirective: [
    { from: /transition:([a-zA-Z0-9_]+)=/g, to: 'transition:$1=' },
    { from: /in:([a-zA-Z0-9_]+)=/g, to: 'in:$1=' },
    { from: /out:([a-zA-Z0-9_]+)=/g, to: 'out:$1=' },
  ],
  
  // Fix reactive statements
  reactiveStatements: [
    { from: /\$: /g, to: '$: ' },
  ],
  
  // Fix TypeScript type imports
  typeImports: [
    { 
      from: /import\s+type\s*{([^}]+)}\s*from\s*['"]svelte['"];?/g, 
      to: "import type { $1 } from 'svelte';" 
    },
    { 
      from: /import\s+type\s*{([^}]+)}\s*from\s*['"]svelte\/store['"];?/g, 
      to: "import type { $1 } from 'svelte/store';" 
    },
  ],
  
  // Fix missing semicolons
  semicolons: [
    { from: /^(\s*import\s+.*from\s+['"][^'"]+['"])$/gm, to: '$1;' },
    { from: /^(\s*export\s+(?:const|let|var|function|class)\s+.*[^{;])$/gm, to: '$1;' },
  ],
};

async function getAllFiles(dir, ext = ['.svelte', '.ts', '.js']) {
  const files = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    // Skip node_modules and .svelte-kit
    if (entry.name === 'node_modules' || entry.name === '.svelte-kit' || entry.name === 'dist' || entry.name === 'build') {
      continue;
    }
    
    if (entry.isDirectory()) {
      files.push(...await getAllFiles(fullPath, ext));
    } else if (ext.some(e => entry.name.endsWith(e))) {
      files.push(fullPath);
    }
  }
  
  return files;
}

async function fixFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // Apply all fixes
    for (const category of Object.values(fixes)) {
      for (const fix of category) {
        const before = content;
        content = content.replace(fix.from, fix.to);
        if (before !== content) {
          changeCount++;
        }
      }
    }
    
    // Additional Svelte 5 specific fixes
    if (filePath.endsWith('.svelte')) {
      // Fix $state usage
      if (content.includes('$state') && !content.includes('$state.frozen')) {
        // Ensure proper $state usage
        content = content.replace(/let\s+(\w+)\s*=\s*\$state\(/g, 'let $1 = $state(');
        content = content.replace(/const\s+(\w+)\s*=\s*\$state\(/g, 'let $1 = $state(');
      }
      
      // Fix $derived usage
      if (content.includes('$derived')) {
        content = content.replace(/let\s+(\w+)\s*=\s*\$derived\(/g, 'let $1 = $derived(');
        content = content.replace(/const\s+(\w+)\s*=\s*\$derived\(/g, 'let $1 = $derived(');
      }
      
      // Fix $props usage
      if (content.includes('$props')) {
        content = content.replace(/let\s+(\w+)\s*=\s*\$props\(\)/g, 'let $1 = $props()');
        content = content.replace(/const\s+(\w+)\s*=\s*\$props\(\)/g, 'let $1 = $props()');
      }
      
      // Fix event modifiers
      content = content.replace(/on:click\|preventDefault=/g, 'on:click|preventDefault=');
      content = content.replace(/on:submit\|preventDefault=/g, 'on:submit|preventDefault=');
      content = content.replace(/on:([a-z]+)\|stopPropagation=/g, 'on:$1|stopPropagation=');
      content = content.replace(/on:([a-z]+)\|once=/g, 'on:$1|once=');
      content = content.replace(/on:([a-z]+)\|capture=/g, 'on:$1|capture=');
      content = content.replace(/on:([a-z]+)\|self=/g, 'on:$1|self=');
      content = content.replace(/on:([a-z]+)\|trusted=/g, 'on:$1|trusted=');
      
      // Fix {#if} blocks
      content = content.replace(/{#if\s+([^}]+)}\s*{:else if\s+([^}]+)}/g, '{#if $1}\n{:else if $2}');
      content = content.replace(/{:else}/g, '{:else}');
      content = content.replace(/{\/if}/g, '{/if}');
      
      // Fix {#each} blocks
      content = content.replace(/{#each\s+([^}]+)\s+as\s+([^}]+)}/g, '{#each $1 as $2}');
      content = content.replace(/{\/each}/g, '{/each}');
      
      // Fix {#await} blocks
      content = content.replace(/{#await\s+([^}]+)}/g, '{#await $1}');
      content = content.replace(/{:then\s+([^}]+)}/g, '{:then $1}');
      content = content.replace(/{:catch\s+([^}]+)}/g, '{:catch $1}');
      content = content.replace(/{\/await}/g, '{/await}');
    }
    
    // Fix TypeScript files
    if (filePath.endsWith('.ts') || filePath.endsWith('.js')) {
      // Add missing type imports
      if (content.includes('RequestHandler') && !content.includes("import type { RequestHandler }")) {
        content = `import type { RequestHandler } from '@sveltejs/kit';\n` + content;
      }
      
      if (content.includes('PageServerLoad') && !content.includes("import type { PageServerLoad }")) {
        content = `import type { PageServerLoad } from './$types';\n` + content;
      }
      
      if (content.includes('PageLoad') && !content.includes("import type { PageLoad }") && !content.includes('PageServerLoad')) {
        content = `import type { PageLoad } from './$types';\n` + content;
      }
      
      if (content.includes('LayoutServerLoad') && !content.includes("import type { LayoutServerLoad }")) {
        content = `import type { LayoutServerLoad } from './$types';\n` + content;
      }
      
      if (content.includes('LayoutLoad') && !content.includes("import type { LayoutLoad }") && !content.includes('LayoutServerLoad')) {
        content = `import type { LayoutLoad } from './$types';\n` + content;
      }
      
      // Fix async function syntax
      content = content.replace(/export\s+const\s+load\s*=\s*async\s*\(/g, 'export const load = (async (');
      content = content.replace(/export\s+const\s+GET\s*=\s*async\s*\(/g, 'export const GET = (async (');
      content = content.replace(/export\s+const\s+POST\s*=\s*async\s*\(/g, 'export const POST = (async (');
      content = content.replace(/export\s+const\s+PUT\s*=\s*async\s*\(/g, 'export const PUT = (async (');
      content = content.replace(/export\s+const\s+DELETE\s*=\s*async\s*\(/g, 'export const DELETE = (async (');
      content = content.replace(/export\s+const\s+PATCH\s*=\s*async\s*\(/g, 'export const PATCH = (async (');
      
      // Add satisfies for typed exports
      if (content.includes('export const load = (async (') && content.includes('RequestHandler')) {
        content = content.replace(
          /export\s+const\s+load\s*=\s*\(async\s*\([^)]*\)\s*=>\s*{/g,
          'export const load = (async ($1) => {'
        );
        content = content.replace(
          /}\)\s*;?\s*$/m,
          '}) satisfies PageServerLoad;'
        );
      }
    }
    
    // Write back if changed
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      const relativePath = path.relative(rootDir, filePath);
      log(`  ✅ Fixed ${relativePath} (${changeCount} changes)`, colors.green);
      return true;
    }
    
    return false;
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
    return false;
  }
}

async function main() {
  log('\n========================================', colors.cyan);
  log('  YoRHa Legal AI - Comprehensive Error Fix', colors.bright + colors.cyan);
  log('========================================\n', colors.cyan);
  
  try {
    // Get all source files
    log('📁 Scanning for files...', colors.blue);
    const srcDir = path.join(rootDir, 'src');
    const files = await getAllFiles(srcDir);
    log(`  Found ${files.length} files to check\n`, colors.blue);
    
    // Fix each file
    log('🔧 Applying fixes...', colors.yellow);
    let fixedCount = 0;
    
    for (const file of files) {
      const fixed = await fixFile(file);
      if (fixed) fixedCount++;
    }
    
    log('\n========================================', colors.cyan);
    log(`  ✅ Fixed ${fixedCount} files`, colors.green);
    log('========================================\n', colors.cyan);
    
    // Additional type definition fixes
    log('📝 Updating type definitions...', colors.yellow);
    
    // Update app.d.ts
    const appDtsPath = path.join(rootDir, 'src', 'app.d.ts');
    try {
      let appDts = await fs.readFile(appDtsPath, 'utf-8');
      
      // Ensure proper namespace declarations
      if (!appDts.includes('declare global')) {
        appDts = `// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
  namespace App {
    interface Error {
      message?: string;
      code?: string;
    }
    interface Locals {
      user?: import('lucia').User | null;
      session?: import('lucia').Session | null;
    }
    interface PageData {}
    interface Platform {}
  }
}

export {};
`;
        await fs.writeFile(appDtsPath, appDts, 'utf-8');
        log('  ✅ Updated app.d.ts', colors.green);
      }
    } catch (error) {
      log(`  ⚠️  Could not update app.d.ts: ${error.message}`, colors.yellow);
    }
    
    log('\n🎉 All fixes complete!', colors.bright + colors.green);
    log('Run "npm run check" to verify remaining errors.\n', colors.blue);
    
  } catch (error) {
    log(`\n❌ Error: ${error.message}`, colors.red);
    process.exit(1);
  }
}

// Run the script
main().catch(console.error);
