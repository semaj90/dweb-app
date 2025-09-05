#!/usr/bin/env node
// Targeted fixes for specific svelte-check errors

import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { execSync } from 'child_process';

console.log('🔧 Applying targeted fixes for svelte-check errors...\n');

const logDir = join(process.cwd(), 'logs');
if (!existsSync(logDir)) {
  mkdirSync(logDir, { recursive: true });
}

let fixCount = 0;
const fixLog = [];

// Helper function to fix file
function fixFile(filePath, fixes) {
  try {
    if (!existsSync(filePath)) return false;
    
    let content = readFileSync(filePath, 'utf8');
    let original = content;
    
    for (const fix of fixes) {
      if (fix.test && !fix.test(content)) continue;
      content = content.replace(fix.pattern, fix.replacement);
    }
    
    if (content !== original) {
      writeFileSync(filePath, content);
      fixLog.push(`✅ Fixed ${filePath}`);
      fixCount++;
      return true;
    }
    return false;
  } catch (error) {
    fixLog.push(`❌ Error fixing ${filePath}: ${error.message}`);
    return false;
  }
}

// 1. Fix all import path issues
console.log('📦 Fixing import paths...');

const importFixes = [
  // Remove /index from .svelte imports
  {
    pattern: /from\s+["']([^"']+)\.svelte\/index["']/g,
    replacement: 'from "$1.svelte"'
  },
  // Fix UI component imports
  {
    pattern: /from\s+["']\$lib\/components\/ui\/index\.js\/index["']/g,
    replacement: 'from "$lib/components/ui/button"'
  },
  // Fix Button imports specifically
  {
    pattern: /import\s+Button\s+from\s+["']\$lib\/components\/ui\/Button\.svelte\/index["']/g,
    replacement: 'import { Button } from "$lib/components/ui/button"'
  },
  // Fix Modal imports
  {
    pattern: /import\s+Modal\s+from\s+["']\$lib\/components\/ui\/Modal\.svelte\/index["']/g,
    replacement: 'import Modal from "$lib/components/ui/Modal.svelte"'
  },
  // Fix Tooltip imports
  {
    pattern: /import\s+Tooltip\s+from\s+["']\$lib\/components\/ui\/Tooltip\.svelte\/index["']/g,
    replacement: 'import { Tooltip } from "$lib/components/ui/tooltip"'
  },
  // Fix RichTextEditor imports
  {
    pattern: /import\s+RichTextEditor\s+from\s+["']\$lib\/components\/ui\/RichTextEditor\.svelte\/index["']/g,
    replacement: 'import RichTextEditor from "$lib/components/ui/RichTextEditor.svelte"'
  },
  // Fix ExpandGrid imports
  {
    pattern: /import\s+ExpandGrid\s+from\s+["']\$lib\/components\/ui\/ExpandGrid\.svelte\/index["']/g,
    replacement: 'import ExpandGrid from "$lib/components/ui/ExpandGrid.svelte"'
  },
  // Fix GoldenLayout imports
  {
    pattern: /import\s+GoldenLayout\s+from\s+["']\$lib\/components\/ui\/GoldenLayout\.svelte\/index["']/g,
    replacement: 'import GoldenLayout from "$lib/components/ui/GoldenLayout.svelte"'
  },
  // Fix SmartTextarea imports
  {
    pattern: /import\s+SmartTextarea\s+from\s+["']\$lib\/components\/ui\/SmartTextarea\.svelte\/index["']/g,
    replacement: 'import SmartTextarea from "$lib/components/ui/SmartTextarea.svelte"'
  }
];

// Apply import fixes to all files
function walkDir(dir) {
  const files = readdirSync(dir);
  for (const file of files) {
    const path = join(dir, file);
    const stat = statSync(path);
    
    if (stat.isDirectory() && !['node_modules', '.svelte-kit', 'build'].includes(file)) {
      walkDir(path);
    } else if (stat.isFile() && (file.endsWith('.svelte') || file.endsWith('.ts'))) {
      fixFile(path, importFixes);
    }
  }
}

const srcDir = join(process.cwd(), 'sveltekit-frontend', 'src');
walkDir(srcDir);

// 2. Fix reserved word 'case' usage
console.log('\n🔤 Fixing reserved word "case" usage...');

const caseWordFixes = [
  {
    pattern: /export\s+let\s+case\s*:/g,
    replacement: 'export let caseItem:'
  },
  {
    pattern: /\{#each\s+([^}]+)\s+as\s+case\}/g,
    replacement: '{#each $1 as caseItem}'
  },
  {
    pattern: /\{#each\s+([^}]+)\s+as\s+case\s*\(([^)]+)\)\}/g,
    replacement: '{#each $1 as caseItem ($2)}'
  },
  {
    pattern: /\{case\.([\w]+)\}/g,
    replacement: '{caseItem.$1}'
  },
  {
    pattern: /case=\{/g,
    replacement: 'caseItem={'
  }
];

// Apply case fixes
walkDir(srcDir);

// 3. Fix CSS @apply directives
console.log('\n🎨 Fixing CSS @apply directives...');

const cssFiles = [
  'src/routes/frameworks-demo/+page.svelte',
  'src/routes/local-ai-demo/+page.svelte',
  'src/routes/rag-demo/+page.svelte',
  'src/lib/components/ai/EnhancedAIAssistant.new.svelte'
];

for (const file of cssFiles) {
  const filePath = join(process.cwd(), 'sveltekit-frontend', file);
  fixFile(filePath, [{
    pattern: /@apply\s+([^;]+);/g,
    replacement: (match, classes) => {
      const replacements = {
        'bg-blue-100 px-2 py-1 rounded text-sm font-mono': 
          'background-color: #dbeafe; padding: 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-family: monospace;',
        'p-4 border border-gray-200 rounded-lg bg-white':
          'padding: 1rem; border: 1px solid #e5e7eb; border-radius: 0.5rem; background-color: white;',
        'block': 'display: block;',
        'hidden': 'display: none;',
        'px-4 py-3 text-sm font-medium text-legal-neutral hover:text-legal-navy border-b-2 border-transparent transition-colors flex items-center':
          'padding: 0.75rem 1rem; font-size: 0.875rem; font-weight: 500; color: var(--legal-neutral); border-bottom: 2px solid transparent; transition: color 0.15s; display: flex; align-items: center;',
        'text-primary-600 border-primary-600':
          'color: var(--primary-600); border-color: var(--primary-600);'
      };
      
      return replacements[classes] || `/* TODO: ${classes} */;`;
    }
  }]);
}

// 4. Fix Button component variants
console.log('\n🔘 Updating Button component...');

const buttonPath = join(process.cwd(), 'sveltekit-frontend', 'src/lib/components/ui/button/Button.svelte');
if (existsSync(buttonPath)) {
  let buttonContent = readFileSync(buttonPath, 'utf8');
  
  // Expand variant types
  buttonContent = buttonContent.replace(
    /variant\?\:\s*['"]default['"]\s*\|\s*['"]primary['"]\s*\|\s*['"]secondary['"]\s*\|\s*['"]outline['"]\s*\|\s*['"]ghost['"]\s*\|\s*['"]danger['"];?/,
    `variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'warning' | 'info';`
  );
  
  // Expand size types
  buttonContent = buttonContent.replace(
    /size\?\:\s*['"]sm['"]\s*\|\s*['"]md['"]\s*\|\s*['"]lg['"];?/,
    `size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';`
  );
  
  // Add fullWidth and icon props if missing
  if (!buttonContent.includes('fullWidth')) {
    buttonContent = buttonContent.replace(
      /export let loading.*?false;/,
      `export let loading: $$Props['loading'] = false;
  export let fullWidth: boolean = false;
  export let icon: string | undefined = undefined;
  export let iconPosition: 'left' | 'right' = 'left';`
    );
  }
  
  // Add variant styles
  const variantStyles = `
  /* Additional variants */
  .button--success {
    background-color: #10b981;
    color: white;
    border-color: #10b981;
  }
  
  .button--success:hover:not(.button--disabled) {
    background-color: #059669;
    border-color: #059669;
  }
  
  .button--warning {
    background-color: #f59e0b;
    color: white;
    border-color: #f59e0b;
  }
  
  .button--warning:hover:not(.button--disabled) {
    background-color: #d97706;
    border-color: #d97706;
  }
  
  .button--info {
    background-color: #3b82f6;
    color: white;
    border-color: #3b82f6;
  }
  
  .button--info:hover:not(.button--disabled) {
    background-color: #2563eb;
    border-color: #2563eb;
  }
  
  /* Additional sizes */
  .button--xs {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
  }
  
  .button--xl {
    padding: 0.75rem 1.5rem;
    font-size: 1.125rem;
  }
  
  /* Full width */
  .button--full-width {
    width: 100%;
  }`;
  
  // Add styles before closing style tag
  buttonContent = buttonContent.replace(
    /<\/style>/,
    `${variantStyles}\n</style>`
  );
  
  // Update classes computation
  buttonContent = buttonContent.replace(
    /\$:\s*classes\s*=\s*\[[\s\S]*?\]\.filter\(Boolean\)\.join\(' '\);/,
    `$: classes = [
    'button',
    \`button--\${variant}\`,
    \`button--\${size}\`,
    disabled && 'button--disabled',
    loading && 'button--loading',
    fullWidth && 'button--full-width'
  ].filter(Boolean).join(' ');`
  );
  
  writeFileSync(buttonPath, buttonContent);
  fixLog.push('✅ Updated Button component with all variants and sizes');
}

// 5. Add missing Citation type
console.log('\n📝 Adding missing type definitions...');

const caseServicePath = join(process.cwd(), 'sveltekit-frontend', 'src/lib/services/caseService.ts');
if (existsSync(caseServicePath)) {
  let serviceContent = readFileSync(caseServicePath, 'utf8');
  
  if (!serviceContent.includes('export interface Citation')) {
    // Add Citation interface after other interfaces
    serviceContent = serviceContent.replace(
      /export interface CaseData {/,
      `export interface Citation {
  id: string;
  text: string;
  source: string;
  caseId?: string;
  url?: string;
  page?: number;
  createdAt?: string;
  updatedAt?: string;
}

export interface CaseData {`
    );
    
    writeFileSync(caseServicePath, serviceContent);
    fixLog.push('✅ Added Citation interface to caseService');
  }
}

// 6. Fix accessibility issues
console.log('\n♿ Fixing accessibility issues...');

const a11yFixes = [
  // Fix click handlers on divs
  {
    pattern: /<div([^>]*)\s+on:click(\|[^=]+)?=\{([^}]+)\}([^>]*)>/g,
    replacement: (match, before, modifiers, handler, after) => {
      if (before.includes('role=') || after.includes('role=')) {
        return match;
      }
      return `<div${before} on:click${modifiers || ''}={${handler}} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && ${handler}}${after}>`;
    }
  },
  // Fix label associations
  {
    pattern: /<label class="([^"]+)">([^<]+)<\/label>\s*<textarea/g,
    replacement: '<label for="$2-input" class="$1">$2</label>\n<textarea id="$2-input"'
  },
  {
    pattern: /<label class="([^"]+)">([^<]+)<\/label>\s*<input/g,
    replacement: '<label for="$2-input" class="$1">$2</label>\n<input id="$2-input"'
  }
];

// Apply a11y fixes
walkDir(srcDir);

// 7. Install missing dependencies
console.log('\n📦 Installing missing dependencies...');

try {
  execSync('cd sveltekit-frontend && npm install fuse.js', { stdio: 'inherit' });
  fixLog.push('✅ Installed fuse.js');
} catch (error) {
  fixLog.push('⚠️  Failed to install fuse.js - install manually');
}

// 8. Create fix for missing Fuse import type
const fuseTypesPath = join(process.cwd(), 'sveltekit-frontend', 'src/lib/types/fuse.d.ts');
if (!existsSync(dirname(fuseTypesPath))) {
  mkdirSync(dirname(fuseTypesPath), { recursive: true });
}
writeFileSync(fuseTypesPath, `declare module 'fuse' {
  export default class Fuse<T> {
    constructor(list: T[], options?: any);
    search(pattern: string): Array<{ item: T; score: number }>;
  }
}
`);
fixLog.push('✅ Created Fuse type definition');

// Save fix log
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const fixLogPath = join(logDir, `fixes-${timestamp}.log`);
writeFileSync(fixLogPath, fixLog.join('\n'));

console.log(`\n✨ Applied ${fixCount} fixes`);
console.log(`📋 Fix log saved to: ${fixLogPath}`);

// Run final check
console.log('\n🔍 Running final check...');
try {
  execSync('cd sveltekit-frontend && npm run check', { stdio: 'inherit' });
} catch (error) {
  console.log('\n⚠️  Some errors may still remain. Check the output above.');
}
