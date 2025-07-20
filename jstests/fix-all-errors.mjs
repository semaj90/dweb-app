#!/usr/bin/env node
// Comprehensive error logging and fixing script

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import chalk from 'chalk';

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const logDir = join(process.cwd(), 'logs');
const errorLogPath = join(logDir, `svelte-check-errors-${timestamp}.log`);
const fixLogPath = join(logDir, `fixes-applied-${timestamp}.log`);

// Ensure log directory exists
if (!existsSync(logDir)) {
  mkdirSync(logDir, { recursive: true });
}

console.log(chalk.bold.cyan('🔍 Running comprehensive error check and fix...\n'));

// Step 1: Run svelte-check and capture output
console.log(chalk.yellow('📋 Running svelte-check and logging errors...'));
let checkOutput = '';
try {
  checkOutput = execSync('cd sveltekit-frontend && npm run check', { 
    encoding: 'utf8',
    stdio: 'pipe'
  });
} catch (error) {
  checkOutput = error.stdout || error.message;
}

// Save the error log
writeFileSync(errorLogPath, checkOutput);
console.log(chalk.green(`✅ Error log saved to: ${errorLogPath}\n`));

// Step 2: Parse and categorize errors
console.log(chalk.yellow('🔧 Analyzing and fixing errors...\n'));

const fixes = [];

// Fix patterns
const fixPatterns = [
  // Fix incorrect module imports
  {
    name: 'Remove /index from imports',
    pattern: /from\s+["']([^"']+)\.svelte\/index["']/g,
    replacement: 'from "$1.svelte"',
    files: '**/*.svelte,**/*.ts'
  },
  {
    name: 'Fix UI component imports',
    pattern: /from\s+["']\$lib\/components\/ui\/index\.js\/index["']/g,
    replacement: 'from "$lib/components/ui/button"',
    files: '**/*.svelte,**/*.ts'
  },
  {
    name: 'Fix Modal imports',
    pattern: /from\s+["']\$lib\/components\/ui\/Modal\.svelte\/index["']/g,
    replacement: 'from "$lib/components/ui/Modal.svelte"',
    files: '**/*.svelte,**/*.ts'
  },
  {
    name: 'Fix reserved word "case"',
    pattern: /\{#each\s+([^}]+)\s+as\s+case\s*\}/g,
    replacement: '{#each $1 as caseItem}',
    files: '**/*.svelte'
  },
  {
    name: 'Fix case property access',
    pattern: /\{case\.([\w]+)\}/g,
    replacement: '{caseItem.$1}',
    files: '**/*.svelte'
  },
  {
    name: 'Fix export let case',
    pattern: /export\s+let\s+case\s*:/g,
    replacement: 'export let caseItem:',
    files: '**/*.svelte'
  },
  // Fix CSS @apply
  {
    name: 'Convert @apply to CSS',
    pattern: /@apply\s+([^;]+);/g,
    replacement: (match, classes) => {
      const classMap = {
        'bg-blue-100': 'background-color: #dbeafe',
        'px-2': 'padding-left: 0.5rem; padding-right: 0.5rem',
        'py-1': 'padding-top: 0.25rem; padding-bottom: 0.25rem',
        'rounded': 'border-radius: 0.25rem',
        'text-sm': 'font-size: 0.875rem',
        'font-mono': 'font-family: monospace',
        'block': 'display: block',
        'hidden': 'display: none',
        'p-4': 'padding: 1rem',
        'border': 'border: 1px solid',
        'border-gray-200': 'border-color: #e5e7eb',
        'rounded-lg': 'border-radius: 0.5rem',
        'bg-white': 'background-color: white'
      };
      
      const cssRules = classes.split(/\s+/)
        .map(cls => classMap[cls])
        .filter(Boolean)
        .join('; ');
      
      return cssRules ? cssRules + ';' : '/* ' + classes + ' */;';
    },
    files: '**/*.svelte'
  }
];

// Apply fixes
let totalFixes = 0;
const fixLog = [];

function applyFixes(filePath, patterns) {
  try {
    let content = readFileSync(filePath, 'utf8');
    let originalContent = content;
    let fileFixed = false;
    
    for (const fix of patterns) {
      const matches = content.match(fix.pattern);
      if (matches) {
        content = content.replace(fix.pattern, fix.replacement);
        fixLog.push(`✅ ${fix.name} in ${filePath} (${matches.length} occurrences)`);
        totalFixes += matches.length;
        fileFixed = true;
      }
    }
    
    if (fileFixed) {
      writeFileSync(filePath, content);
      return true;
    }
    return false;
  } catch (error) {
    fixLog.push(`❌ Error processing ${filePath}: ${error.message}`);
    return false;
  }
}

// Step 3: Apply systematic fixes
console.log(chalk.yellow('📝 Applying fixes to files...\n'));

// Fix all files
import glob from 'glob';
const files = glob.sync('sveltekit-frontend/src/**/*.{svelte,ts,js}', {
  ignore: ['**/node_modules/**', '**/.svelte-kit/**', '**/build/**']
});

let filesFixed = 0;
for (const file of files) {
  if (applyFixes(file, fixPatterns)) {
    filesFixed++;
  }
}

// Step 4: Create missing type definitions
console.log(chalk.yellow('\n📦 Creating missing type definitions...\n'));

const typeDefinitions = [
  {
    path: 'sveltekit-frontend/src/lib/types/case.ts',
    content: `// Case-related type definitions
export interface Citation {
  id: string;
  text: string;
  source: string;
  caseId?: string;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface CaseItem {
  id: string;
  title: string;
  description?: string;
  status: string;
  createdAt: Date;
  updatedAt: Date;
  [key: string]: any;
}
`
  },
  {
    path: 'sveltekit-frontend/src/lib/types/ui.ts',
    content: `// UI component type definitions
export type ButtonVariant = 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  icon?: string;
  iconPosition?: 'left' | 'right';
}
`
  }
];

for (const typeDef of typeDefinitions) {
  const dir = join(process.cwd(), typeDef.path.substring(0, typeDef.path.lastIndexOf('/')));
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(join(process.cwd(), typeDef.path), typeDef.content);
  fixLog.push(`✅ Created type definition: ${typeDef.path}`);
}

// Step 5: Fix specific component issues
console.log(chalk.yellow('\n🔨 Fixing specific component issues...\n'));

// Fix Button component to accept more variants
const buttonComponentPath = 'sveltekit-frontend/src/lib/components/ui/button/Button.svelte';
if (existsSync(buttonComponentPath)) {
  let buttonContent = readFileSync(buttonComponentPath, 'utf8');
  
  // Update variant type
  buttonContent = buttonContent.replace(
    /type:\s*'default'\s*\|\s*'primary'\s*\|\s*'secondary'\s*\|\s*'outline'\s*\|\s*'ghost'\s*\|\s*'danger'/,
    `type: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'warning' | 'info'`
  );
  
  // Update size type
  buttonContent = buttonContent.replace(
    /size\?\:\s*'sm'\s*\|\s*'md'\s*\|\s*'lg'/,
    `size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'`
  );
  
  // Add fullWidth prop
  if (!buttonContent.includes('fullWidth')) {
    buttonContent = buttonContent.replace(
      /export let loading.*?false;/s,
      `export let loading: $$Props['loading'] = false;
  export let fullWidth: boolean = false;`
    );
    
    // Add fullWidth class
    buttonContent = buttonContent.replace(
      /\$\{classes\}/,
      `\${classes} \${fullWidth ? 'w-full' : ''}`
    );
  }
  
  writeFileSync(buttonComponentPath, buttonContent);
  fixLog.push('✅ Updated Button component with additional variants and props');
}

// Step 6: Save fix log
const fixLogContent = fixLog.join('\n');
writeFileSync(fixLogPath, fixLogContent);

// Step 7: Run check again
console.log(chalk.yellow('\n🔍 Running final check...\n'));
let finalCheckOutput = '';
try {
  finalCheckOutput = execSync('cd sveltekit-frontend && npm run check', { 
    encoding: 'utf8',
    stdio: 'pipe'
  });
  console.log(chalk.green('✅ All checks passed!'));
} catch (error) {
  finalCheckOutput = error.stdout || error.message;
  const remainingErrors = (finalCheckOutput.match(/Error:/g) || []).length;
  const remainingWarnings = (finalCheckOutput.match(/Warn:/g) || []).length;
  
  console.log(chalk.yellow(`⚠️  Remaining issues: ${remainingErrors} errors, ${remainingWarnings} warnings`));
}

// Summary
console.log(chalk.bold.cyan('\n📊 Summary:\n'));
console.log(`Total fixes applied: ${totalFixes}`);
console.log(`Files modified: ${filesFixed}`);
console.log(`Error log: ${errorLogPath}`);
console.log(`Fix log: ${fixLogPath}`);

// Show remaining issues
if (finalCheckOutput.includes('Error:')) {
  console.log(chalk.yellow('\n⚠️  Some errors still remain. Common solutions:'));
  console.log('1. Install missing dependencies: npm install fuse.js');
  console.log('2. Update component imports to use correct paths');
  console.log('3. Check for circular dependencies');
  console.log('4. Ensure all UI components are properly exported');
}

console.log(chalk.green('\n✨ Error check and fix complete!\n'));
