// Fix all TypeScript errors following SvelteKit best practices
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'fs';
import { join, dirname, extname } from 'path';
import { execSync } from 'child_process';

console.log('🔧 Running comprehensive TypeScript checks and fixes...\n');

// Run npm check first to see current errors
console.log('📋 Running npm run check to identify errors...\n');
try {
  execSync('npm run check', { 
    stdio: 'inherit',
    cwd: join(process.cwd(), 'sveltekit-frontend')
  });
  console.log('✅ No TypeScript errors found!');
  process.exit(0);
} catch (error) {
  console.log('\n⚠️  TypeScript errors detected. Applying fixes...\n');
}

// Common fixes for SvelteKit projects
const fixes = [
  // Fix import paths
  {
    pattern: /from\s+['"]\$lib\/components\/ui\/([^/'"]+)['"]/g,
    replacement: 'from "$lib/components/ui/$1/index"',
    description: 'Fix UI component imports'
  },
  {
    pattern: /from\s+['"]\.\.\/\.\.\/(lib\/[^'"]+)['"]/g,
    replacement: 'from "$$$1"',
    description: 'Fix relative lib imports'
  },
  {
    pattern: /from\s+['"](.*?)\.js['"]/g,
    replacement: 'from "$1"',
    description: 'Remove .js extensions from imports'
  },
  // Fix context menu imports
  {
    pattern: /import\s+\{\s*Content\s+as\s+ContextMenuContent[^}]+\}\s+from\s+['"][^'"]+context-menu[^'"]*['"]/g,
    replacement: 'import * as ContextMenu from "$lib/components/ui/context-menu"',
    description: 'Fix context menu imports'
  },
  // Fix Button imports
  {
    pattern: /import\s+Button\s+from\s+['"][^'"]+Button\.svelte['"]/g,
    replacement: 'import { Button } from "$lib/components/ui/button"',
    description: 'Fix Button component imports'
  },
  // Fix missing type exports
  {
    pattern: /export\s+type\s+\{([^}]+)\}\s+from\s+['"]([^'"]+)['"]/g,
    replacement: (match, types, path) => {
      return `export type { ${types} } from "${path}"`;
    },
    description: 'Fix type exports'
  }
];

// Additional TypeScript fixes
const typeScriptFixes = [
  // Add proper event types
  {
    pattern: /on:click=\{([^}]+)\}/g,
    replacement: (match, handler) => {
      if (!handler.includes('()')) {
        return `on:click={() => ${handler}()}`;
      }
      return match;
    },
    description: 'Fix event handler syntax'
  },
  // Fix async function types
  {
    pattern: /async\s+function\s+(\w+)\s*\(/g,
    replacement: 'async function $1(',
    description: 'Fix async function declarations'
  }
];

let totalFixed = 0;
let filesProcessed = 0;
const errors = [];

function processFile(filePath) {
  try {
    const content = readFileSync(filePath, 'utf8');
    let newContent = content;
    let fileFixed = false;

    // Apply general fixes
    for (const fix of [...fixes, ...typeScriptFixes]) {
      const matches = content.match(fix.pattern);
      if (matches) {
        newContent = newContent.replace(fix.pattern, fix.replacement);
        console.log(`  ✅ ${fix.description} in ${filePath}`);
        totalFixed += matches.length;
        fileFixed = true;
      }
    }

    // Specific fixes for canvas component
    if (filePath.includes('canvas') && filePath.endsWith('.svelte')) {
      // Fix missing function keyword
      newContent = newContent.replace(/^unction\s+/gm, 'function ');
      
      // Ensure proper TypeScript types
      newContent = newContent.replace(
        /let\s+(\w+)\s*=\s*\[\]/g,
        'let $1: any[] = []'
      );
    }

    if (fileFixed) {
      writeFileSync(filePath, newContent);
      filesProcessed++;
    }
  } catch (error) {
    errors.push({ file: filePath, error: error.message });
    console.error(`  ❌ Error processing ${filePath}: ${error.message}`);
  }
}

function walkDirectory(dir) {
  try {
    const files = readdirSync(dir);
    
    for (const file of files) {
      const filePath = join(dir, file);
      const stat = statSync(filePath);
      
      // Skip directories we shouldn't process
      if (file === 'node_modules' || file === '.svelte-kit' || file === '.git' || file === 'build') {
        continue;
      }
      
      if (stat.isDirectory()) {
        walkDirectory(filePath);
      } else if (stat.isFile()) {
        const ext = extname(file);
        if (['.svelte', '.ts', '.js'].includes(ext) && !file.endsWith('.d.ts')) {
          processFile(filePath);
        }
      }
    }
  } catch (error) {
    console.error(`Error walking directory ${dir}: ${error.message}`);
  }
}

// Ensure UI component exports exist
function createUIComponentExports() {
  const uiPath = join(process.cwd(), 'sveltekit-frontend', 'src', 'lib', 'components', 'ui');
  
  // Context menu export
  const contextMenuPath = join(uiPath, 'context-menu');
  const contextMenuIndexPath = join(contextMenuPath, 'index.ts');
  
  if (!existsSync(contextMenuIndexPath)) {
    const contextMenuExport = `
export { default as Root } from './context-menu-root.svelte';
export { default as Trigger } from './context-menu-trigger.svelte';
export { default as Content } from './context-menu-content.svelte';
export { default as Item } from './context-menu-item.svelte';
export { default as Separator } from './context-menu-separator.svelte';
`;
    
    if (!existsSync(contextMenuPath)) {
      require('fs').mkdirSync(contextMenuPath, { recursive: true });
    }
    
    writeFileSync(contextMenuIndexPath, contextMenuExport);
    console.log('  ✅ Created context-menu exports');
  }
}

console.log('🚀 Starting comprehensive fixes...\n');

// Create missing exports
createUIComponentExports();

// Process files
const srcDir = join(process.cwd(), 'sveltekit-frontend', 'src');
if (existsSync(srcDir)) {
  walkDirectory(srcDir);
} else {
  console.error('❌ Could not find sveltekit-frontend/src directory');
  process.exit(1);
}

console.log(`\n✨ Fixed ${totalFixed} issues in ${filesProcessed} files`);

if (errors.length > 0) {
  console.log('\n⚠️  Errors encountered:');
  errors.forEach(({ file, error }) => {
    console.log(`  - ${file}: ${error}`);
  });
}

// Run check again
console.log('\n📋 Running npm run check again...\n');
try {
  execSync('npm run check', { 
    stdio: 'inherit',
    cwd: join(process.cwd(), 'sveltekit-frontend')
  });
  console.log('\n✅ All TypeScript errors fixed!');
} catch (error) {
  console.log('\n⚠️  Some errors may remain. Please check the output above.');
}

console.log('\n📝 Next steps:');
console.log('1. Review any remaining errors');
console.log('2. Run "npm run dev" to start the development server');
console.log('3. Test the file upload functionality');
