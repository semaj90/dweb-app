#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const glob = require('glob');

console.log('🔧 Fixing common TypeScript syntax errors...\n');

// Common patterns to fix
const fixes = [
  {
    description: 'Fix type alias name cannot be "type" errors',
    pattern: /^(\s*)type\s+type\s+([^=]+)=/gm,
    replacement: '$1type $2='
  },
  {
    description: 'Fix unexpected keyword import errors in .ts files',
    pattern: /^(\s*)import\s+type\s+(\{[^}]+\})\s+from\s+([^;]+);?\s*$/gm,
    replacement: '$1import type $2 from $3;'
  },
  {
    description: 'Fix missing semicolons in import statements',
    pattern: /^(\s*)import\s+([^;]+)$/gm,
    replacement: (match, whitespace, importStatement) => {
      if (!importStatement.endsWith(';')) {
        return `${whitespace}import ${importStatement};`;
      }
      return match;
    }
  },
  {
    description: 'Fix malformed destructuring imports',
    pattern: /^(\s*)import\s+\{\s*([^}]*)\s*\}\s*from\s*['"]([^'"]+)['"]\s*$/gm,
    replacement: '$1import { $2 } from "$3";'
  }
];

function fixFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    let fixedContent = content;
    let hasChanges = false;

    fixes.forEach(fix => {
      if (typeof fix.replacement === 'function') {
        const newContent = fixedContent.replace(fix.pattern, fix.replacement);
        if (newContent !== fixedContent) {
          hasChanges = true;
          fixedContent = newContent;
        }
      } else {
        const newContent = fixedContent.replace(fix.pattern, fix.replacement);
        if (newContent !== fixedContent) {
          hasChanges = true;
          fixedContent = newContent;
        }
      }
    });

    if (hasChanges) {
      fs.writeFileSync(filePath, fixedContent);
      console.log(`✅ Fixed: ${filePath}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error(`❌ Error fixing ${filePath}:`, error.message);
    return false;
  }
}

// Find all TypeScript and JavaScript files
const patterns = [
  'src/**/*.ts',
  'src/**/*.js',
  'src/**/*.svelte'
];

let totalFixed = 0;

patterns.forEach(pattern => {
  const files = glob.sync(pattern, { ignore: ['node_modules/**', '.svelte-kit/**', 'build/**'] });
  
  files.forEach(file => {
    if (fixFile(file)) {
      totalFixed++;
    }
  });
});

console.log(`\n🎉 Fixed ${totalFixed} files with syntax errors`);

// Additional specific fixes for common SvelteKit issues
console.log('\n🔧 Applying SvelteKit-specific fixes...\n');

// Fix .svelte-kit/tsconfig.json if it doesn't exist
const tsconfigPath = '.svelte-kit/tsconfig.json';
if (!fs.existsSync(tsconfigPath)) {
  const baseTsconfig = {
    "extends": "../tsconfig.json",
    "compilerOptions": {
      "allowJs": true,
      "checkJs": false,
      "esModuleInterop": true,
      "forceConsistentCasingInFileNames": true,
      "resolveJsonModule": true,
      "skipLibCheck": true,
      "sourceMap": true,
      "strict": false,
      "types": ["vite/client"],
      "moduleResolution": "bundler",
      "target": "esnext",
      "module": "esnext",
      "lib": ["esnext", "dom", "dom.iterable"]
    },
    "include": [
      "../src/**/*",
      "../tests/**/*"
    ]
  };
  
  // Ensure .svelte-kit directory exists
  if (!fs.existsSync('.svelte-kit')) {
    fs.mkdirSync('.svelte-kit', { recursive: true });
  }
  
  fs.writeFileSync(tsconfigPath, JSON.stringify(baseTsconfig, null, 2));
  console.log('✅ Created .svelte-kit/tsconfig.json');
}

console.log('\n✨ Syntax error fixes completed!');
console.log('Run `npm run dev` to test the fixes.');