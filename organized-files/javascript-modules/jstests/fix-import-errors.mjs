// Comprehensive fix for import path errors in the SvelteKit app
import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

const ROOT_DIR = 'C:\\Users\\james\\Desktop\\web-app\\sveltekit-frontend\\src';

class ImportFixer {
  constructor() {
    this.errors = [];
    this.fixes = [];
  }

  // Get all .svelte and .ts files recursively
  getAllFiles(dir, files = []) {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Skip node_modules and .svelte-kit
        if (!item.startsWith('.') && item !== 'node_modules') {
          this.getAllFiles(fullPath, files);
        }
      } else if (['.svelte', '.ts', '.js'].includes(extname(item))) {
        files.push(fullPath);
      }
    }
    
    return files;
  }

  // Fix malformed import paths
  fixImportPaths(content, filePath) {
    let modified = false;
    const lines = content.split('\n');
    const fixedLines = [];

    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];
      const originalLine = line;

      // Fix 1: Remove extra /index from import paths
      line = line.replace(
        /import\s+.*?\s+from\s+["']([^"']+)\.svelte\/index["']/g,
        (match, path) => {
          this.fixes.push(`${filePath}: Fixed .svelte/index import: ${path}`);
          return match.replace('.svelte/index', '.svelte');
        }
      );

      // Fix 2: Remove double /index from UI component imports
      line = line.replace(
        /import\s+.*?\s+from\s+["']([^"']+)\/index\.js\/index["']/g,
        (match, path) => {
          this.fixes.push(`${filePath}: Fixed double index import: ${path}`);
          return match.replace('/index.js/index', '');
        }
      );

      // Fix 3: Correct UI component imports to use proper structure
      line = line.replace(
        /import\s+\{([^}]+)\}\s+from\s+["']\$lib\/components\/ui\/index\.js\/index["']/g,
        (match, imports) => {
          this.fixes.push(`${filePath}: Fixed UI index import`);
          return `import { ${imports} } from "$lib/components/ui"`;
        }
      );

      // Fix 4: Clean up malformed component imports
      line = line.replace(
        /import\s+(\w+)\s+from\s+["']([^"']+)\.svelte\/index["']/g,
        (match, component, path) => {
          this.fixes.push(`${filePath}: Fixed component import: ${component}`);
          return `import ${component} from "${path}.svelte"`;
        }
      );

      // Fix 5: Fix ExpandGrid, GoldenLayout, SmartTextarea direct imports
      const componentMap = {
        'ExpandGrid': '$lib/components/ui/ExpandGrid.svelte',
        'GoldenLayout': '$lib/components/ui/GoldenLayout.svelte', 
        'SmartTextarea': '$lib/components/ui/SmartTextarea.svelte'
      };

      for (const [component, correctPath] of Object.entries(componentMap)) {
        const malformedPattern = new RegExp(
          `import\\s+${component}\\s+from\\s+["']\\$lib\\/components\\/ui\\/${component}\\.svelte\\/index["']`,
          'g'
        );
        if (malformedPattern.test(line)) {
          line = line.replace(malformedPattern, `import ${component} from "${correctPath}"`);
          this.fixes.push(`${filePath}: Fixed ${component} import`);
        }
      }

      // Fix 6: Correct Button imports from UI components
      line = line.replace(
        /import\s+\{\s*Button\s*\}\s+from\s+["']\$lib\/components\/ui\/index\.js\/index["']/g,
        `import { Button } from "$lib/components/ui/button"`
      );

      // Fix 7: Clean up any remaining /index duplications
      line = line.replace(
        /from\s+["']([^"']+)\/index\/index["']/g,
        (match, path) => {
          this.fixes.push(`${filePath}: Fixed duplicate index: ${path}`);
          return `from "${path}"`;
        }
      );

      if (line !== originalLine) {
        modified = true;
      }

      fixedLines.push(line);
    }

    return { content: fixedLines.join('\n'), modified };
  }

  // Fix CSS @apply issues
  fixCSSIssues(content, filePath) {
    let modified = false;
    
    // Fix UnoCSS @apply usage - ensure it's properly configured
    if (content.includes('@apply') && !content.includes('/* unocss-ignore */')) {
      // Add unocss comment if @apply is used
      content = content.replace(
        /@apply\s+([^;]+);/g,
        (match, classes) => {
          this.fixes.push(`${filePath}: Added UnoCSS ignore comment for @apply`);
          modified = true;
          return `/* @unocss-ignore */\n  @apply ${classes};`;
        }
      );
    }

    return { content, modified };
  }

  // Process a single file
  processFile(filePath) {
    try {
      const content = readFileSync(filePath, 'utf8');
      let result = { content, modified: false };

      // Fix import paths
      const importFix = this.fixImportPaths(result.content, filePath);
      result = importFix;

      // Fix CSS issues
      const cssFix = this.fixCSSIssues(result.content, filePath);
      if (cssFix.modified) {
        result.content = cssFix.content;
        result.modified = true;
      }

      // Write back if modified
      if (result.modified) {
        writeFileSync(filePath, result.content, 'utf8');
        console.log(`✓ Fixed: ${filePath}`);
      }

    } catch (error) {
      this.errors.push(`Error processing ${filePath}: ${error.message}`);
    }
  }

  // Main execution
  async run() {
    console.log('🔧 Starting comprehensive import fix...\n');

    const files = this.getAllFiles(ROOT_DIR);
    console.log(`📁 Found ${files.length} files to process\n`);

    for (const file of files) {
      this.processFile(file);
    }

    console.log('\n📊 Fix Summary:');
    console.log(`✅ Fixes applied: ${this.fixes.length}`);
    console.log(`❌ Errors encountered: ${this.errors.length}\n`);

    if (this.fixes.length > 0) {
      console.log('🔧 Applied fixes:');
      this.fixes.forEach(fix => console.log(`  • ${fix}`));
    }

    if (this.errors.length > 0) {
      console.log('\n⚠️  Errors:');
      this.errors.forEach(error => console.log(`  • ${error}`));
    }

    console.log('\n✨ Import fix complete!');
  }
}

// Create additional fix for UnoCSS configuration
function createUnoCSSSafeConfig() {
  const unoConfig = `
import { defineConfig, presetUno, presetAttributify, presetTypography } from 'unocss';

export default defineConfig({
  presets: [
    presetUno(),
    presetAttributify(),
    presetTypography()
  ],
  rules: [
    // Custom rules if needed
  ],
  shortcuts: {
    // Common shortcuts
    'btn': 'px-4 py-2 rounded bg-blue-500 text-white hover:bg-blue-600',
    'card': 'bg-white rounded-lg shadow-md p-4',
  },
  safelist: [
    // Add commonly used classes that might be dynamic
    'bg-blue-100',
    'px-2',
    'py-1', 
    'rounded',
    'text-sm',
    'font-mono'
  ]
});
`;

  try {
    writeFileSync('C:\\Users\\james\\Desktop\\web-app\\sveltekit-frontend\\uno.config.ts', unoConfig);
    console.log('✓ Updated UnoCSS configuration');
  } catch (error) {
    console.error('Error updating UnoCSS config:', error.message);
  }
}

// Execute the fix
const fixer = new ImportFixer();
fixer.run().then(() => {
  console.log('\n🎨 Updating UnoCSS configuration...');
  createUnoCSSSafeConfig();
  
  console.log('\n🚀 Ready to run npm commands!');
  console.log('Next steps:');
  console.log('1. cd C:\\Users\\james\\Desktop\\web-app');
  console.log('2. npm run check');
  console.log('3. npm run dev');
});
