#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { glob } from 'glob';

// Enhanced critical syntax fixes for Svelte components
const fixPatterns = [
  // Fix malformed $derived with semicolon issue: $derived(expr;) -> $derived(expr)
  {
    pattern: /\$derived\(([^)]+);?\)/g,
    replacement: (match, expr) => `$derived(${expr.trim().replace(/;$/, '')})`
  },

  // Fix duplicate variable declarations with $bindable()
  {
    pattern: /let\s*\{\s*(\w+)\s*=\s*\$bindable\(\)\s*\}\s*=\s*\$props\(\);[^}]*let\s*\{\s*\1\s*=\s*\$bindable\(\)\s*\}\s*=\s*\$props\(\);/g,
    replacement: (match, varName) => {
      // Keep only the first declaration
      return match.replace(new RegExp(`let\\s*\\{\\s*${varName}\\s*=\\s*\\$bindable\\(\\)\\s*\\}\\s*=\\s*\\$props\\(\\);`, 'g'), (m, offset, string) => {
        // Only keep the first occurrence
        return offset === string.indexOf(m) ? m : '';
      });
    }
  },

  // Fix type annotation syntax issues: prop: Type ; -> prop: Type
  {
    pattern: /(\w+)\s*:\s*([A-Za-z_][A-Za-z0-9_<>|]+)\s*;/g,
    replacement: '$1: $2'
  },

  // Fix empty $derived expressions: $derived({}) -> $derived({})
  {
    pattern: /\$derived\(\s*\{\s*\}\s*\)/g,
    replacement: '$derived({})'
  },

  // Fix orphaned semicolons at start of lines
  {
    pattern: /^[\s]*;[\s]*$/gm,
    replacement: ''
  },

  // Fix import statements with missing 'from': import 'module' -> import 'module';
  {
    pattern: /^import\s+['"]([^'"]+)['"](?!\s*;)/gm,
    replacement: "import '$1';"
  },

  // Fix malformed prop destructuring: }: Props = $props(); -> } = $props();
  {
    pattern: /\}:\s*\w+\s*=\s*\$props\(\)/g,
    replacement: '} = $props()'
  },

  // Fix incomplete expressions: .filter(); -> .filter(item => item)
  {
    pattern: /\.filter\(\s*\)/g,
    replacement: '.filter(item => item)'
  },

  // Fix incomplete array literals: $derived([); -> $derived([])
  {
    pattern: /\$derived\(\s*\[\s*\)\s*;/g,
    replacement: '$derived([])'
  },

  // Fix missing closing parentheses in function calls: getSaveStatus(); -> getSaveStatus()
  {
    pattern: /(\w+)\(\s*\)\s*;\s*\)/g,
    replacement: '$1()'
  }
];

// Create missing component files
const missingComponents = [
  'src/lib/components/dashboard/CachePerformanceMonitor.svelte'
];

async function createMissingFiles() {
  console.log('🛠️ Creating missing component files...');

  for (const componentPath of missingComponents) {
    const fullPath = join(process.cwd(), componentPath);
    const componentName = componentPath.split('/').pop().replace('.svelte', '');

    const basicComponent = `<script>
  // Basic ${componentName} component
  let { data = {} } = $props();

  // Placeholder implementation
  let status = $derived('operational');
</script>

<div class="cache-performance-monitor">
  <h3>${componentName}</h3>
  <p>Status: {status}</p>
  <p>Component temporarily stubbed - TODO: Implement full functionality</p>
</div>

<style>
  .cache-performance-monitor {
    padding: 1rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #f9f9f9;
  }
</style>`;

    try {
      // Create directory if it doesn't exist
      const dir = fullPath.split('/').slice(0, -1).join('/');
      const { mkdirSync } = await import('fs');
      mkdirSync(dir, { recursive: true });

      writeFileSync(fullPath, basicComponent, 'utf8');
      console.log(`✅ Created: ${componentPath}`);
    } catch (err) {
      console.error(`❌ Failed to create ${componentPath}:`, err.message);
    }
  }
}

async function fixDuplicateExports() {
  const schemaPath = 'src/lib/server/db/schema-postgres.ts';
  console.log(`🔧 Fixing duplicate exports in ${schemaPath}...`);

  try {
    let content = readFileSync(schemaPath, 'utf8');

    // Remove duplicate usersRelations export (keep the first one)
    const lines = content.split('\n');
    let foundFirstUsersRelations = false;

    const fixedLines = lines.filter(line => {
      if (line.includes('export const usersRelations')) {
        if (foundFirstUsersRelations) {
          console.log('  Removing duplicate usersRelations export');
          return false; // Remove this line
        } else {
          foundFirstUsersRelations = true;
          return true; // Keep first occurrence
        }
      }
      return true;
    });

    writeFileSync(schemaPath, fixedLines.join('\n'), 'utf8');
    console.log(`✅ Fixed duplicate exports in ${schemaPath}`);

  } catch (err) {
    console.error(`❌ Error fixing ${schemaPath}:`, err.message);
  }
}

async function applyCriticalFixes() {
  console.log('🚀 Starting critical syntax fixes...');

  // Find all Svelte files
  const svelteFiles = await glob('src/**/*.svelte', { ignore: 'node_modules/**' });
  console.log(`Found ${svelteFiles.length} Svelte files to process`);

  let totalFixes = 0;

  for (const file of svelteFiles) {
    try {
      let content = readFileSync(file, 'utf8');
      let modified = false;
      let fileFixes = 0;

      // Apply each fix pattern
      for (const { pattern, replacement } of fixPatterns) {
        const before = content;
        if (typeof replacement === 'function') {
          content = content.replace(pattern, replacement);
        } else {
          content = content.replace(pattern, replacement);
        }
        if (content !== before) {
          modified = true;
          fileFixes++;
        }
      }

      // Special handling for complex duplicate prop declarations
      const propDeclPattern = /let\s*\{\s*(\w+)[\s\S]*?\}\s*=\s*\$props\(\);/g;
      const matches = [...content.matchAll(propDeclPattern)];
      const seenProps = new Set();

      for (const match of matches) {
        const propName = match[1];
        if (seenProps.has(propName)) {
          // Remove duplicate
          content = content.replace(match[0], '');
          modified = true;
          fileFixes++;
        } else {
          seenProps.add(propName);
        }
      }

      if (modified) {
        writeFileSync(file, content, 'utf8');
        console.log(`✅ Fixed ${fileFixes} issues in ${file}`);
        totalFixes += fileFixes;
      }
    } catch (err) {
      console.error(`❌ Error processing ${file}:`, err.message);
    }
  }

  console.log(`🎯 Total fixes applied: ${totalFixes}`);
}

async function main() {
  console.log('🔥 Critical Svelte 5 Syntax Repair Tool');
  console.log('=======================================');

  try {
    await createMissingFiles();
    await fixDuplicateExports();
    await applyCriticalFixes();

    console.log('\n✅ Critical fixes completed!');
    console.log('📝 Next steps:');
    console.log('   1. Run: npm run dev');
    console.log('   2. Check for remaining errors');
    console.log('   3. Continue with manual fixes if needed');

  } catch (error) {
    console.error('❌ Critical error:', error);
    process.exit(1);
  }
}

main();
