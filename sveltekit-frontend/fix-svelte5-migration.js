#!/usr/bin/env node
/**
 * Svelte 5 Migration Artifact Fixer
 * Fixes transition attributes, event bindings, and prop forwarding
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(__dirname, 'src');

// Migration patterns to fix
const migrationFixes = [
  // Transition artifacts
  {
    pattern: /transitionfly\s*=\s*"([^"]+)"/g,
    replacement: 'transition:fly="$1"',
    description: 'Fix transitionfly → transition:fly'
  },
  {
    pattern: /transitionfade\s*=\s*"([^"]+)"/g,
    replacement: 'transition:fade="$1"',
    description: 'Fix transitionfade → transition:fade'
  },
  {
    pattern: /transitionfly\s*=\s*\{([^}]+)\}/g,
    replacement: 'transition:fly={$1}',
    description: 'Fix transitionfly objects → transition:fly'
  },
  {
    pattern: /transitionfade\s*=\s*\{([^}]+)\}/g,
    replacement: 'transition:fade={$1}',
    description: 'Fix transitionfade objects → transition:fade'
  },

  // Event binding artifacts
  {
    pattern: /\bonresponse\s*=\s*"([^"]+)"/g,
    replacement: 'on:response="$1"',
    description: 'Fix onresponse → on:response'
  },
  {
    pattern: /\bonupload\s*=\s*"([^"]+)"/g,
    replacement: 'on:upload="$1"',
    description: 'Fix onupload → on:upload'
  },
  {
    pattern: /\bonchange\s*=\s*"([^"]+)"/g,
    replacement: 'on:change="$1"',
    description: 'Fix onchange → on:change'
  },
  {
    pattern: /\bonsubmit\s*=\s*"([^"]+)"/g,
    replacement: 'on:submit="$1"',
    description: 'Fix onsubmit → on:submit'
  },
  {
    pattern: /\bonclose\s*=\s*"([^"]+)"/g,
    replacement: 'on:close="$1"',
    description: 'Fix onclose → on:close'
  },

  // Event binding with functions
  {
    pattern: /\bonresponse\s*=\s*\{([^}]+)\}/g,
    replacement: 'on:response={$1}',
    description: 'Fix onresponse functions → on:response'
  },
  {
    pattern: /\bonupload\s*=\s*\{([^}]+)\}/g,
    replacement: 'on:upload={$1}',
    description: 'Fix onupload functions → on:upload'
  },
  {
    pattern: /\bonchange\s*=\s*\{([^}]+)\}/g,
    replacement: 'on:change={$1}',
    description: 'Fix onchange functions → on:change'
  },
  {
    pattern: /\bonsubmit\s*=\s*\{([^}]+)\}/g,
    replacement: 'on:submit={$1}',
    description: 'Fix onsubmit functions → on:submit'
  },
  {
    pattern: /\bonclose\s*=\s*\{([^}]+)\}/g,
    replacement: 'on:close={$1}',
    description: 'Fix onclose functions → on:close'
  }
];

// Button variant mappings
const variantMappings = [
  {
    pattern: /variant\s*=\s*"primary"/g,
    replacement: 'variant="legal"',
    description: 'Map primary variant → legal'
  },
  {
    pattern: /variant\s*=\s*"danger"/g,
    replacement: 'variant="destructive"',
    description: 'Map danger variant → destructive'
  },
  {
    pattern: /variant\s*=\s*"success"/g,
    replacement: 'variant="evidence"',
    description: 'Map success variant → evidence'
  }
];

function getAllSvelteFiles(dir, files = []) {
  const items = fs.readdirSync(dir);
  
  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
      getAllSvelteFiles(fullPath, files);
    } else if (item.endsWith('.svelte') || item.endsWith('.ts')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

function fixFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');
  const originalContent = content;
  let fixCount = 0;
  const appliedFixes = [];

  // Apply migration fixes
  for (const fix of migrationFixes) {
    const matches = content.match(fix.pattern);
    if (matches) {
      content = content.replace(fix.pattern, fix.replacement);
      fixCount += matches.length;
      appliedFixes.push(`${fix.description} (${matches.length} times)`);
    }
  }

  // Apply variant mappings
  for (const mapping of variantMappings) {
    const matches = content.match(mapping.pattern);
    if (matches) {
      content = content.replace(mapping.pattern, mapping.replacement);
      fixCount += matches.length;
      appliedFixes.push(`${mapping.description} (${matches.length} times)`);
    }
  }

  // Write back if changes were made
  if (content !== originalContent) {
    fs.writeFileSync(filePath, content, 'utf8');
    return { fixed: true, count: fixCount, fixes: appliedFixes };
  }

  return { fixed: false, count: 0, fixes: [] };
}

function addClassPropToComponents() {
  // Add class prop to common UI components that are missing it
  const componentPatterns = [
    {
      file: 'src/lib/components/ui/Button.svelte',
      pattern: /export let variant:/,
      insertion: 'export let class: string = "";\n\t'
    },
    {
      file: 'src/lib/components/ui/Card.svelte', 
      pattern: /export let/,
      insertion: 'export let class: string = "";\n\t'
    }
  ];

  for (const comp of componentPatterns) {
    const fullPath = path.join(__dirname, comp.file);
    if (fs.existsSync(fullPath)) {
      let content = fs.readFileSync(fullPath, 'utf8');
      if (!content.includes('export let class:') && content.match(comp.pattern)) {
        content = content.replace(comp.pattern, comp.insertion + '$&');
        fs.writeFileSync(fullPath, content, 'utf8');
        console.log(`✅ Added class prop to ${comp.file}`);
      }
    }
  }
}

function main() {
  console.log('🔧 Starting Svelte 5 Migration Artifact Fix...\n');

  const files = getAllSvelteFiles(srcDir);
  let totalFixed = 0;
  let totalChanges = 0;
  const fileResults = [];

  for (const file of files) {
    const result = fixFile(file);
    if (result.fixed) {
      totalFixed++;
      totalChanges += result.count;
      fileResults.push({
        file: path.relative(__dirname, file),
        changes: result.count,
        fixes: result.fixes
      });
    }
  }

  // Add class props to components
  addClassPropToComponents();

  // Summary
  console.log('📊 Migration Fix Summary:');
  console.log(`   Files processed: ${files.length}`);
  console.log(`   Files fixed: ${totalFixed}`);
  console.log(`   Total changes: ${totalChanges}\n`);

  if (fileResults.length > 0) {
    console.log('📝 Detailed Results:');
    for (const result of fileResults) {
      console.log(`\n   ${result.file} (${result.changes} changes):`);
      for (const fix of result.fixes) {
        console.log(`     • ${fix}`);
      }
    }
  }

  console.log('\n✅ Svelte 5 migration artifacts fixed!');
  console.log('\n🚀 Next steps:');
  console.log('   1. Run: npm run check');
  console.log('   2. Run: npm run lint');
  console.log('   3. Test components manually');
  console.log('   4. Run production build: npm run build');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}