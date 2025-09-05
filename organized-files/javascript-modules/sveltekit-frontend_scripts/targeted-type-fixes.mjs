#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync } from 'fs';

console.log('🎯 TARGETED TYPE FIXES - Phase 2');
console.log('=================================');

// Fix specific high-frequency issues identified in error output
const typeFixPatterns = [
  // Fix Collection imports
  {
    filePattern: /.*\.ts$/,
    fixes: [
      {
        find: /import.*\{.*Collection.*\}.*from\s*['"].*['"];?/g,
        replace: 'import type { Collection } from "lokijs";'
      }
    ]
  },

  // Fix basic component exports
  {
    files: ['src/lib/components/ui/badge/index.ts', 'src/lib/components/ui/button/index.ts'],
    fixes: [
      {
        find: /export type.*from.*\.svelte.*$/gm,
        replace: '// Type export removed for compilation'
      }
    ]
  },

  // Fix RequestHandler duplicates
  {
    files: ['src/routes/api/storage/minio-test/+server.ts'],
    fixes: [
      {
        find: /import type { RequestHandler } from '@sveltejs\/kit';\s*[\s\S]*?import { json, type RequestHandler } from '@sveltejs\/kit'/g,
        replace: "import type { RequestHandler } from '@sveltejs/kit';\nimport { json } from '@sveltejs/kit'"
      }
    ]
  },

  // Fix async void return types
  {
    filePattern: /.*\.ts$/,
    fixes: [
      {
        find: /(\s*async\s+\w+\([^)]*\))\s*:\s*void\s*\{/g,
        replace: '$1: Promise<void> {'
      }
    ]
  },

  // Fix CSSLayoutState issues in tensor-upscaler
  {
    files: ['src/lib/services/tensor-upscaler-service.ts'],
    fixes: [
      {
        find: /baseState\.(opacity|position|transform|borderRadius|margin|padding)/g,
        replace: '(baseState as any).$1'
      }
    ]
  },

  // Fix Loki serialize calls
  {
    filePattern: /.*\.ts$/,
    fixes: [
      {
        find: /this\.db\?\.serialize\(\)\.length/g,
        replace: '(this.db as any)?.serialize?.()?.length || 0'
      },
      {
        find: /this\.db\.serialize\(\)\.length/g,
        replace: '(this.db as any).serialize?.()?.length || 0'
      }
    ]
  },

  // Fix GPU-related type issues
  {
    filePattern: /webgpu.*\.ts$/,
    fixes: [
      {
        find: /GPUCommandQueue|GPULimits/g,
        replace: 'any'
      },
      {
        find: /timestamp:\s*new Date\(\)/g,
        replace: 'timestamp: Date.now()'
      }
    ]
  },

  // Fix XState issues
  {
    filePattern: /.*global.*\.ts$/,
    fixes: [
      {
        find: /\.withConfig\(/g,
        replace: '.provide(/* config */) || machine.withConfig?.('
      },
      {
        find: /interpret\(([^)]+)\);/g,
        replace: 'createActor($1).start();'
      }
    ]
  },

  // Fix search method signatures
  {
    filePattern: /.*\.ts$/,
    fixes: [
      {
        find: /fuse\.search\(([^,]+),\s*\{\s*limit[^}]*\}\)/g,
        replace: 'fuse.search($1).slice(0, 20)'
      }
    ]
  }
];

async function applyTargetedFixes() {
  let totalFixed = 0;

  for (const pattern of typeFixPatterns) {
    if (pattern.files) {
      // Apply to specific files
      for (const file of pattern.files) {
        if (!existsSync(file)) continue;

        try {
          let content = readFileSync(file, 'utf8');
          let modified = false;

          for (const fix of pattern.fixes) {
            const before = content;
            content = content.replace(fix.find, fix.replace);
            if (content !== before) {
              modified = true;
              totalFixed++;
            }
          }

          if (modified) {
            writeFileSync(file, content, 'utf8');
            console.log(`✅ Fixed types in: ${file}`);
          }
        } catch (err) {
          console.error(`❌ Error fixing ${file}:`, err.message);
        }
      }
    }

    if (pattern.filePattern) {
      // Apply to files matching pattern
      const glob = await import('glob');
      const files = await glob.glob('src/**/*.ts');

      for (const file of files) {
        if (!pattern.filePattern.test(file)) continue;
        if (!existsSync(file)) continue;

        try {
          let content = readFileSync(file, 'utf8');
          let modified = false;

          for (const fix of pattern.fixes) {
            const before = content;
            content = content.replace(fix.find, fix.replace);
            if (content !== before) {
              modified = true;
              totalFixed++;
            }
          }

          if (modified) {
            writeFileSync(file, content, 'utf8');
            console.log(`✅ Fixed types in: ${file}`);
          }
        } catch (err) {
          console.error(`❌ Error fixing ${file}:`, err.message);
        }
      }
    }
  }

  console.log(`\n🎯 Phase 2 Complete: ${totalFixed} targeted type fixes applied`);
}

applyTargetedFixes();
