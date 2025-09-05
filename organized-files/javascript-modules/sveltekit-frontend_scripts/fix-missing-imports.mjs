#!/usr/bin/env zx
// Fix Missing Node.js Module Imports
// Addresses the 184 missing module import errors

import { $, glob } from 'zx';
import fs from 'fs/promises';
import path from 'path';

console.log('🔧 Starting missing imports fix...');

// Node.js modules that need proper imports
const nodeModules = {
  'path': 'import path from "path";',
  'fs': 'import fs from "fs";',
  'http': 'import http from "http";',
  'https': 'import https from "https";',
  'net': 'import net from "net";',
  'crypto': 'import crypto from "crypto";',
  'os': 'import os from "os";',
  'util': 'import util from "util";',
  'stream': 'import stream from "stream";',
  'events': 'import { EventEmitter } from "events";',
  'url': 'import { URL } from "url";',
  'querystring': 'import querystring from "querystring";'
};

// Custom modules that need path fixes
const customModules = {
  '@text/benchmark-splitter': 'import { benchmarkSplitter } from "../shared/text/benchmark-splitter.js";',
  '@shared/performance': 'import { performance } from "../shared/performance/index.js";'
};

// Find all TypeScript and JavaScript files
const targetFiles = await glob([
  'src/**/*.ts',
  'src/**/*.js', 
  'tests/**/*.ts',
  'backups/**/*.ts',
  'vite.config.ts'
], { ignore: ['node_modules/**', 'dist/**', '.svelte-kit/**'] });

console.log(`📁 Found ${targetFiles.length} files to process`);

let totalFixes = 0;
let processedFiles = 0;

for (const file of targetFiles) {
  try {
    const content = await fs.readFile(file, 'utf-8');
    let modified = content;
    let fileFixes = 0;

    // Check for missing Node.js modules
    for (const [module, importStatement] of Object.entries(nodeModules)) {
      // Check if module is used but not imported
      const moduleRegex = new RegExp(`\\b${module}\\.\\w+|\\bnew ${module}\\(|\\b${module}\\s*\\(`, 'g');
      const importRegex = new RegExp(`import.*${module}`, 'g');
      
      if (content.match(moduleRegex) && !content.match(importRegex)) {
        // Add import at the top after existing imports
        const lines = modified.split('\n');
        let insertIndex = 0;
        
        // Find last import statement
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].startsWith('import ') || lines[i].startsWith('/// <reference')) {
            insertIndex = i + 1;
          }
        }
        
        lines.splice(insertIndex, 0, importStatement);
        modified = lines.join('\n');
        fileFixes++;
        console.log(`  ➕ Added ${module} import to ${path.basename(file)}`);
      }
    }

    // Fix custom module imports
    for (const [module, replacement] of Object.entries(customModules)) {
      const regex = new RegExp(`import.*["']${module}["']`, 'g');
      if (content.match(regex)) {
        modified = modified.replace(regex, replacement);
        fileFixes++;
        console.log(`  🔄 Fixed ${module} import in ${path.basename(file)}`);
      }
    }

    // Fix relative import extensions
    modified = modified.replace(
      /import\s+([^'"]+)\s+from\s+['"]([^'"]+)['"];/g,
      (match, imports, modulePath) => {
        // Add .js extension to relative imports without extension
        if (modulePath.startsWith('./') || modulePath.startsWith('../')) {
          if (!modulePath.endsWith('.js') && !modulePath.endsWith('.ts') && !modulePath.endsWith('.svelte')) {
            return `import ${imports} from '${modulePath}.js';`;
          }
        }
        return match;
      }
    );

    // Fix Node.js globals for browser compatibility
    const browserFixes = [
      [/\bprocess\.env\./g, 'import.meta.env.'],
      [/\b__dirname\b/g, 'import.meta.url'],
      [/\b__filename\b/g, 'import.meta.url']
    ];

    for (const [pattern, replacement] of browserFixes) {
      if (content.match(pattern)) {
        modified = modified.replace(pattern, replacement);
        fileFixes++;
      }
    }

    // Write file if modified
    if (fileFixes > 0) {
      await fs.writeFile(file, modified);
      console.log(`✅ Fixed ${fileFixes} imports in ${path.basename(file)}`);
      totalFixes += fileFixes;
      processedFiles++;
    }

  } catch (error) {
    console.log(`❌ Error processing ${file}: ${error.message}`);
  }
}

console.log(`\n🎉 Import fix complete!`);
console.log(`📊 Processed: ${processedFiles} files`);
console.log(`🔧 Total fixes: ${totalFixes}`);
console.log(`✅ Ready for recompilation`);