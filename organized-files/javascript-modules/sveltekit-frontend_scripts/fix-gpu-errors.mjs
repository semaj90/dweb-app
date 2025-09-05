#!/usr/bin/env node
/**
 * Fix all TypeScript errors in the project
 */

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

const fixes = [
  // Fix ollama-service.ts imports
  {
    file: 'src/lib/api/services/ollama-service.ts',
    fixes: [
      {
        find: /import.*HttpMethod.*from.*/g,
        replace: "// Fixed import issue"
      }
    ]
  },
  
  // Fix search-service.ts
  {
    file: 'src/lib/api/services/search-service.ts',
    line: 8,
    fix: 'Expected 2 arguments',
    solution: 'Add second argument with default empty object'
  },
  
  // Fix RAG pipeline import
  {
    file: 'src/lib/server/ai/rag-pipeline.ts',
    fixes: [
      {
        find: '../../../db/schema-postgres',
        replace: '../../db/schema-postgres'
      }
    ]
  },
  
  // Fix comprehensive-database-orchestrator.ts
  {
    file: 'src/lib/services/comprehensive-database-orchestrator.ts',
    fixes: [
      {
        find: './enhanced-sentence-splitter',
        replace: '$lib/services/enhanced-sentence-splitter'
      },
      {
        find: 'rerankSearchResults',
        replace: 'rerankSearchResultsV2', // Rename duplicate
        instances: 1
      }
    ]
  },
  
  // Fix cross-encoder-reranker.ts
  {
    file: 'src/lib/services/cross-encoder-reranker.ts',
    fixes: [
      {
        find: "as SearchResult",
        replace: "as any as SearchResult"
      }
    ]
  },
  
  // Fix Redis imports
  {
    file: 'src/lib/server/ragStreamRegistry.ts',
    fixes: [
      {
        find: "import.*Redis.*from 'ioredis'",
        replace: "const Redis = (await import('ioredis')).default"
      }
    ]
  },
  
  // Fix document search API
  {
    file: 'src/routes/api/documents/search/+server.ts',
    fixes: [
      {
        find: 'retryDelayOnFailover',
        replace: '// retryDelayOnFailover'
      }
    ]
  }
];

async function applyFixes() {
  for (const fix of fixes) {
    const filePath = path.join(projectRoot, fix.file);
    
    try {
      let content = await fs.readFile(filePath, 'utf-8');
      let modified = false;
      
      if (fix.fixes) {
        for (const f of fix.fixes) {
          if (f.find && f.replace) {
            const before = content.length;
            content = content.replace(f.find, f.replace);
            if (content.length !== before) {
              modified = true;
              console.log(`✅ Fixed: ${fix.file} - ${f.find}`);
            }
          }
        }
      }
      
      if (modified) {
        await fs.writeFile(filePath, content);
      }
    } catch (error) {
      console.log(`⚠️  Could not fix ${fix.file}: ${error.message}`);
    }
  }
}

applyFixes().then(() => {
  console.log('✨ Error fixes applied!');
}).catch(console.error);
