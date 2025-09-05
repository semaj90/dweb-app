#!/usr/bin/env node

import { promises as fs } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Common fixes for database schema and TypeScript errors
const fixes = [
  // Fix database schema imports and exports
  {
    pattern: /type UserSettings,/g,
    replacement: 'type UserSettingsExt as UserSettings,',
    description: 'Fix UserSettings import'
  },
  {
    pattern: /type CaseMetadata,/g,
    replacement: 'type CaseMetadataType as CaseMetadata,',
    description: 'Fix CaseMetadata import'
  },
  {
    pattern: /type EvidenceMetadata,/g,
    replacement: 'type EvidenceMetadataType as EvidenceMetadata,',
    description: 'Fix EvidenceMetadata import'
  },
  
  // Fix Drizzle ORM issues
  {
    pattern: /import { and, eq, or, sql } from "drizzle-orm";/g,
    replacement: 'import { and, eq, or, sql } from "drizzle-orm";\nimport { onConflictDoNothing } from "drizzle-orm/pg-core";',
    description: 'Add missing onConflictDoNothing import'
  },
  
  // Fix vector schema uuid issues
  {
    pattern: /pgTable\(\s*"([^"]+)",\s*{\s*id: uuid\("id"\)/g,
    replacement: 'pgTable("$1", {\n  id: uuid("id")',
    description: 'Fix pgTable schema definition'
  },
  
  // Fix embedding type issues
  {
    pattern: /embedding: Array\.isArray\(embedding\[0\]\) \? embedding\[0\] : embedding,/g,
    replacement: 'embedding: Array.isArray(embedding[0]) ? (embedding[0] as number[]) : (embedding as number[]),',
    description: 'Fix embedding type assertion'
  },
  
  // Fix error handling
  {
    pattern: /\$\{error\.message\}/g,
    replacement: '${error instanceof Error ? error.message : String(error)}',
    description: 'Fix error message handling'
  },
  
  // Fix metadata object issues
  {
    pattern: /metadata:\s*{\s*([^}]+)\s*}\s*\|\|\s*{}/g,
    replacement: 'metadata: {\n          $1\n        }',
    description: 'Fix metadata object truthy check'
  },
  
  // Fix cache.get type annotations
  {
    pattern: /cache\.get<([^>]+)>\(/g,
    replacement: 'cache.get(',
    description: 'Remove type annotations from cache.get calls'
  },
  
  // Fix import issues
  {
    pattern: /import { db, isPostgreSQL } from "\$lib\/server\/db\/index";/g,
    replacement: 'import { db } from "$lib/server/db/index";',
    description: 'Remove non-existent isPostgreSQL import'
  },
  
  // Fix seed file syntax errors
  {
    pattern: /return await seedDatabase\(\)\s*}\s*\)\s*\.onConflictDoNothing\(\);/g,
    replacement: 'return await seedDatabase();\n  } catch (error) {\n    console.error("Seed error:", error);\n    throw error;\n  }',
    description: 'Fix seed function syntax'
  }
];

// File-specific fixes
const fileSpecificFixes = {
  'src/lib/server/db/seed.ts': [
    {
      pattern: /\.onConflictDoNothing\(\);/g,
      replacement: '.onConflictDoNothing();',
      description: 'Fix onConflictDoNothing call'
    },
    {
      pattern: /await db\.insert\(legalDocuments\)\.values\(sampleLegalDocuments\);/g,
      replacement: `await db.insert(legalDocuments).values(sampleLegalDocuments.map(doc => ({
        ...doc,
        metadata: JSON.stringify(doc.metadata)
      })));`,
      description: 'Fix legal documents insert'
    },
    {
      pattern: /await db\.insert\(savedCitations\)\.values\(sampleCitations\);/g,
      replacement: `await db.insert(savedCitations).values(sampleCitations.map(citation => ({
        ...citation,
        citationData: {
          type: citation.category === 'case' ? 'case' : 'statute',
          id: citation.id,
          source: citation.citation,
          text: citation.context,
          relevanceScore: citation.relevanceScore
        }
      })));`,
      description: 'Fix saved citations insert'
    }
  ],
  
  'src/lib/server/database/vector-schema.ts': [
    {
      pattern: /pgTable\(\s*"([^"]+)",\s*{\s*id: uuid\("id"\)/g,
      replacement: 'pgTable("$1", (t) => ({\n  id: t.uuid("id")',
      description: 'Fix pgTable schema definition with column helper'
    }
  ],
  
  'src/lib/services/ai-service.ts': [
    {
      pattern: /embedding: Array\.isArray\(embedding\[0\]\) \? embedding\[0\] : embedding,/g,
      replacement: 'embedding: Array.isArray(embedding[0]) ? (embedding[0] as number[]) : (embedding as number[]),',
      description: 'Fix embedding type assertion'
    }
  ]
};

async function findFiles(dir, extension) {
  const files = [];
  
  async function traverse(currentDir) {
    const items = await fs.readdir(currentDir, { withFileTypes: true });
    
    for (const item of items) {
      const fullPath = join(currentDir, item.name);
      
      if (item.isDirectory() && !item.name.startsWith('.') && item.name !== 'node_modules') {
        await traverse(fullPath);
      } else if (item.isFile() && item.name.endsWith(extension)) {
        files.push(fullPath);
      }
    }
  }
  
  await traverse(dir);
  return files;
}

async function applyFixes() {
  const frontendDir = join(__dirname, 'sveltekit-frontend');
  const tsFiles = await findFiles(frontendDir, '.ts');
  
  let totalChanges = 0;
  const results = [];
  
  for (const file of tsFiles) {
    try {
      let content = await fs.readFile(file, 'utf8');
      const originalContent = content;
      let fileChanges = 0;
      
      // Apply general fixes
      for (const fix of fixes) {
        const matches = content.match(fix.pattern);
        if (matches) {
          content = content.replace(fix.pattern, fix.replacement);
          fileChanges += matches.length;
          console.log(`✓ ${fix.description} in ${file} (${matches.length} changes)`);
        }
      }
      
      // Apply file-specific fixes
      const relativePath = file.replace(frontendDir + '\\', '').replace(/\\/g, '/');
      const specificFixes = fileSpecificFixes[relativePath];
      
      if (specificFixes) {
        for (const fix of specificFixes) {
          const matches = content.match(fix.pattern);
          if (matches) {
            content = content.replace(fix.pattern, fix.replacement);
            fileChanges += matches.length;
            console.log(`✓ ${fix.description} in ${file} (${matches.length} changes)`);
          }
        }
      }
      
      if (content !== originalContent) {
        await fs.writeFile(file, content, 'utf8');
        totalChanges += fileChanges;
        results.push({ file, changes: fileChanges });
      }
      
    } catch (error) {
      console.error(`Error processing ${file}:`, error.message);
    }
  }
  
  console.log(`\n🎉 Database schema fixes completed!`);
  console.log(`📊 Total changes: ${totalChanges}`);
  console.log(`📁 Files modified: ${results.length}`);
  
  if (results.length > 0) {
    console.log('\n📋 Files modified:');
    results.forEach(({ file, changes }) => {
      console.log(`  • ${file}: ${changes} changes`);
    });
  }
  
  return results;
}

// Run the fixes
applyFixes().catch(console.error);
