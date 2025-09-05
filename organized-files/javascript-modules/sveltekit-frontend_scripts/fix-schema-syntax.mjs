#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'fs';

const schemaFile = 'src/lib/server/db/schema-postgres.ts';
console.log('🔧 Fixing schema syntax issues...');

try {
  let content = readFileSync(schemaFile, 'utf8');

  // Fix the unclosed embeddingJobsRelations
  content = content.replace(
    /export const embeddingJobsRelations = relations\(embeddingJobs, \(\{ one \}\) => \(\{\s*\/\/ Optional relation to legal documents if needed\s*\/\/ === EXTENDED USER RELATIONS ===/,
    `export const embeddingJobsRelations = relations(embeddingJobs, ({ one }) => ({
  // Optional relation to legal documents if needed
}));

// === EXTENDED USER RELATIONS ===`
  );

  writeFileSync(schemaFile, content, 'utf8');
  console.log('✅ Schema syntax fixed!');
} catch (error) {
  console.error('❌ Error fixing schema:', error.message);
}
