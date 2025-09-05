#!/usr/bin/env node
/**
 * Drizzle Schema Diagnostic Script
 * Scans repository for common anti-patterns:
 *  1. vector imported from drizzle-orm/pg-core (should NOT be there)
 *  2. (Allowed) 'pgvector/drizzle-orm' – canonical import
 *  3. Manual constructed pg = {...} namespace mocks
 *  4. Placeholder schema usage outside allowed path
 */
import { globby } from 'globby';
import fs from 'fs/promises';
import path from 'path';

const ROOT = process.cwd();
const problems = [];

function push(file, message, fix) { problems.push({ file, message, fix }); }

async function scanFile(file) {
  const content = await fs.readFile(file, 'utf8');
  if (/vector\s*,|,\s*vector/.test(content) && content.includes("drizzle-orm/pg-core")) {
    push(file, 'vector imported from drizzle-orm/pg-core', "Remove from pg-core import; use: import { vector } from 'pgvector/drizzle-orm';");
  }
  if (/const\s+pg\s*=\s*\{[^}]*vector:/s.test(content)) {
    push(file, 'Manual pg namespace with vector field detected', 'Remove custom pg object; use official imports.');
  }
  if (file.endsWith('db/schema-postgres.ts') && !content.includes('IMPORTANT: Placeholder')) {
    push(file, 'Placeholder schema missing IMPORTANT header', 'Add warning comment or remove file.');
  }
}

(async () => {
  const files = await globby(['**/*.ts','**/*.js','**/*.mjs','!node_modules/**','!dist/**']);
  await Promise.all(files.map(scanFile));
  if (problems.length === 0) {
    console.log('✅ No Drizzle schema anti-patterns found.');
    process.exit(0);
  }
  console.log('⚠️ Drizzle Schema Diagnostics Report');
  for (const p of problems) {
    console.log(`\nFile: ${p.file}\nIssue: ${p.message}\nFix: ${p.fix}`);
  }
  console.log(`\nTotal issues: ${problems.length}`);
  process.exit(1);
})();
