import fs from 'node:fs';
import path from 'node:path';
import pg from 'pg';

async function main() {
  const [,, sqlPathArg] = process.argv;
  if (!sqlPathArg) {
    console.error('Usage: node scripts/apply-sql-file.mjs <path-to-sql>');
    process.exit(1);
  }
  const sqlPath = path.resolve(sqlPathArg);
  if (!fs.existsSync(sqlPath)) {
    console.error('SQL file not found:', sqlPath);
    process.exit(1);
  }

  const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';
  const client = new pg.Client({ connectionString });

  try {
    console.log('Connecting to', connectionString.replace(/:(?:[^:@/]+)@/, ':****@'));
    await client.connect();
    const sql = fs.readFileSync(sqlPath, 'utf8');
    await client.query('BEGIN');
    await client.query(sql);
    await client.query('COMMIT');
    console.log('✅ Applied SQL:', sqlPath);
  } catch (err) {
    try { await client.query('ROLLBACK'); } catch {}
    console.error('❌ Failed to apply SQL:', err.message);
    process.exitCode = 1;
  } finally {
    await client.end();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
