import { Client } from 'pg';
import fs from 'fs';
import path from 'path';

// Minimal native-Postgres test harness for Windows.
// Usage: set environment variables PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
// then run: node tests/setup-db.ts

async function main() {
  const cfg = {
    host: process.env.PGHOST || 'localhost',
    port: parseInt(process.env.PGPORT || '5432', 10),
    user: process.env.PGUSER || 'postgres',
    password: process.env.PGPASSWORD || 'postgres',
    database: process.env.PGDATABASE || 'legal_ai_test',
  };

  console.log('Connecting to Postgres at', `${cfg.host}:${cfg.port}`);
  const client = new Client(cfg as any);
  try {
    await client.connect();
    console.log('Connected. Creating test database (if not exists) and running migrations.');

    // Ensure database exists (connect to default postgres to create if needed)
    const dbName = cfg.database;
    const defaultClient = new Client({ ...cfg, database: 'postgres' } as any);
    await defaultClient.connect();
    const check = await defaultClient.query("SELECT 1 FROM pg_database WHERE datname = $1", [dbName]);
    if (check.rowCount === 0) {
      console.log('Database not found. Creating', dbName);
      await defaultClient.query(`CREATE DATABASE \"${dbName}\"`);
    }
    await defaultClient.end();

    // Run migration SQL file
    const migrationsDir = path.resolve(process.cwd(), 'tests', 'migrations');
    const sqlFile = path.join(migrationsDir, 'init_test_schema.sql');
    if (!fs.existsSync(sqlFile)) {
      throw new Error(`Migration file not found: ${sqlFile}`);
    }
    const sql = fs.readFileSync(sqlFile, 'utf8');
    console.log('Running migration SQL...');
    await client.query(sql);
    console.log('Migration complete.');

    // simple smoke test
    const res = await client.query("SELECT count(*) as c FROM information_schema.tables WHERE table_schema='public' AND table_name='cases';");
    console.log('Tables present check:', res.rows[0]);

    console.log('Success. Test DB ready.');
  } catch (err) {
    console.error('Error setting up test DB:', err);
    process.exitCode = 1;
  } finally {
    await client.end().catch(() => {});
  }
}

if (require.main === module) main();
