import 'dotenv/config';
import postgres from 'postgres';
import { drizzle } from 'drizzle-orm/postgres-js';

async function main() {
  const connectionString =
	process.env.DATABASE_URL || 'postgres://postgres:postgres@127.0.0.1:5432/postgres';

  // create postgres-js client and drizzle instance
  const sql = postgres(connectionString, { max: 1 });
  const db = drizzle(sql);

  try {
	// simple raw query to verify connection
	const [{ ping }] = await sql`select 1 as ping`;
	console.log('✅ Postgres connection OK — ping =', ping);

	// optional: demonstrate a drizzle usage (simple raw select)
	const [{ now }] = await sql`select now() as now`;
	console.log('⌚ db time:', now);
  } catch (err) {
	console.error('❌ Postgres connection failed:', err && err.message ? err.message : err);
	process.exitCode = 1;
  } finally {
	// close the connection pool
	await sql.end({ timeout: 1000 });
  }
}

main();
