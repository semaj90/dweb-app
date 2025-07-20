import { defineConfig } from 'drizzle-kit';
import { loadEnv } from 'vite';

const env = loadEnv('', process.cwd(), '');

export default defineConfig({
 schema: './src/lib/db/schema.ts',
 out: './drizzle',
 dialect: 'postgresql',
 dbCredentials: {
 url: env.DATABASE_URL || 'postgresql://detective:secure_password@localhost:5433/detective_evidence_db'
 },
 verbose: true,
 strict: true,
 migrations: {
 prefix: 'timestamp',
 table: '__drizzle_migrations__',
 schema: 'public'
 }
});