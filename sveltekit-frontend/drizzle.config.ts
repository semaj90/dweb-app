import { defineConfig } from 'drizzle-kit';
import { loadEnv } from 'vite';

const env = loadEnv('', process.cwd(), '');

export default defineConfig({
 schema: './src/lib/db/schema.ts',
 out: './drizzle',
 dialect: 'postgresql',
 dbCredentials: {
 url: env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3'
 },
 verbose: true,
 strict: true,
 migrations: {
 prefix: 'timestamp',
 table: '__drizzle_migrations__',
 schema: 'public'
 }
});