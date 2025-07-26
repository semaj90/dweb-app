import { defineConfig } from 'drizzle-kit';

export default defineConfig({
 schema: './src/lib/db/schema.ts',
 out: './drizzle',
 dialect: 'postgresql',
 dbCredentials: {
 url: process.env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/prosecutor_db'
 },
 verbose: true,
 strict: true,
 migrations: {
 prefix: 'timestamp',
 table: '__drizzle_migrations__',
 schema: 'public'
 }
});
