import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  schema: './sveltekit-frontend/src/lib/server/db/schema.ts',
  out: './database/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    host: 'localhost',
    port: 5432,
    user: 'postgres',
    password: 'postgres',
    database: 'prosecutor_db',
  },
  verbose: true,
  strict: true,
});