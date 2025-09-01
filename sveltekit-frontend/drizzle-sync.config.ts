import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  dialect: 'postgresql',
  schema: './src/lib/server/db/schema.ts',
  out: './drizzle',
  dbCredentials: {
    connectionString: 'postgresql://postgres:123456@localhost:5432/legal_ai_db',
  },
  introspect: {
    casing: 'snake_case',
  },
  schemaFilter: ['public'],
});