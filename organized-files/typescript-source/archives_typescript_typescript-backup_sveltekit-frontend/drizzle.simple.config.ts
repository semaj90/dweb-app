import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  schema: './src/lib/server/db/enhanced-unified-schema.ts',
  out: './drizzle',
  dialect: 'postgresql',
  dbCredentials: {
    host: 'localhost',
    port: 5432,
    user: 'postgres',
    password: '123456',
    database: 'legal_ai_db',
    ssl: false
  },
  verbose: true,
  strict: true,
});