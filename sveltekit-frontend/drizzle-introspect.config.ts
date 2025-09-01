import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  dialect: 'postgresql',
  dbCredentials: {
    url: 'postgresql://postgres:123456@localhost:5432/legal_ai_db',
  },
  out: './src/lib/server/db',
  schemaFilter: ['public'],
  tablesFilter: '*',
  verbose: true,
  strict: true,
});