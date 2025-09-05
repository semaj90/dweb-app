import { defineConfig } from 'drizzle-kit';

export default defineConfig({
  dialect: 'postgresql',
  dbCredentials: {
    host: 'localhost',
    port: 5432,
    user: 'postgres',
    password: '123456',
    database: 'legal_ai_db',
    ssl: false
  },
  out: './drizzle-introspected',
  introspect: {
    casing: 'snake_case'
  },
  verbose: true
});