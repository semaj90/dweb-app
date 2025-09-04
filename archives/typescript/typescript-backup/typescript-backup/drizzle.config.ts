import { defineConfig } from 'drizzle-kit';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export default defineConfig({
  schema: './src/schema.ts',
  out: './database/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
  },
  verbose: true,
  strict: true,
  migrations: {
    prefix: 'timestamp',
    table: '__drizzle_migrations__',
    schema: 'public',
  },
  tablesFilter: ['!__drizzle_migrations__'],
  
  // Enable pgvector extension support
  extensionsFilters: ['vector'],
  
  // Development vs Production settings
  ...(process.env.NODE_ENV === 'production' ? {
    verbose: false,
    strict: true,
  } : {
    verbose: true,
    strict: false,
  }),
});