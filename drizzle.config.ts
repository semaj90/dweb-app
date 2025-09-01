import { defineConfig } from 'drizzle-kit';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const url = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

export default defineConfig({
  schema: './src/lib/server/db/unified-schema.ts',
  out: './database/migrations',
  dialect: 'postgresql',
  dbCredentials: { url },
  migrations: {
    table: '__drizzle_migrations__',
    schema: 'public',
    prefix: 'timestamp'
  },
  tablesFilter: ['!__drizzle_migrations__'],
  // Keep logging/dev friendliness
  verbose: process.env.NODE_ENV !== 'production',
  strict: process.env.NODE_ENV === 'production'
});