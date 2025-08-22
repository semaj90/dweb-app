import type { Config } from 'drizzle-kit';
import * as dotenv from 'dotenv';

// Load environment variables from .env file in the parent directory
dotenv.config({ path: '../.env' });

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  throw new Error('DATABASE_URL is not set in the environment variables. Please ensure it is set in the .env file.');
}

export default {
  schema: './src/lib/db/schema.ts',
  out: './drizzle',
  driver: 'pg',
  dbCredentials: {
    connectionString: databaseUrl
  },
  verbose: true,
  strict: true
} satisfies Config;

