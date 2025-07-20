// drizzle/pg.config.ts
// Postgres-specific Drizzle config

export default {
  schema: './web-app/sveltekit-frontend/src/lib/server/db/schema-new.ts',
  out: './drizzle',
  dialect: 'postgresql',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/prosecutor_db',
  },
};
