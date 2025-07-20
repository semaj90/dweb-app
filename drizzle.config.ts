import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";

// Load environment variables
config();

// Enforce Postgres only
export default defineConfig({
  schema: "./sveltekit-frontend/src/lib/server/db/unified-schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url:
      process.env.DATABASE_URL ||
      "postgresql://postgres:postgres@localhost:5432/prosecutor_db",
  },
  strict: true,
  verbose: true,
});
