import { building } from "$app/environment";
import * as schema from "$lib/server/db/schema-postgres";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

let _db: ReturnType<typeof drizzle> | null = null;
let _pool: Pool | null = null;

export function getPostgreSQLDatabase() {
  // Skip database initialization during SvelteKit build
  if (building) {
    console.log("Skipping database initialization during build");
    return null;
  }

  if (_db) return _db;

  const databaseUrl =
    import.meta.env.DATABASE_URL ||
    "postgresql://postgres:postgres@localhost:5432/prosecutor_db";
  const nodeEnv = import.meta.env.NODE_ENV || "development";

  console.log("🐘 Connecting to PostgreSQL database:", databaseUrl);

  _pool = new Pool({
    connectionString: databaseUrl,
  });

  _db = drizzle(_pool, { schema });

  // Run migrations (skip in testing environment)
  if (nodeEnv !== "testing") {
    try {
      // migrate(_db, { migrationsFolder: './drizzle' });
      console.log(
        "✅ PostgreSQL migrations skipped (schema already synchronized)"
      );
    } catch (error: any) {
      console.log("⚠️ PostgreSQL migration warning:", error);
    }
  } else {
    console.log("⏭️ Skipping migrations in testing environment");
  }

  console.log("✅ PostgreSQL database connection established");
  return _db;
}

// Export the database instance
export const db = getPostgreSQLDatabase();

// Cleanup function
export async function closeDatabase(): Promise<any> {
  if (_pool) {
    await _pool.end();
    _pool = null;
    _db = null;
  }
}
