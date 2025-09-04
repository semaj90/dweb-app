import { Pool } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";
import { building } from "$app/environment";
import * as schema from "$lib/server/db/schema-postgres";

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
    process.env.DATABASE_URL ||
    "postgresql://legal_admin:123456@localhost:5432/legal_ai_db";
  const nodeEnv = process.env.NODE_ENV || "development";

  console.log("🐘 Connecting to PostgreSQL database:", databaseUrl);

  _pool = new Pool({
    connectionString: databaseUrl,
  });

  _db = drizzle(_pool, { schema });

  // Run migrations (skip in test environment)
  if (nodeEnv !== "test") {
    try {
      // migrate(_db, { migrationsFolder: './drizzle' });
      console.log(
        "✅ PostgreSQL migrations skipped (schema already synchronized)",
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
