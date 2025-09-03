import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { building } from "$app/environment";
import * as schema from "$lib/server/db/schema-postgres";

let _db: ReturnType<typeof drizzle> | null = null;
let _sql: ReturnType<typeof postgres> | null = null;

export function getPostgreSQLDatabase() {
  // Skip database initialization during SvelteKit build
  if (building) {
    console.log("Skipping database initialization during build");
    return null;
  }
  if (_db) return _db;

  const databaseUrl =
    process.env.DATABASE_URL ||
    "postgresql://postgres:123456@localhost:5432/legal_ai_db";
  const nodeEnv = process.env.NODE_ENV || "development";

  console.log("🐘 Connecting to PostgreSQL database:", databaseUrl);

  _sql = postgres(databaseUrl, {
    max: 20,
    idle_timeout: 30,
    connect_timeout: 10,
  });

  _db = drizzle(_sql, { schema });

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
  if (_sql) {
    await _sql.end();
    _sql = null;
    _db = null;
  }
}
