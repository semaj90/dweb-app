// @ts-nocheck
import { building } from "$app/environment";
// Orphaned content: import * as schema from "$lib/server/db/schema-postgres";
import {

dotenv from "dotenv";
// Orphaned content: import {
drizzle, type NodePgDatabase
import { migrate } from "drizzle-orm/node-postgres/migrator";
// Orphaned content: import {

// Load environment-specific variables
const envFile = `.env.${import.meta.env.NODE_ENV || "development"}`;
dotenv.config({ path: envFile });

let _db: NodePgDatabase<typeof schema> | null = null;
let _pool: Pool | null = null;

function initializeDatabase(): NodePgDatabase<typeof schema> | null {
  // Skip database initialization during SvelteKit build
  if (building) {
    console.log("Skipping database initialization during build");
    return null;
  }
  if (_db) return _db;

  // Use PostgreSQL for all environments
  const databaseUrl =
    import.meta.env.DATABASE_URL ||
    "postgresql://postgres:postgres@localhost:5432/prosecutor_db";
  const nodeEnv = import.meta.env.NODE_ENV || "development";

  console.log("🔧 Database Configuration:");
  console.log("  NODE_ENV:", nodeEnv);
  console.log("  DATABASE_URL:", databaseUrl);

  // PostgreSQL connection
  console.log("🐘 Connecting to PostgreSQL database:", databaseUrl);

  _pool = new Pool({
    connectionString: databaseUrl,
  });

  _db = drizzle(_pool, { schema });

  // Skip migrations in testing environment
  if (nodeEnv !== "testing") {
    try {
      console.log("Running PostgreSQL migrations...");
      migrate(_db, { migrationsFolder: "./drizzle" });
      console.log("✅ PostgreSQL migrations completed");
    } catch (error) {
      console.error("❌ PostgreSQL migration error:", error);
    }
  }
  console.log("✅ PostgreSQL database connected successfully");
  return _db;
}
// Main database connection
export const db: NodePgDatabase<typeof schema> = new Proxy({} as any, {
  get(target, prop, receiver) {
    const database = initializeDatabase();
    if (!database) {
      throw new Error("Database not initialized");
    }
    return Reflect.get(database, prop, receiver);
  },
});

// Call Qdrant tag seeding on startup
initializeQdrantCollection().catch((err) => {
  console.error("Qdrant tag seeding failed:", err);
});

// Database metadata
export const isPostgreSQL = true;
export const isSQLite = false;
export const pool = _pool;

// Schema exports
export * from "$lib/server/db/schema-postgres";

// Graceful shutdown
export function closeDatabase() {
  if (_pool) {
    console.log("Closing PostgreSQL connection pool...");
    _pool.end();
  }
}
