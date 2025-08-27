
import "dotenv/config";
import { drizzle } from "drizzle-orm/node-postgres";
import { migrate } from "drizzle-orm/node-postgres/migrator";
import { Pool } from "pg";

async function runMigrations() {
  const databaseUrl = process.env.DATABASE_URL || process.env?.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error("DATABASE_URL environment variable is not set.");
  }
  const pool = new Pool({ connectionString: databaseUrl });
  const db = drizzle(pool);

  console.log("⏳ Running database migrations...");
  console.log(
    "📍 Database URL:",
    process.env.DATABASE_URL.replace(/\/\/[^:]+:[^@]+@/, "//***:***@"),
  );

  try {
    await migrate(db, { migrationsFolder: "./drizzle" });
    console.log("✅ Migrations completed successfully.");
  } catch (error) {
    console.error("❌ Migration failed:", error);
    throw error;
  } finally {
    // Close the connection pool
    await pool.end();
  }
}
runMigrations().catch((err) => {
  console.error("❌ Migration failed:", err);
  process.exit(1);
});
