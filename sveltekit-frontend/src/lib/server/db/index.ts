// Database connection and schema exports
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
// Import the main schema (includes vector support)
import * as schema from "../db/schema";

// Database type helper - exported first to avoid temporal dead zone
export const isPostgreSQL = true;

// Use the main schema
export const fullSchema = schema;

// Create the connection
const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3";

// For query purposes
const queryClient = postgres(connectionString);
export const db = drizzle(queryClient, { schema: fullSchema });

// For migrations
const migrationClient = postgres(connectionString, { max: 1 });
export const migrationDb = drizzle(migrationClient);

// Export all schemas and types
export * from "../db/schema";

// Helper function to test database connection
export async function testConnection() {
  try {
    await queryClient`SELECT 1`;
    console.log("✅ Database connection successful");

    // Check for pgvector extension
    const result = await queryClient`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as has_vector
    `;

    if (result[0].has_vector) {
      console.log("✅ pgvector extension is installed");
    } else {
      console.log("⚠️  pgvector extension not found, installing...");
      await queryClient`CREATE EXTENSION IF NOT EXISTS vector`;
      console.log("✅ pgvector extension installed");
    }
    return true;
  } catch (error) {
    console.error("❌ Database connection failed:", error);
    return false;
  }
}
// Initialize pgvector on first run
if (process.env.NODE_ENV !== "production") {
  testConnection().catch(console.error);
}
