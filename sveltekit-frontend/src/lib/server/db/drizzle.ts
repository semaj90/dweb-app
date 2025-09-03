
// src/lib/server/db/drizzle.ts
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from './schema-postgres';
import type { PostgresJsDatabase } from "drizzle-orm/postgres-js";

// Create a mock sql for build time
const createMockSql = () =>
  ({
    connect: () =>
      Promise.reject(new Error("Database not available during build")),
    end: () => Promise.resolve(),
    query: () =>
      Promise.reject(new Error("Database not available during build")),
  }) as any;

// Database configuration
const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://postgres:123456@localhost:5432/legal_ai_db";

// Create postgres client - use mock during build or when DATABASE_URL indicates build environment
const isBuilding =
  process.env.NODE_ENV === "production" ||
  process.env.DATABASE_URL?.includes("build") ||
  process.env.BUILDING === "true";

let sql: any;
let db: PostgresJsDatabase<typeof schema>;

try {
  if (isBuilding) {
    console.log('[DB] Using mock database for build environment');
    sql = createMockSql();
  } else {
    console.log('[DB] Initializing PostgreSQL connection:', {
      connectionString: connectionString.replace(/:[^@]*@/, ':****@'), // Hide password in logs
      isBuilding
    });
    
    sql = postgres(connectionString, {
      max: 20,
      idle_timeout: 30,
      connect_timeout: 10,
    });
    
    // Test connection
    sql`SELECT 1 as test`.then(() => {
      console.log('[DB] ✅ PostgreSQL connection successful');
    }).catch((err) => {
      console.error('[DB] ❌ PostgreSQL connection failed:', err.message);
    });
  }

  // Properly typed database instance  
  db = drizzle(sql, {
    schema,
    logger: false, // Disable verbose logging in development
  });
  
  console.log('[DB] Drizzle database instance created');

} catch (error) {
  console.error('[DB] Failed to initialize database:', error);
  
  // Fallback to mock if initialization fails
  sql = createMockSql();
  db = drizzle(sql, {
    schema,
    logger: false,
  });
}

export { sql, db };

// Export sql from drizzle-orm for query building (different from postgres client)
export { sql as drizzleSql } from "drizzle-orm";
