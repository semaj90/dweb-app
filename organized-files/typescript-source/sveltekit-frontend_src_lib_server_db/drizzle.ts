// @ts-nocheck
// src/lib/server/db/drizzle.ts
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema-postgres";
import type { NodePgDatabase } from "drizzle-orm/node-postgres";

// Create a mock pool for build time
const createMockPool = () =>
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
  "postgresql://legal_admin:123456@localhost:5432/legal_ai_db";

// Create pool - use mock during build or when DATABASE_URL indicates build environment
const isBuilding =
  process.env.NODE_ENV === "build" ||
  process.env.DATABASE_URL?.includes("build");

export const pool = isBuilding
  ? createMockPool()
  : new Pool({
      connectionString,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

// Properly typed database instance
export const db: NodePgDatabase<typeof schema> = drizzle(pool, {
  schema,
  logger: false,
});

export { sql } from "drizzle-orm";
