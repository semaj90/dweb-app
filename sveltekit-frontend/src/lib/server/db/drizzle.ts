// src/lib/server/db/drizzle.ts
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema-postgres";

const connectionString =
  process.env.DATABASE_URL ||
  "postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db";

export const pool = postgres(connectionString);
export const db = drizzle(pool, { schema });

export { sql } from "drizzle-orm";
