// src/lib/server/db/drizzle.ts
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { env } from '$env/dynamic/private';

const connectionString = env.DATABASE_URL || 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db';
const client = postgres(connectionString);
export const db = drizzle(client);

export { sql } from 'drizzle-orm';