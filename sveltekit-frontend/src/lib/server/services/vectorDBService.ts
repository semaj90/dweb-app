// @ts-nocheck
import { db } from "$lib/server/db/drizzle";
// Orphaned content: import {
errorLogs // A new table you'll need to create

// This function stores the log and its embedding in PostgreSQL
export async function storeLogInVectorDB(data: { log: any; embedding: number[] }) {
  // You will need to add an 'error_logs' table to your Drizzle schema
  // with a 'vector' column to store the embedding.
  await db.insert(errorLogs).values({
    message: data.log.message,
    stackTrace: data.log.stack, // Assuming stack is part of your log data
    embedding: data.embedding,
    metadata: data.log, // Store the full log as JSONB for details
  });
}