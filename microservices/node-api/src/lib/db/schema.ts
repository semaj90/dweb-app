import { pgTable, serial, text, timestamp } from 'drizzle-orm/pg-core';

export const nats_messages = pgTable('nats_messages', {
  id: serial('id').primaryKey(),
  subject: text('subject').notNull(),
  payload: text('payload').notNull(),
  created_at: timestamp('created_at').defaultNow()
});

export const pipeline_logs = pgTable('pipeline_logs', {
  id: serial('id').primaryKey(),
  message_id: text('message_id').notNull(),
  gpu: text('gpu'),
  wasm: text('wasm'),
  llm: text('llm'),
  embedding: text('embedding'),
  embedding_hash: text('embedding_hash'),
  retrieval: text('retrieval'),
  context: text('context'),
  created_at: timestamp('created_at').defaultNow()
});
