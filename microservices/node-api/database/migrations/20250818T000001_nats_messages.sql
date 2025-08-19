-- Drizzle manual migration for nats_messages with JSONB payload and index
CREATE TABLE IF NOT EXISTS nats_messages (
  id SERIAL PRIMARY KEY,
  subject TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_nats_messages_subject_created_at ON nats_messages(subject, created_at DESC);
