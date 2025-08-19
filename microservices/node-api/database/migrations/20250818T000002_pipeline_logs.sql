-- Drizzle manual migration for pipeline_logs
CREATE TABLE IF NOT EXISTS pipeline_logs (
  id SERIAL PRIMARY KEY,
  message_id TEXT NOT NULL,
  gpu TEXT NULL,
  wasm TEXT NULL,
  llm TEXT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_message_created ON pipeline_logs(message_id, created_at DESC);
