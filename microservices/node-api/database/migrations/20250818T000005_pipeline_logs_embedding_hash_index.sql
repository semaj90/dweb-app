-- Index to speed up deduplication lookups by embedding_hash
CREATE INDEX IF NOT EXISTS idx_pipeline_logs_embedding_hash ON pipeline_logs(embedding_hash);
