-- Migration: add packed_embedding and embedding_scale to embedding_cache
ALTER TABLE embedding_cache
  ADD COLUMN IF NOT EXISTS packed_embedding text,
  ADD COLUMN IF NOT EXISTS embedding_scale numeric(10,6);
