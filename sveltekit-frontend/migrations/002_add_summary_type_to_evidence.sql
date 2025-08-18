-- Migration: Add summary_type column to evidence and index for filtering
-- Reversible (down) section included as comments for manual rollback.

ALTER TABLE evidence ADD COLUMN IF NOT EXISTS summary_type text;
CREATE INDEX IF NOT EXISTS evidence_summary_type_idx ON evidence(summary_type);

-- Down (manual):
-- DROP INDEX IF EXISTS evidence_summary_type_idx;
-- ALTER TABLE evidence DROP COLUMN IF EXISTS summary_type;
