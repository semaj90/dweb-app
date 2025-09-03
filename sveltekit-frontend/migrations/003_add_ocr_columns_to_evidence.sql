-- Normalized to varchar columns (matching runtime Drizzle schema) to avoid type mismatch.
-- If numeric comparisons become common, consider later casting or migrating types.

ALTER TABLE evidence
  ADD COLUMN IF NOT EXISTS ocr_confidence varchar(32),
  ADD COLUMN IF NOT EXISTS ocr_word_count varchar(32),
  ADD COLUMN IF NOT EXISTS ocr_processing_time_ms varchar(32),
  ADD COLUMN IF NOT EXISTS ocr_metadata jsonb DEFAULT '{}'::jsonb;

-- Optional indexes if querying by OCR quality/performance later:
-- CREATE INDEX IF NOT EXISTS evidence_ocr_confidence_idx ON evidence(ocr_confidence);
-- CREATE INDEX IF NOT EXISTS evidence_ocr_word_count_idx ON evidence(ocr_word_count);

-- Down (manual rollback example):
-- ALTER TABLE evidence DROP COLUMN IF EXISTS ocr_metadata;
-- ALTER TABLE evidence DROP COLUMN IF EXISTS ocr_processing_time_ms;
-- ALTER TABLE evidence DROP COLUMN IF EXISTS ocr_word_count;
-- ALTER TABLE evidence DROP COLUMN IF EXISTS ocr_confidence;
