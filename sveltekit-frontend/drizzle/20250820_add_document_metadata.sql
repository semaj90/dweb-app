-- Migration: Add document_metadata table for Enhanced RAG service
-- Date: 2025-08-20

-- Create document_metadata table
CREATE TABLE IF NOT EXISTS "document_metadata" (
    "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    "case_id" uuid REFERENCES "cases"("id") ON DELETE CASCADE,
    "original_filename" varchar(512) NOT NULL,
    "summary" text,
    "content_type" varchar(100),
    "file_size" integer,
    "processing_status" varchar(50) DEFAULT 'pending',
    "metadata" jsonb DEFAULT '{}' NOT NULL,
    "created_at" timestamp DEFAULT now() NOT NULL,
    "updated_at" timestamp DEFAULT now() NOT NULL
);

-- Add indexes
CREATE INDEX IF NOT EXISTS "doc_metadata_case_id_idx" ON "document_metadata" ("case_id");
CREATE INDEX IF NOT EXISTS "doc_metadata_filename_idx" ON "document_metadata" ("original_filename");

-- Alter document_embeddings table to add document_id reference
ALTER TABLE "document_embeddings" 
ADD COLUMN IF NOT EXISTS "document_id" uuid REFERENCES "document_metadata"("id") ON DELETE CASCADE;

-- Add index for document_id
CREATE INDEX IF NOT EXISTS "doc_embeddings_document_id_idx" ON "document_embeddings" ("document_id");

-- Insert some sample data for testing
INSERT INTO "document_metadata" ("case_id", "original_filename", "summary", "content_type", "processing_status") 
SELECT 
    c.id, 
    'legal_document_' || c.title || '.pdf',
    'Sample legal document summary for case: ' || c.title,
    'application/pdf',
    'processed'
FROM "cases" c 
WHERE NOT EXISTS (
    SELECT 1 FROM "document_metadata" dm WHERE dm.case_id = c.id
)
LIMIT 10;