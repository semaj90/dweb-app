import { Pool } from 'pg';

const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db';

export const pgPool = new Pool({ connectionString });

export async function query<T=any>(text: string, params?: any[]): Promise<{ rows: T[] }> {
  const client = await pgPool.connect();
  try {
    const res = await client.query(text, params);
    return { rows: res.rows };
  } finally {
    client.release();
  }
}

export async function ensureEvidenceTable() {
  await query(`CREATE TABLE IF NOT EXISTS evidence_files (
    id SERIAL PRIMARY KEY,
    case_id UUID NULL,
    title TEXT NOT NULL,
    description TEXT NULL,
    evidence_type TEXT NOT NULL DEFAULT 'UNKNOWN',
    storage_bucket TEXT NOT NULL,
    object_name TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type TEXT NOT NULL,
    file_type TEXT NOT NULL,
    uploaded_by INT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    confidentiality_level TEXT NOT NULL DEFAULT 'standard',
    is_admissible BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NULL,
    embeddings VECTOR(768) NULL,
    checksum TEXT NULL,
    UNIQUE(storage_bucket, object_name)
  );`);
  await query(`CREATE INDEX IF NOT EXISTS idx_evidence_files_case_id ON evidence_files(case_id);`);
  await query(`CREATE INDEX IF NOT EXISTS idx_evidence_files_checksum ON evidence_files(checksum);`);
}
