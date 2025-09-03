import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { ensureEvidenceTable, query } from '$lib/server/db/client.js';
import { minioService } from '$lib/server/storage/minio-service.js';
import crypto from 'crypto';

function sha256(buf: Buffer) { return crypto.createHash('sha256').update(buf).digest('hex'); }

export const GET: RequestHandler = async ({ url }) => {
  await ensureEvidenceTable();

  // Presigned download
  const downloadId = url.searchParams.get('download');
  if (downloadId) {
    const { rows } = await query(`SELECT storage_bucket, object_name, mime_type, title FROM evidence_files WHERE id=$1`, [downloadId]);
    if (!rows.length) return json({ success: false, error: 'Not found' }, { status: 404 });
    const rec = rows[0] as any;
    await minioService.initialize();
    const urlSigned = await minioService.getFileUrl(rec.storage_bucket, rec.object_name, 60 * 10); // 10 min expiry
    return json({ success: true, url: urlSigned, fileName: rec.object_name, title: rec.title });
  }

  const id = url.searchParams.get('id');
  if (id) {
    const { rows } = await query(`SELECT * FROM evidence_files WHERE id=$1`, [id]);
    if (!rows.length) return json({ success: false, error: 'Not found' }, { status: 404 });
    return json({ success: true, record: rows[0] });
  }
  const limit = parseInt(url.searchParams.get('limit') || '50', 10);
  const { rows } = await query(`SELECT id, title, evidence_type, file_size, mime_type, uploaded_at FROM evidence_files ORDER BY uploaded_at DESC LIMIT $1`, [limit]);
  return json({ success: true, items: rows });
};

export const POST: RequestHandler = async ({ request }) => {
  await ensureEvidenceTable();
  const formData = await request.formData();
  const file = formData.get('file') as File | null;
  if (!file) return json({ success: false, error: 'file missing' }, { status: 400 });
  const title = (formData.get('title') as string) || file.name;
  const case_id = formData.get('case_id') as string | null;
  const evidence_type = (formData.get('evidence_type') as string) || 'UNKNOWN';
  await minioService.initialize();
  const arrayBuf = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuf);
  const checksum = sha256(buffer);

  // Check for duplicate by checksum
  const existing = await query(`SELECT id FROM evidence_files WHERE checksum=$1 LIMIT 1`, [checksum]);
  if (existing.rows.length) {
    return json({ success: true, duplicate: true, id: existing.rows[0].id });
  }
  const upload = await minioService.uploadFile(buffer, file.name, { bucket: 'legal-documents' });
  if (!upload.success) return json({ success: false, error: upload.error || 'upload failed' }, { status: 500 });
  const { rows } = await query(`INSERT INTO evidence_files (case_id, title, description, evidence_type, storage_bucket, object_name, file_size, mime_type, file_type, uploaded_by, tags, confidentiality_level, is_admissible, metadata, checksum)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10, ARRAY[]::TEXT[], 'standard', TRUE, $11, $12)
    RETURNING id, uploaded_at`, [case_id, title, null, evidence_type, upload.metadata.bucket, upload.metadata.fileName, upload.metadata.fileSize, upload.metadata.mimeType, upload.metadata.fileType, 1, JSON.stringify({}), checksum]);
  return json({ success: true, record: rows[0], upload });
};

export const DELETE: RequestHandler = async ({ url }) => {
  await ensureEvidenceTable();
  const id = url.searchParams.get('id');
  if (!id) return json({ success: false, error: 'id required' }, { status: 400 });
  const { rows } = await query(`DELETE FROM evidence_files WHERE id=$1 RETURNING storage_bucket, object_name`, [id]);
  if (!rows.length) return json({ success: false, error: 'Not found' }, { status: 404 });
  const { storage_bucket, object_name } = rows[0] as any;
  await minioService.initialize();
  await minioService.deleteFile(storage_bucket, object_name);
  return json({ success: true, id });
};
