import express from 'express';
import multer from 'multer';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
// Qdrant client library exports { QdrantClient }; handle both default and named
let QdrantClient;
try {
  const qdrantMod = require('@qdrant/js-client-rest');
  QdrantClient = qdrantMod.QdrantClient || qdrantMod.default || qdrantMod; // fallback
} catch (e) {
  console.warn('Failed to load @qdrant/js-client-rest:', e?.message || e);
}
const Minio = require('minio');
import fetch from 'node-fetch';
import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const PORT = process.env.STANDALONE_PORT || 4000;

// Env / service config
const MINIO_ENDPOINT = process.env.MINIO_ENDPOINT || process.env.MINIO_HOST || 'localhost';
const MINIO_PORT = parseInt(process.env.MINIO_PORT || '9000', 10);
const MINIO_USE_SSL = (process.env.MINIO_USE_SSL || 'false') === 'true';
const MINIO_ACCESS_KEY = process.env.MINIO_ACCESS_KEY || 'minioadmin';
const MINIO_SECRET_KEY = process.env.MINIO_SECRET_KEY || 'minioadmin';
const MINIO_BUCKET = process.env.MINIO_BUCKET || 'evidence';

const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';
const OLLAMA_URL = process.env.OLLAMA_URL || process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const POSTGRES_URL = process.env.POSTGRES_URL || process.env.DATABASE_URL || null;
const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL || null; // e.g. http://localhost:8601

// Clients
let minioClient = null;
let qdrantClient = null;
let pgPool = null;

try {
  minioClient = new Minio.Client({
    endPoint: MINIO_ENDPOINT,
    port: MINIO_PORT,
    useSSL: MINIO_USE_SSL,
    accessKey: MINIO_ACCESS_KEY,
    secretKey: MINIO_SECRET_KEY
  });
  console.log('MinIO client initialized');
} catch (e) {
  console.warn('MinIO client not initialized:', e?.message || e);
}

try {
  if (QdrantClient) {
    qdrantClient = new QdrantClient({ url: QDRANT_URL });
    console.log('Qdrant client initialized');
  }
} catch (e) {
  console.warn('Qdrant client not initialized:', e?.message || e);
}

if (POSTGRES_URL) {
  pgPool = new Pool({ connectionString: POSTGRES_URL });
  console.log('Postgres pool initialized');
}

async function ensureMinioBucket(bucket) {
  if (!minioClient) return false;
  try {
    const exists = await minioClient.bucketExists(bucket);
    if (!exists) await minioClient.makeBucket(bucket);
    return true;
  } catch (e) {
    console.warn('MinIO ensure bucket error:', e?.message || e);
    return false;
  }
}

async function putMinioObject(bucket, objectName, buffer, meta) {
  if (!minioClient) throw new Error('MinIO client not configured');
  return new Promise((resolve, reject) => {
    minioClient.putObject(bucket, objectName, buffer, meta || {}, (err, etag) => {
      if (err) return reject(err);
      resolve({ etag });
    });
  });
}

app.get('/', (req, res) => {
  res.json({ ok: true, message: 'Standalone upload server' });
});

app.post('/upload', upload.single('file'), async (req, res) => {
  const file = req.file;
  const metadata = {};
  const result = {
    savedToMinio: false,
    minioObject: null,
    ocr: null,
    embedding: null,
    qdrantUpsert: null,
    pgInsert: null,
    errors: []
  };

  if (!file) return res.status(400).json({ error: 'No file uploaded' });

  try {
    // Save to local uploads folder for inspection
    const uploadsDir = path.join(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });
    const localPath = path.join(uploadsDir, `${Date.now()}_${file.originalname}`);
    fs.writeFileSync(localPath, file.buffer);
    metadata.localPath = localPath;

    // MinIO
    if (minioClient) {
      await ensureMinioBucket(MINIO_BUCKET);
      const objectName = `${Date.now()}_${file.originalname}`;
      await putMinioObject(MINIO_BUCKET, objectName, file.buffer, { 'content-type': file.mimetype });
      result.savedToMinio = true;
      result.minioObject = { bucket: MINIO_BUCKET, objectName };
    }

    // OCR / text extraction (lazy-load heavy libs) with optional delegation
    let extractedText = '';
    const isPdf = file.mimetype === 'application/pdf';
    const isImage = file.mimetype && file.mimetype.startsWith('image/');
    if (OCR_SERVICE_URL && (isPdf || isImage)) {
      try {
        // Delegate to external OCR microservice
        const fdModule = await import('form-data');
        const FormDataCls = fdModule.default || fdModule;
        const form = new FormDataCls();
        form.append('file', file.buffer, { filename: file.originalname, contentType: file.mimetype });
        const ocrResp = await fetch(`${OCR_SERVICE_URL}/api/ocr/extract`, { method: 'POST', body: form });
        if (!ocrResp.ok) throw new Error(`OCR service responded ${ocrResp.status}`);
        const ocrJson = await ocrResp.json();
        extractedText = ocrJson.text || '';
        result.ocr = { delegated: true, type: ocrJson.mimeType ? (ocrJson.mimeType.startsWith('image/') ? 'image' : 'pdf') : (isPdf ? 'pdf' : 'image'), length: extractedText.length, confidence: ocrJson.confidence };
      } catch (e) {
        result.errors.push({ stage: 'ocr-delegated', message: e?.message || String(e) });
      }
    }
    if (!extractedText) {
      if (isPdf) {
        try {
          const pdfModule = await import('pdf-parse');
          const pdfFn = pdfModule.default || pdfModule;
          const pdfData = await pdfFn(file.buffer);
          extractedText = pdfData.text || '';
          result.ocr = { ...(result.ocr||{}), type: 'pdf', length: extractedText.length };
        } catch (e) {
          result.errors.push({ stage: 'pdf-parse', message: e?.message || String(e) });
        }
      } else if (isImage) {
        try {
          const tesseract = await import('tesseract.js');
          const { createWorker } = tesseract;
          const worker = createWorker();
          await worker.load();
          await worker.loadLanguage('eng');
          await worker.initialize('eng');
          const { data } = await worker.recognize(file.buffer);
          extractedText = data?.text || '';
          result.ocr = { ...(result.ocr||{}), type: 'image', words: data?.words?.length || 0, length: extractedText.length };
          await worker.terminate();
        } catch (e) {
          result.errors.push({ stage: 'tesseract', message: e?.message || String(e) });
        }
      } else {
        extractedText = file.buffer.toString('utf-8');
        result.ocr = { ...(result.ocr||{}), type: 'text', length: extractedText.length };
      }
    }

    // Embeddings via Ollama (if available)
    if (OLLAMA_URL && extractedText.length > 0) {
      try {
        const embResp = await fetch(`${OLLAMA_URL}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: 'nomic-embed-text', input: extractedText.substring(0, 8192) })
        });
        const embJson = await embResp.json();
        // Expect embJson to have field 'embedding' or 'embeddings'
        result.embedding = embJson.embedding || embJson.embeddings || null;
      } catch (e) {
        result.errors.push({ stage: 'ollama-embeddings', message: e?.message || e });
      }
    }

    // Qdrant upsert
  if (qdrantClient && result.embedding) {
      try {
        const collectionName = 'standalone_uploads';
        // Ensure collection exists - best effort
    const vecSize = Array.isArray(result.embedding) ? result.embedding.length : (Array.isArray(result.embedding[0]) ? result.embedding[0].length : null);
    const vector = Array.isArray(result.embedding[0]) ? result.embedding[0] : result.embedding; // normalize possible nested shape
    if (!Array.isArray(vector)) throw new Error('Invalid embedding vector shape');
    try { await qdrantClient.collections.create({ collection_name: collectionName, vectors: { size: vector.length, distance: 'Cosine' } }); } catch (e) {/* collection may already exist */}
    await qdrantClient.points.upsert({ collection_name: collectionName, points: [{ id: Date.now().toString(), vector, payload: { filename: file.originalname, size: file.size, mimetype: file.mimetype } }] });
        result.qdrantUpsert = { collection: collectionName };
      } catch (e) {
        result.errors.push({ stage: 'qdrant-upsert', message: e?.message || e });
      }
    }

    // Postgres insert (best-effort)
    if (pgPool) {
      try {
        const insertText = `INSERT INTO evidence_test(filename, content, created_at) VALUES($1, $2, now()) RETURNING id`;
        const r = await pgPool.query(insertText, [file.originalname, extractedText.substring(0, 100000)]);
        result.pgInsert = { id: r.rows[0]?.id };
      } catch (e) {
        result.errors.push({ stage: 'postgres-insert', message: e?.message || e });
      }
    }

    return res.json({ ok: true, result });
  } catch (err) {
    console.error('Standalone upload error:', err);
    return res.status(500).json({ error: err?.message || String(err) });
  }
});

app.listen(PORT, () => {
  console.log(`Standalone upload server listening on http://localhost:${PORT}`);
});
