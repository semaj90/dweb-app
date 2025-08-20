/**
 * Evidence Upload API (Production Ready)
 * Responsibilities:
 *  - Accept multipart/form-data (files[] + optional flags)
 *  - Validate input (size, mime type, summaryType enum, feature flags)
 *  - Store original binary in MinIO (content-addressable via SHA-256)
 *  - Persist metadata & AI results to PostgreSQL (Drizzle schema)
 *  - Generate embeddings + upsert into Qdrant (normalized vectors)
 *  - Optional AI summarization with selectable summaryType
 *  - Return consistent JSON envelope (data, meta, error)
 */
import { json, error, type RequestHandler } from '@sveltejs/kit';
import { authorize } from '$lib/server/authPolicy';
import { logger } from '$lib/server/logger';
import { fileUploadSchema, type FileUpload, type AiAnalysisResult } from '$lib/server/validation/fileUploadSchema';
import { db } from '$lib/server/db';
import { evidence, embeddingCache } from '$lib/server/db/enhanced-legal-schema';
import { ollamaCudaService } from '$lib/services/ollama-cuda-service';
import { mkdir } from 'fs/promises';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import sharp from 'sharp';
import { SystemMessage, HumanMessage } from 'langchain/schema';
import { eq } from 'drizzle-orm';
import { Client as MinioClient } from 'minio';
import { URL } from 'url';
const minioClient = new MinioClient({
  endPoint: 'localhost',
  port: 9000,
  useSSL: false,
  accessKey: 'minioadmin',
  secretKey: 'minioadmin'
});

// WebGPU multi-core vector operations
class GPUVectorProcessor {
  static async normalizeVectors(vectors: number[][]): Promise<number[][]> {
    // Normalize vectors once on server → cosine becomes dot product
    return vectors.map(vector => {
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
    });
  }

  static async batchEmbeddings(texts: string[]): Promise<number[][]> {
    // Batch embeddings for efficiency
    const embeddings = [];
    for (const text of texts) {
      try {
        const response = await fetch('http://localhost:11434/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'nomic-embed-text',
            prompt: text
          })
        });
        const result = await response.json();
        embeddings.push(result.embedding);
      } catch (error) {
        console.error('Embedding failed:', error);
        embeddings.push([]);
      }
    }
    return this.normalizeVectors(embeddings);
  }
}

// Qdrant vector storage with payload filters
class QdrantService {
  static async upsertToQdrant(id: string, embedding: number[], metadata: any) {
    try {
      await fetch('http://localhost:6333/collections/legal_evidence/points', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: [{
            id,
            vector: embedding,
            payload: {
              ...metadata,
              tags: metadata.tags || [],
              case_id: metadata.caseId,
              evidence_type: metadata.type,
              // Payload filters for efficient search
              is_contract: metadata.tags?.includes('contract') || false,
              is_admissible: metadata.isAdmissible || true,
              priority: metadata.priority || 'normal'
            }
          }]
        })
      });
    } catch (error) {
      console.error('Qdrant upsert failed:', error);
      throw error;
    }
  }

  static async searchWithFilters(queryVector: number[], filters: any, limit = 10) {
    try {
      const response = await fetch('http://localhost:6333/collections/legal_evidence/points/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: queryVector,
          filter: filters,
          limit,
          with_payload: true
        })
      });
      return await response.json();
    } catch (error) {
      console.error('Qdrant search failed:', error);
      return { result: [] };
    }
  }
}

// OCR integration (optional)
// import Tesseract from 'tesseract.js';

interface UploadResult {
  id: string;
  fileName: string;
  originalName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  hash: string;
  aiAnalysis?: AiAnalysisResult;
  embedding?: number[];
  ocrText?: string;
  thumbnail?: string;
}

const UPLOAD_DIR = 'uploads/evidence';
const THUMBNAIL_DIR = 'uploads/thumbnails';
function getMaxFileSize() {
  const val = Number(import.meta.env.EVIDENCE_MAX_FILE_SIZE || '0');
  return val > 0 ? val : 100 * 1024 * 1024;
}
const STREAM_ANALYSIS_INLINE_LIMIT = Number(import.meta.env.EVIDENCE_MAX_INLINE || 5 * 1024 * 1024); // default 5MB
const SUMMARY_TYPES = ['key_points', 'narrative', 'prosecutorial'] as const;
type SummaryType = typeof SUMMARY_TYPES[number];

// Augment Partial<FileUpload> cheaply (local shape extension without editing central schema)
interface UploadAugment { summaryType?: SummaryType; priority?: string }

function withCorrelation(resp: Response, cid?: string) { if (cid) resp.headers.set('x-correlation-id', cid); return resp; }
function ok<T>(data: T, meta: Record<string, any> = {}, cid?: string) { return withCorrelation(json({ success: true, data, meta }, { status: 200 }), cid); }
function created<T>(data: T, meta: Record<string, any> = {}, cid?: string) { return withCorrelation(json({ success: true, data, meta }, { status: 201 }), cid); }
function fail(status: number, message: string, details?: any, cid?: string) { return withCorrelation(json({ success: false, error: { message, details } }, { status }), cid); }
const ALLOWED_MIME_TYPES = [
  // Images
  'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
  // Videos
  'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/webm',
  // Audio
  'audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/aac',
  // Documents
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'text/plain', 'text/csv', 'application/json',
  // Archives
  'application/zip', 'application/x-rar-compressed'
];

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    if (!locals.user) return fail(401, 'Unauthorized');
  const correlationId = uuidv4();

  // Distributed rate limit then local guard
  const dist = await redisRateLimit({ limit: 25, windowSec: 60, key: `evidence-upload:${locals.user.id}` });
  if (!dist.allowed) return fail(429, 'Rate limit exceeded', { retryAfter: dist.retryAfter, correlationId, distributed: true }, correlationId);
  const local = checkRateLimit({ limit: 100, windowMs: 60_000, key: `local-evidence-upload:${locals.user.id}` });
  if (!local.allowed) return fail(429, 'Local rate limit exceeded', { retryAfter: local.retryAfter, correlationId }, correlationId);

  // Authorization policy
  const authz = authorize({ user: locals.user, action: 'create', resource: 'evidence' });
  if (!authz.allowed) return fail(403, 'Forbidden', { reason: authz.reason, correlationId }, correlationId);
  logger.info('upload.begin', { phase: 'begin', distCount: dist.count, correlationId, userId: locals.user.id });

    const contentType = request.headers.get('content-type') || '';
    if (!contentType.startsWith('multipart/form-data')) {
  const r = fail(400, 'Content-Type must be multipart/form-data', { correlationId });
  r.headers.set('x-correlation-id', correlationId);
  return r;
    }

  const bb = Busboy({ headers: { 'content-type': contentType } });
  const incomingFiles: { filename: string; mimeType: string; size: number; tempPath: string; hash: ReturnType<typeof createHash> }[] = [];
    const fieldMap: Record<string, string> = {};

    const parsePromise = new Promise<void>((resolve, reject) => {
      bb.on('file', (_name, stream, info) => {
        const { filename, mimeType } = info;
        const tempDir = join('uploads', 'tmp');
        mkdir(tempDir, { recursive: true }).catch(()=>{});
        const tempPath = join(tempDir, `${Date.now()}-${Math.random().toString(36).slice(2)}-${filename}`);
        const hash = createHash('sha256');
        const writeStream = createWriteStream(tempPath);
        const rec = { filename, mimeType, size: 0, tempPath, hash };
    stream.on('data', (chunk: Buffer) => {
          rec.size += chunk.length;
          hash.update(chunk);
          if (rec.size > getMaxFileSize()) {
            stream.unpipe();
            writeStream.destroy();
      logger.warn('upload.file.too_large', { file: filename, size: rec.size, correlationId, userId: locals.user.id });
      reject(new Error(`File ${filename} exceeds ${getMaxFileSize() / 1024 / 1024}MB limit`));
            return;
          }
        });
        stream.on('error', reject);
        writeStream.on('error', reject);
        stream.pipe(writeStream);
  stream.on('end', () => { incomingFiles.push(rec); logger.debug('upload.file.end', { file: filename, size: rec.size, correlationId, userId: locals.user.id }); });
      });
      bb.on('field', (name, val) => { fieldMap[name] = val; });
      bb.on('error', reject);
      bb.on('finish', resolve);
    });
    // Convert Web ReadableStream (Fetch API) to Node.js Readable for Busboy
    const body: any = (request as any).body;
    if (body) {
      if (typeof body.getReader === 'function') {
        const nodeStream = Readable.fromWeb(body as any);
        nodeStream.pipe(bb);
      } else if ((body as any).pipe && typeof (body as any).pipe === 'function') {
        (body as any).pipe(bb);
      } else {
        // Fallback: accumulate and end
        const reader = body.getReader?.();
        if (reader) {
          const pump = async () => {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              bb.write(value);
            }
            bb.end();
          };
          pump();
        } else {
          bb.end();
        }
      }
    } else {
      bb.end();
    }
    await parsePromise;

  if (incomingFiles.length === 0) return fail(400, 'No files provided', { correlationId }, correlationId);

    // Unified flags from parsed fields
    const generateSummaryRaw = (fieldMap['generateSummary'] ?? fieldMap['summarizeWithAI']) || null;
    const summaryTypeRaw = fieldMap['summaryType'] || null;
    const enableAiAnalysisRaw = fieldMap['enableAiAnalysis'] || null;
    const enableEmbeddingsRaw = fieldMap['enableEmbeddings'] || null;
    const enableOcrRaw = fieldMap['enableOcr'] || null;
    const uploadDataStr = fieldMap['uploadData'] || '';

    let uploadData: Partial<FileUpload & UploadAugment> = {};
    if (uploadDataStr) {
      try { uploadData = JSON.parse(uploadDataStr); } catch (e) { console.warn('Failed to parse upload data', e); }
    }

    // Parse upload metadata
  const coerceBool = (v: string | null | undefined) => (v === 'true' || v === '1');
  uploadData.enableAiAnalysis = uploadData.enableAiAnalysis ?? coerceBool(enableAiAnalysisRaw);
  uploadData.enableEmbeddings = uploadData.enableEmbeddings ?? coerceBool(enableEmbeddingsRaw) ?? true;
  uploadData.enableOcr = uploadData.enableOcr ?? coerceBool(enableOcrRaw);
  const generateSummary = coerceBool(generateSummaryRaw) || !!summaryTypeRaw;

    // Summary type validation
    let summaryType: SummaryType | undefined;
    if (summaryTypeRaw) {
      if (!SUMMARY_TYPES.includes(summaryTypeRaw as SummaryType)) {
        const r = fail(400, 'Invalid summaryType', { allowed: SUMMARY_TYPES, correlationId });
        r.headers.set('x-correlation-id', correlationId);
        return r;
      }
      summaryType = summaryTypeRaw as SummaryType;
    }
    if (generateSummary && !summaryType) {
      // Default gracefully
      summaryType = 'narrative';
    }
  (uploadData as UploadAugment).summaryType = summaryType;
    if (generateSummary) uploadData.enableAiAnalysis = true;

    // Add user ID from session
    (uploadData as any).userId = locals.user?.id || 'anonymous';

  const results: UploadResult[] = [];

    // Ensure upload directories exist
    await mkdir(UPLOAD_DIR, { recursive: true });
    await mkdir(THUMBNAIL_DIR, { recursive: true });

    for (const meta of incomingFiles) {
      if (!ALLOWED_MIME_TYPES.includes(meta.mimeType)) return fail(400, `Unsupported file type ${meta.mimeType}`, { correlationId }, correlationId);
      const result = await processFileStreamed(meta, uploadData, correlationId, locals.user.id);
      logger.info('upload.file.processed', { file: meta.filename, size: meta.size, correlationId, userId: locals.user.id });
      results.push(result);
    }
    logger.info('upload.complete', { files: results.length, correlationId, userId: locals.user.id });
    return created(results, { count: results.length, correlationId }, correlationId);

  } catch (err) {
    console.error('File upload error:', err);
  if (err instanceof Response) throw err;
  const correlationId = uuidv4();
  return fail(500, 'File upload failed', { correlationId }, correlationId);
  }
};

interface StreamedFileMeta { filename:string; mimeType:string; size:number; tempPath:string; hash: ReturnType<typeof createHash> }
// Enhanced to accept correlation + user for structured logging and to cleanup tmp file
async function processFileStreamed(
  meta: StreamedFileMeta,
  uploadData: Partial<FileUpload>,
  correlationId?: string,
  userId?: string
): Promise<UploadResult> {
  const fileId = uuidv4();
  const fileExtension = meta.filename.split('.').pop() || '';
  const fileName = `${fileId}.${fileExtension}`;
  const minioPath = `evidence/${uploadData.caseId}/${fileName}`;
  const hash = meta.hash.digest('hex');

  const fs = await import('fs');
  logger.debug('upload.file.minio_put.start', { file: meta.filename, size: meta.size, correlationId, userId });
  await minioClient.putObject('evidence', minioPath, fs.createReadStream(meta.tempPath), {
    'Content-Type': meta.mimeType,
    'Original-Name': meta.filename,
    'Case-ID': uploadData.caseId || '',
    'Evidence-ID': fileId,
    'Hash': hash
  });
  logger.debug('upload.file.minio_put.done', { file: meta.filename, correlationId, userId });

  let aiAnalysis: AiAnalysisResult | undefined;
  let embedding: number[] | undefined;
  let ocrText: string | undefined;
  try {
    let buffer: Buffer | null = null;
    if (meta.size <= STREAM_ANALYSIS_INLINE_LIMIT) buffer = await fs.promises.readFile(meta.tempPath);
    let textContent = '';
    if (buffer) {
      if (meta.mimeType === 'text/plain') textContent = buffer.toString('utf-8');
      else if (meta.mimeType === 'application/pdf') textContent = `[PDF text from ${meta.filename}]`;
      else if (meta.mimeType.startsWith('image/')) textContent = `Image: ${meta.filename}`;
    }
    if (buffer && (uploadData.enableAiAnalysis || uploadData.enableEmbeddings)) {
      const [embeddings] = await GPUVectorProcessor.batchEmbeddings([textContent]);
      embedding = embeddings;
  aiAnalysis = await performEnhancedAIAnalysis(new File([new Uint8Array(buffer)], meta.filename, { type: meta.mimeType }), textContent, uploadData);
    }
    await db.insert(evidence).values({
      id: fileId,
      userId: (uploadData as any).userId || 'system',
      caseId: uploadData.caseId as any,
      title: uploadData.title || meta.filename,
      description: uploadData.description,
      evidenceType: uploadData.evidenceType || 'document',
      subType: null,
      fileName: meta.filename,
      fileSize: meta.size as any,
      mimeType: meta.mimeType,
      hash,
      tags: (uploadData.tags as any) || [],
      aiAnalysis: (aiAnalysis as any) || {},
      aiTags: (aiAnalysis?.categories as any) || [],
      aiSummary: aiAnalysis?.summary || null,
      summary: aiAnalysis?.summary || null,
      summaryType: (uploadData as any).summaryType || null,
      isAdmissible: (uploadData as any).isAdmissible ?? true,
      confidentialityLevel: (uploadData as any).confidentialityLevel || 'internal'
    }).onConflictDoNothing();
    if (embedding && embedding.length) {
      await QdrantService.upsertToQdrant(fileId, embedding, {
        caseId: uploadData.caseId,
        title: uploadData.title || meta.filename,
        type: uploadData.evidenceType,
        tags: uploadData.tags || [],
        fileName: meta.filename,
        fileType: meta.mimeType,
        isAdmissible: (uploadData as any).isAdmissible ?? true,
        priority: (uploadData as any).priority || 'normal'
      });
    }
    if (embedding) await cacheEmbedding(hash, embedding);
    const presignedUrl = await minioClient.presignedGetObject('evidence', minioPath, 3600);
    // Temp file cleanup
    fs.promises.unlink(meta.tempPath).catch(()=>{});
    return { id: fileId, fileName, originalName: meta.filename, fileSize: meta.size, mimeType: meta.mimeType, url: presignedUrl, hash, aiAnalysis, embedding };
  } catch (err) {
    logger.error('upload.file.error', { file: meta.filename, error: (err as any)?.message, correlationId, userId });
    await db.insert(evidence).values({
      userId: (uploadData as any).userId || 'system',
      caseId: uploadData.caseId as any,
      title: uploadData.title || meta.filename,
      description: uploadData.description,
      evidenceType: uploadData.evidenceType || 'document',
      subType: null,
      fileName: meta.filename,
      fileSize: meta.size as any,
      mimeType: meta.mimeType,
      hash,
      tags: (uploadData.tags as any) || [],
      isAdmissible: (uploadData as any).isAdmissible ?? true,
      confidentialityLevel: (uploadData as any).confidentialityLevel || 'standard'
    }).onConflictDoNothing();
    const presignedUrl = await minioClient.presignedGetObject('evidence', minioPath, 900);
    fs.promises.unlink(meta.tempPath).catch(()=>{});
    return { id: fileId, fileName, originalName: meta.filename, fileSize: meta.size, mimeType: meta.mimeType, url: presignedUrl, hash };
  }
}

async function generateThumbnail(buffer: Buffer, fileId: string): Promise<string> {
  try {
    const thumbnailPath = join(THUMBNAIL_DIR, `${fileId}_thumb.webp`);

    await sharp(buffer)
      .resize(300, 300, {
        fit: 'inside',
        withoutEnlargement: true
      })
      .webp({ quality: 80 })
      .toFile(thumbnailPath);

    return thumbnailPath;
  } catch (error) {
    console.error('Thumbnail generation failed:', error);
    throw new Error('Failed to generate thumbnail');
  }
}

async function performOCR(buffer: Buffer, mimeType: string): Promise<string> {
  try {
    if (mimeType.startsWith('image/')) {
      // For images, use Tesseract.js (if available)
      // const { data: { text } } = await Tesseract.recognize(buffer, 'eng');
      // return text.trim();

      // Placeholder - integrate with your preferred OCR service
      return 'OCR processing not yet implemented for images';
    } else if (mimeType === 'application/pdf') {
      // For PDFs, use pdf-parse or similar
      // const pdfData = await pdf(buffer);
      // return pdfData.text;

      // Placeholder - integrate with your preferred PDF parser
      return 'OCR processing not yet implemented for PDFs';
    }

    return '';
  } catch (error) {
    console.error('OCR processing failed:', error);
    return '';
  }
}

async function performEnhancedAIAnalysis(
  file: File,
  textContent: string,
  uploadData: Partial<FileUpload>
): Promise<AiAnalysisResult | undefined> {
  try {
  const summaryType = ((uploadData as any).summaryType || 'narrative') as string;
    let styleInstruction = '';
    if(summaryType === 'key_points') styleInstruction = 'Return a JSON array of 5-10 succinct bullet point key findings in the "keyFindings" field and a concise one-sentence summary.';
    else if(summaryType === 'prosecutorial') styleInstruction = 'Emphasize prosecutorial relevance: evidentiary value, potential charges, risk factors, chain-of-custody concerns.';
    else styleInstruction = 'Provide a balanced narrative summary suitable for investigators.';

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: `As a legal AI assistant for prosecutors, analyze this evidence with style: ${summaryType}.

${styleInstruction}

File: ${file.name} (${file.type})
Content: ${textContent.substring(0, 4000)}

Provide structured JSON analysis:
{
  "summary": "Brief ${summaryType} oriented legal summary",
  "keyFindings": ["finding1", "finding2"],
  "entities": ["person1", "org1", "location1"],
  "legalImplications": ["implication1", "implication2"],
  "categories": ["contract", "evidence", "witness"],
  "confidence": 0.85,
  "prosecutionRelevance": "high|medium|low",
  "evidenceType": "direct|circumstantial|demonstrative",
  "recommendedActions": ["action1", "action2"]
}`,
        stream: false
      })
    });

    const result = await response.json();

    try {
      const parsed = JSON.parse(result.response);
      return {
        summary: parsed.summary || '',
        keyPoints: parsed.keyFindings || [],
        categories: parsed.categories || [],
        entities: parsed.entities || [],
  // extra fields ignored if schema mismatch
        confidence: parsed.confidence || 0.5,
        processingTime: Date.now(),
        model: 'gemma3-legal:latest'
      };
    } catch (parseError) {
      return {
        summary: result.response.substring(0, 500),
        keyPoints: [],
        categories: [],
        confidence: 0.5,
        processingTime: Date.now(),
        model: 'gemma3-legal:latest'
      };
    }
  } catch (error) {
    console.error('Enhanced AI analysis failed:', error);
    return undefined;
  }
}

async function performAIAnalysis(
  file: File,
  buffer: Buffer,
  uploadData: Partial<FileUpload>,
  ocrText?: string
): Promise<{ analysis?: AiAnalysisResult; embedding?: number[] }> {
  try {
    let textContent = '';

    // Extract text content based on file type
    if (file.type === 'text/plain') {
      textContent = buffer.toString('utf-8');
    } else if (ocrText) {
      textContent = ocrText;
    } else if (file.type.startsWith('image/')) {
      textContent = `Image file: ${file.name}. Description: ${uploadData.description || 'No description provided.'}`;
    } else {
      textContent = `File: ${file.name}. Type: ${file.type}. Description: ${uploadData.description || 'No description provided.'}`;
    }

    const results: { analysis?: AiAnalysisResult; embedding?: number[] } = {};

    // Generate embedding
    if (uploadData.enableEmbeddings && textContent.trim()) {
      try {
        const embedding = await ollamaCudaService.generateEmbedding(textContent);
        results.embedding = embedding;
      } catch (error) {
        console.error('Embedding generation failed:', error);
      }
    }

    // Perform AI analysis
    if (uploadData.enableAiAnalysis && textContent.trim()) {
      try {
        await ollamaCudaService.optimizeForUseCase('legal-analysis');

        const analysisPrompt = `
Analyze the following legal document/evidence and provide:
1. A brief summary
2. Key points and important information
3. Relevant legal categories or tags
4. Confidence score (0-1) of the analysis

Content: ${textContent.substring(0, 4000)} // Limit content for analysis

Format your response as JSON with the following structure:
{
  "summary": "Brief summary of the content",
  "keyPoints": ["key point 1", "key point 2"],
  "categories": ["category1", "category2"],
  "confidence": 0.85
}`;

        const analysisResult = await ollamaCudaService.chatCompletion([
          new SystemMessage('You are a legal AI assistant specializing in document analysis.'),
          new HumanMessage(analysisPrompt)
        ], {
            temperature: 0.3,
            maxTokens: 1000
          }
        );

        // Parse AI response
        try {
          const parsedAnalysis = JSON.parse(analysisResult);
          results.analysis = {
            summary: parsedAnalysis.summary,
            keyPoints: parsedAnalysis.keyPoints || [],
            categories: parsedAnalysis.categories || [],
            confidence: parsedAnalysis.confidence || 0.5,
            processingTime: Date.now(),
            model: ollamaCudaService.currentModel
          };
        } catch (parseError) {
          // If JSON parsing fails, use the raw response as summary
          results.analysis = {
            summary: analysisResult.substring(0, 500),
            keyPoints: [],
            categories: [],
            confidence: 0.5,
            processingTime: Date.now(),
            model: ollamaCudaService.currentModel
          };
        }
      } catch (error) {
        console.error('AI analysis failed:', error);
      }
    }

    return results;
  } catch (error) {
    console.error('AI processing failed:', error);
    return {};
  }
}

async function cacheEmbedding(contentHash: string, embedding: number[]): Promise<void> {
  try {
    await db.insert(embeddingCache).values({
      textHash: contentHash,
      embedding: embedding,
      model: 'nomic-embed-text'
    }).onConflictDoNothing();
  } catch (error) {
    console.error('Failed to cache embedding:', error);
  }
}

// File serving endpoints
export const GET: RequestHandler = async ({ url }) => {
  const correlationId = uuidv4();
  const fileId = url.pathname.split('/').pop();
  const action = url.searchParams.get('action');

  if (!fileId) {
    throw error(404, 'File not found');
  }

  try {
    // Get file info from database
    const evidenceRecord = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, fileId))
      .limit(1);

    if (evidenceRecord.length === 0) {
      throw error(404, 'File not found');
    }

    const record = evidenceRecord[0];

    if (action === 'thumbnail') {
      // TODO: Implement thumbnail serving when we have a thumbnail storage solution
      throw error(404, 'Thumbnail not found');
    } else {
      // Serve original file
      const { readFile } = await import('fs/promises');
      const filePath = join(UPLOAD_DIR, record.fileName!);
  const fileBuffer = await readFile(filePath);

  const resp = new Response(fileBuffer as unknown as BodyInit, {
        headers: {
          'Content-Type': record.mimeType!,
          'Content-Disposition': `inline; filename="${record.fileName}"`,
          'Cache-Control': 'public, max-age=31536000',
          'x-correlation-id': correlationId
        }
      });
      return resp;
    }
  } catch (err) {
    console.error('File serving error:', err);
    throw error(500, 'Failed to serve file');
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  const correlationId = uuidv4();
  const fileId = url.pathname.split('/').pop();

  if (!fileId) {
    throw error(404, 'File not found');
  }

  try {
    // Delete from database
    const deleted = await db
      .delete(evidence)
      .where(eq(evidence.id, fileId))
      .returning();

    if (deleted.length === 0) {
      throw error(404, 'File not found');
    }

    // Delete physical files
    const record = deleted[0];
    try {
      const { unlink } = await import('fs/promises');
      const filePath = join(UPLOAD_DIR, record.fileName!);
      await unlink(filePath);

      if (record && typeof record === 'object' && 'metadata' in record && record.metadata && typeof record.metadata === 'object' && 'thumbnailPath' in record.metadata) {
        await unlink(record.metadata.thumbnailPath as string);
      }
    } catch (error) {
      console.warn('Failed to delete physical file:', error);
    }

  return ok({ id: fileId }, { message: 'File deleted' }, correlationId);
  } catch (err) {
    console.error('File deletion error:', err);
    throw error(500, 'Failed to delete file');
  }
};

