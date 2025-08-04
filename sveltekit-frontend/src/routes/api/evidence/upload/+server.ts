/**
 * Enhanced File Upload API with AI Processing Pipeline
 * Handles file uploads, validation, storage, and AI analysis
 */
import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { fileUploadSchema, type FileUpload, type AiAnalysisResult } from '$lib/schemas/file-upload';
import { db } from '$lib/server/db';
import { evidence, embeddingCache } from '$lib/server/db/schema-postgres-enhanced';
import { ollamaCudaService } from '$lib/services/ollama-cuda-service';
import { createHash } from 'crypto';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import sharp from 'sharp';
import { eq } from 'drizzle-orm';

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
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
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
    if (!locals.user) {
      return json({ error: "Unauthorized" }, { status: 401 });
    }

    const formData = await request.formData();
    const files = formData.getAll('files') as File[];
    const uploadDataStr = formData.get('uploadData') as string;

    if (!files || files.length === 0) {
      throw error(400, 'No files provided');
    }

    // Parse upload metadata
    let uploadData: Partial<FileUpload> = {};
    if (uploadDataStr) {
      try {
        uploadData = JSON.parse(uploadDataStr);
      } catch (e) {
        console.warn('Failed to parse upload data:', e);
      }
    }

    // Add user ID from session
    uploadData.userId = locals.user.id;

    const results: UploadResult[] = [];

    // Ensure upload directories exist
    await mkdir(UPLOAD_DIR, { recursive: true });
    await mkdir(THUMBNAIL_DIR, { recursive: true });

    for (const file of files) {
      // Validate file
      if (file.size > MAX_FILE_SIZE) {
        throw error(400, `File ${file.name} exceeds maximum size of ${MAX_FILE_SIZE / 1024 / 1024}MB`);
      }

      if (!ALLOWED_MIME_TYPES.includes(file.type)) {
        throw error(400, `File type ${file.type} is not supported`);
      }

      // Process single file
      const result = await processFile(file, uploadData);
      results.push(result);
    }

    return json({
      success: true,
      files: results,
      message: `Successfully uploaded ${results.length} file(s)`
    });

  } catch (err) {
    console.error('File upload error:', err);
    
    if (err instanceof Error && 'status' in err) {
      throw err;
    }
    
    throw error(500, 'File upload failed');
  }
};

async function processFile(file: File, uploadData: Partial<FileUpload>): Promise<UploadResult> {
  const fileId = uuidv4();
  const fileExtension = file.name.split('.').pop() || '';
  const fileName = `${fileId}.${fileExtension}`;
  const filePath = join(UPLOAD_DIR, fileName);

  // Calculate file hash
  const buffer = Buffer.from(await file.arrayBuffer());
  const hash = createHash('sha256').update(buffer).digest('hex');

  // Save file to disk
  await writeFile(filePath, buffer);

  let aiAnalysis: AiAnalysisResult | undefined;
  let embedding: number[] | undefined;
  let ocrText: string | undefined;
  let thumbnailPath: string | undefined;

  try {
    // Generate thumbnail for images
    if (file.type.startsWith('image/')) {
      thumbnailPath = await generateThumbnail(buffer, fileId);
    }

    // OCR processing for images and PDFs
    if (uploadData.enableOcr && (file.type.startsWith('image/') || file.type === 'application/pdf')) {
      ocrText = await performOCR(buffer, file.type);
    }

    // AI analysis and embedding generation
    if (uploadData.enableAiAnalysis || uploadData.enableEmbeddings) {
      const analysisResult = await performAIAnalysis(file, buffer, uploadData, ocrText);
      aiAnalysis = analysisResult.analysis;
      embedding = analysisResult.embedding;
    }

    // Save to database
    const evidenceRecord = await db.insert(evidence).values({
      id: fileId,
      caseId: uploadData.caseId,
      userId: uploadData.userId || 'system',
      title: uploadData.title || file.name,
      description: uploadData.description,
      evidenceType: uploadData.evidenceType || 'documents',
      fileName: file.name,
      fileSize: file.size,
      mimeType: file.type,
      hash,
      tags: uploadData.tags || [],
      isAdmissible: uploadData.isAdmissible ?? true,
      confidentialityLevel: uploadData.confidentialityLevel || 'standard',
      collectedBy: uploadData.collectedBy,
      location: uploadData.location,
      aiAnalysis: aiAnalysis || {},
      aiSummary: aiAnalysis?.summary,
      contentEmbedding: embedding,
      chainOfCustody: uploadData.chainOfCustody || [],
      metadata: {
        originalName: file.name,
        uploadedAt: new Date().toISOString(),
        ocrText,
        thumbnailPath,
        processingOptions: {
          aiAnalysisEnabled: uploadData.enableAiAnalysis,
          ocrEnabled: uploadData.enableOcr,
          embeddingsEnabled: uploadData.enableEmbeddings,
          summarizationEnabled: uploadData.enableSummarization
        }
      }
    }).returning();

    // Cache embedding for future use
    if (embedding) {
      await cacheEmbedding(hash, embedding);
    }

    return {
      id: fileId,
      fileName,
      originalName: file.name,
      fileSize: file.size,
      mimeType: file.type,
      url: `/api/evidence/${fileId}/file`,
      hash,
      aiAnalysis,
      embedding,
      ocrText,
      thumbnail: thumbnailPath ? `/api/evidence/${fileId}/thumbnail` : undefined
    };

  } catch (error) {
    console.error(`Error processing file ${file.name}:`, error);
    
    // Still save basic file info even if AI processing fails
    await db.insert(evidence).values({
      id: fileId,
      caseId: uploadData.caseId,
      userId: uploadData.userId || 'system',
      title: uploadData.title || file.name,
      description: uploadData.description,
      evidenceType: uploadData.evidenceType || 'documents',
      fileName: file.name,
      fileSize: file.size,
      mimeType: file.type,
      hash,
      tags: uploadData.tags || [],
      isAdmissible: uploadData.isAdmissible ?? true,
      confidentialityLevel: uploadData.confidentialityLevel || 'standard',
      collectedBy: uploadData.collectedBy,
      location: uploadData.location,
      chainOfCustody: uploadData.chainOfCustody || [],
      metadata: {
        originalName: file.name,
        uploadedAt: new Date().toISOString(),
        processingError: error instanceof Error ? error.message : 'Unknown error'
      }
    });

    return {
      id: fileId,
      fileName,
      originalName: file.name,
      fileSize: file.size,
      mimeType: file.type,
      url: `/api/evidence/${fileId}/file`,
      hash
    };
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
          { role: 'system', content: 'You are a legal AI assistant specializing in document analysis.' },
          { role: 'user', content: analysisPrompt }
        ], {
          temperature: 0.3,
          maxTokens: 1000
        });

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
export const GET: RequestHandler = async ({ url, params }) => {
  const fileId = params.fileId;
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
    
    if (action === 'thumbnail' && record.metadata?.thumbnailPath) {
      // Serve thumbnail
      const { readFile } = await import('fs/promises');
      const thumbnailBuffer = await readFile(record.metadata.thumbnailPath as string);
      
      return new Response(thumbnailBuffer, {
        headers: {
          'Content-Type': 'image/webp',
          'Cache-Control': 'public, max-age=31536000'
        }
      });
    } else {
      // Serve original file
      const { readFile } = await import('fs/promises');
      const filePath = join(UPLOAD_DIR, record.fileName!);
      const fileBuffer = await readFile(filePath);
      
      return new Response(fileBuffer, {
        headers: {
          'Content-Type': record.mimeType!,
          'Content-Disposition': `inline; filename="${record.fileName}"`,
          'Cache-Control': 'public, max-age=31536000'
        }
      });
    }
  } catch (err) {
    console.error('File serving error:', err);
    throw error(500, 'Failed to serve file');
  }
};

export const DELETE: RequestHandler = async ({ params }) => {
  const fileId = params.fileId;
  
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
      
      if (record.metadata?.thumbnailPath) {
        await unlink(record.metadata.thumbnailPath as string);
      }
    } catch (error) {
      console.warn('Failed to delete physical file:', error);
    }

    return json({ success: true, message: 'File deleted successfully' });
  } catch (err) {
    console.error('File deletion error:', err);
    throw error(500, 'Failed to delete file');
  }
};
