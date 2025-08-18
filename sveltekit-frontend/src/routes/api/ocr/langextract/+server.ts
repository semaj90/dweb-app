// Real OCR API endpoint with Tesseract.js and LegalBERT analysis
import { error, json } from '@sveltejs/kit';
import pdfParse from 'pdf-parse';
import sharp from 'sharp';
import { createWorker } from 'tesseract.js';
import type { RequestHandler } from './$types';

// Redis client for caching
import { createRedisConnection } from '$lib/utils/redis-helper';
// ... other imports ...

const redis = createRedisConnection();

// Legal terms and patterns for LegalBERT-style analysis
const LEGAL_ENTITIES = {
  LEGAL_DOCUMENT: [
    'contract',
    'agreement',
    'deed',
    'lease',
    'will',
    'testament',
    'affidavit',
    'motion',
    'brief',
  ],
  LEGAL_PERSON: [
    'plaintiff',
    'defendant',
    'appellant',
    'appellee',
    'petitioner',
    'respondent',
    'grantor',
    'grantee',
  ],
  LEGAL_TERM: [
    'whereas',
    'hereby',
    'thereof',
    'herein',
    'notwithstanding',
    'pursuant',
    'heretofore',
    'hereinafter',
  ],
  COURT: ['supreme court', 'district court', 'circuit court', 'court of appeals', 'magistrate'],
  JURISDICTION: ['federal', 'state', 'local', 'municipal', 'county'],
};

const LEGAL_CONCEPTS = [
  'contract law',
  'tort law',
  'property law',
  'criminal law',
  'constitutional law',
  'civil procedure',
  'evidence',
  'negligence',
  'liability',
  'damages',
  'breach of contract',
  'due process',
  'equal protection',
  'jurisdiction',
];

export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('[OCR] Processing real OCR request...');

    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      throw error(400, 'No file provided');
    }

    const enableLegalBERT = request.headers.get('X-Enable-LegalBERT') === 'true';
    const cacheKey = `ocr:${file.name}:${file.size}:${file.lastModified}`;

    // Check cache first
    try {
      const cached = await redis.get(cacheKey);
      if (cached) {
        console.log(`[OCR] Cache hit for ${file.name}`);
        return json(JSON.parse(cached));
      }
    } catch (err) {
      console.warn('[OCR] Redis cache unavailable:', err);
    }

    console.log(`[OCR] Processing ${file.name}, size: ${file.size}, type: ${file.type}`);

    // Convert file to buffer
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    let extractedText = '';
    let confidence = 0;
    let processingMethod = '';

    try {
      // Handle different file types
      if (file.type === 'application/pdf') {
        // Extract text from PDF
        processingMethod = 'PDF Text Extraction';
        const pdfData = await pdfParse(buffer);
        extractedText = pdfData.text;
        confidence = extractedText.length > 0 ? 0.95 : 0.1;

        console.log(`[OCR] PDF extracted ${extractedText.length} characters`);
      } else if (file.type.startsWith('image/')) {
        // Use Tesseract.js for image OCR
        processingMethod = 'Tesseract.js OCR';

        // Preprocess image with sharp for better OCR accuracy
        let processedBuffer: Buffer = buffer;
        try {
          const sharpResult = await sharp(buffer)
            .greyscale()
            .normalize()
            .sharpen()
            .resize({ width: 2000, height: 2000, fit: 'inside', withoutEnlargement: true })
            .toBuffer();
          processedBuffer = Buffer.from(sharpResult);
        } catch (sharpErr) {
          console.warn('[OCR] Sharp preprocessing failed, using original buffer:', sharpErr);
        }

        // Initialize Tesseract worker
        const worker = await createWorker(); // logger removed (not in types)

        try {
          // @ts-ignore - loadLanguage not declared in current tesseract.js types
          await worker.loadLanguage('eng');
          // @ts-ignore - initialize not declared (reinitialize is typed); runtime API provides initialize
          await worker.initialize('eng');

          // Configure recognition parameters for better accuracy
          await worker.setParameters({
            // Cast page seg mode to any to satisfy type (expects enum PSM)
            tessedit_pageseg_mode: '1' as any, // Automatic page segmentation with OSD
            preserve_interword_spaces: '1',
            tessedit_create_hocr: '1',
          });

          // Perform OCR
          const { data } = await worker.recognize(processedBuffer as any);
          extractedText = data.text;
          confidence = data.confidence / 100; // Convert to 0-1 scale

          console.log(
            `[OCR] Tesseract extracted ${extractedText.length} characters with ${confidence} confidence`
          );
        } finally {
          await worker.terminate();
        }
      } else if (file.type === 'text/plain') {
        // Handle plain text files
        processingMethod = 'Plain Text';
        extractedText = buffer.toString('utf-8');
        confidence = 1.0;
      } else {
        throw new Error(`Unsupported file type: ${file.type}`);
      }

      // Post-process text
      extractedText = cleanExtractedText(extractedText);

      if (!extractedText || extractedText.trim().length === 0) {
        throw new Error('No text could be extracted from the file');
      }

      // Legal analysis using pattern matching (LegalBERT-style)
      let legalAnalysis = null;
      if (enableLegalBERT) {
        console.log('[OCR] Running legal analysis...');
        legalAnalysis = performLegalAnalysis(extractedText);
      }

      const result = {
        success: true,
        text: extractedText,
        originalText: extractedText,
        confidence: confidence,
        processingMethod,
        metadata: {
          filename: file.name,
          filesize: file.size,
          mimetype: file.type,
          processedAt: new Date().toISOString(),
          textLength: extractedText.length,
          wordCount: extractedText.split(/\s+/).length,
        },
        legal: legalAnalysis,
        language: 'en',
      };

      // Cache the result for 1 hour
      try {
        await redis.setex(cacheKey, 3600, JSON.stringify(result));
        console.log('[OCR] Result cached successfully');
      } catch (err) {
        console.warn('[OCR] Failed to cache result:', err);
      }

      console.log('[OCR] Processing completed successfully');
      return json(result);
    } catch (processingError) {
      console.error('[OCR] Processing error:', processingError);
      throw new Error(`OCR processing failed: ${processingError.message}`);
    }
  } catch (err: any) {
    console.error('[OCR] Error:', err);

    return json(
      {
        success: false,
        error: err.message || 'OCR processing failed',
        details: err.stack,
      },
      { status: err.status || 500 }
    );
  }
};

// Clean and normalize extracted text
function cleanExtractedText(text: string): string {
  return (
    text
      // Remove excessive whitespace
      .replace(/\s+/g, ' ')
      // Remove non-printable characters but keep newlines
      .replace(/[^\x20-\x7E\n\r\t]/g, '')
      // Fix common OCR errors
      .replace(/\bthe\b/gi, 'the')
      .replace(/\band\b/gi, 'and')
      .replace(/\bof\b/gi, 'of')
      // Remove excessive line breaks
      .replace(/\n\s*\n\s*\n/g, '\n\n')
      .trim()
  );
}

// Perform legal analysis using pattern matching
function performLegalAnalysis(text: string): any {
  const lowerText = text.toLowerCase();
  const entities: any[] = [];
  const concepts: string[] = [];

  // Extract legal entities
  for (const [type, terms] of Object.entries(LEGAL_ENTITIES)) {
    for (const term of terms) {
      const regex = new RegExp(`\\b${term}\\b`, 'gi');
      const matches = text.match(regex);
      if (matches) {
        entities.push({
          text: term,
          type: type,
          confidence: 0.8 + matches.length * 0.05, // Higher confidence for repeated terms
          count: matches.length,
        });
      }
    }
  }

  // Extract legal concepts
  for (const concept of LEGAL_CONCEPTS) {
    if (lowerText.includes(concept.toLowerCase())) {
      concepts.push(concept);
    }
  }

  // Determine document type based on content
  let documentType = 'general';
  if (lowerText.includes('contract') || lowerText.includes('agreement')) {
    documentType = 'contract';
  } else if (lowerText.includes('deed') || lowerText.includes('property')) {
    documentType = 'deed';
  } else if (lowerText.includes('will') || lowerText.includes('testament')) {
    documentType = 'will';
  } else if (lowerText.includes('motion') || lowerText.includes('court')) {
    documentType = 'motion';
  }

  // Determine jurisdiction
  let jurisdiction = 'unknown';
  if (lowerText.includes('federal') || lowerText.includes('supreme court')) {
    jurisdiction = 'federal';
  } else if (lowerText.includes('state') || lowerText.includes('district court')) {
    jurisdiction = 'state';
  } else if (lowerText.includes('local') || lowerText.includes('municipal')) {
    jurisdiction = 'local';
  }

  return {
    entities: entities.slice(0, 20), // Limit to top 20 entities
    concepts: concepts.slice(0, 10), // Limit to top 10 concepts
    documentType,
    jurisdiction,
    analysisMethod: 'Pattern Matching + Legal Dictionary',
    confidence: entities.length > 0 ? 0.85 : 0.5,
  };
}

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    // Check Redis connection
    let redisStatus = false;
    try {
      const pong = await redis.ping();
      redisStatus = pong === 'PONG';
    } catch (err) {
      console.warn('[OCR] Redis health check failed:', err);
    }

    // Check Tesseract by creating a minimal worker
    let tesseractStatus = false;
    try {
      const worker = await createWorker();
      await worker.terminate();
      tesseractStatus = true;
    } catch (err) {
      console.warn('[OCR] Tesseract health check failed:', err);
    }

    return json({
      status: 'healthy',
      service: 'Real OCR with Tesseract.js',
      features: {
        tesseract: tesseractStatus,
        pdfExtraction: true,
        imagePreprocessing: true,
        legalAnalysis: true,
        redis: redisStatus,
        languages: ['eng'],
        supportedFormats: [
          'image/jpeg',
          'image/png',
          'image/tiff',
          'application/pdf',
          'text/plain',
        ],
      },
      timestamp: new Date().toISOString(),
      version: '2.0.0',
    });
  } catch (err: any) {
    return json(
      {
        status: 'unhealthy',
        error: err.message,
        timestamp: new Date().toISOString(),
      },
      { status: 503 }
    );
  }
};
