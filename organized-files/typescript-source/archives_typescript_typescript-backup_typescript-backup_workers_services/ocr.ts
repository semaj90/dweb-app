// workers/services/ocr.ts
import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import type { OcrResult } from '../../sveltekit-frontend/src/lib/types/progress';

// MinIO client for file retrieval
import { Client as MinioClient } from 'minio';

const minioClient = new MinioClient({
  endPoint: process.env.MINIO_ENDPOINT || 'localhost',
  port: parseInt(process.env.MINIO_PORT || '9000'),
  useSSL: process.env.MINIO_USE_SSL === 'true',
  accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin'
});

export async function runOcrForEvidence(evidenceId: string): Promise<OcrResult> {
  console.log(`🔍 Starting OCR for evidence: ${evidenceId}`);
  
  try {
    // 1. Download file from MinIO
    const bucketName = process.env.MINIO_EVIDENCE_BUCKET || 'evidence';
    const objectName = `${evidenceId}/original`;
    
    console.log(`📥 Downloading file from MinIO: ${bucketName}/${objectName}`);
    
    const tempDir = path.join(process.cwd(), 'temp');
    await fs.mkdir(tempDir, { recursive: true });
    
    const tempFilePath = path.join(tempDir, `${evidenceId}_original`);
    
    // Download file from MinIO
    await minioClient.fGetObject(bucketName, objectName, tempFilePath);
    
    // 2. Determine file type and run appropriate OCR
    const fileExtension = await getFileExtension(tempFilePath);
    
    let ocrResult: OcrResult;
    
    switch (fileExtension.toLowerCase()) {
      case '.pdf':
        ocrResult = await runPdfOcr(tempFilePath, evidenceId);
        break;
      case '.jpg':
      case '.jpeg':
      case '.png':
      case '.tiff':
      case '.bmp':
        ocrResult = await runImageOcr(tempFilePath, evidenceId);
        break;
      case '.docx':
      case '.doc':
        ocrResult = await runDocumentOcr(tempFilePath, evidenceId);
        break;
      default:
        throw new Error(`Unsupported file type: ${fileExtension}`);
    }
    
    // 3. Cleanup temp files
    await fs.unlink(tempFilePath).catch(console.warn);
    
    // 4. Store OCR result in MinIO for caching
    await storeOcrResult(evidenceId, ocrResult);
    
    console.log(`✅ OCR completed for evidence: ${evidenceId}, text length: ${ocrResult.text.length}`);
    
    return ocrResult;
    
  } catch (error) {
    console.error(`❌ OCR failed for evidence ${evidenceId}:`, error);
    throw error;
  }
}

async function getFileExtension(filePath: string): Promise<string> {
  const stats = await fs.stat(filePath);
  if (!stats.isFile()) {
    throw new Error('Path is not a file');
  }
  
  return path.extname(filePath);
}

async function runPdfOcr(filePath: string, evidenceId: string): Promise<OcrResult> {
  console.log(`📄 Running PDF OCR for: ${evidenceId}`);
  
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    
    // Using Tesseract with pdfimages for PDF processing
    const tesseractProcess = spawn('tesseract', [
      filePath,
      'stdout',
      '-l', 'eng',
      '--psm', '6',
      '--oem', '3',
      'pdf'
    ]);
    
    tesseractProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    tesseractProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    tesseractProcess.on('close', (code) => {
      if (code === 0) {
        resolve({
          text: stdout.trim(),
          confidence: extractConfidenceFromStderr(stderr),
          metadata: {
            method: 'tesseract-pdf',
            pages: extractPageCount(stderr),
            processingTime: Date.now()
          }
        });
      } else {
        reject(new Error(`Tesseract failed with code ${code}: ${stderr}`));
      }
    });
    
    tesseractProcess.on('error', (error) => {
      reject(error);
    });
  });
}

async function runImageOcr(filePath: string, evidenceId: string): Promise<OcrResult> {
  console.log(`🖼️ Running image OCR for: ${evidenceId}`);
  
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    
    // Using Tesseract for image OCR
    const tesseractProcess = spawn('tesseract', [
      filePath,
      'stdout',
      '-l', 'eng',
      '--psm', '6',
      '--oem', '3'
    ]);
    
    tesseractProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    tesseractProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    tesseractProcess.on('close', (code) => {
      if (code === 0) {
        resolve({
          text: stdout.trim(),
          confidence: extractConfidenceFromStderr(stderr),
          metadata: {
            method: 'tesseract-image',
            imageFormat: path.extname(filePath),
            processingTime: Date.now()
          }
        });
      } else {
        reject(new Error(`Tesseract failed with code ${code}: ${stderr}`));
      }
    });
    
    tesseractProcess.on('error', (error) => {
      reject(error);
    });
  });
}

async function runDocumentOcr(filePath: string, evidenceId: string): Promise<OcrResult> {
  console.log(`📝 Running document OCR for: ${evidenceId}`);
  
  try {
    // For DOCX files, we can extract text directly without OCR
    if (path.extname(filePath).toLowerCase() === '.docx') {
      const mammoth = await import('mammoth');
      const result = await mammoth.extractRawText({ path: filePath });
      
      return {
        text: result.value,
        confidence: 1.0, // Direct text extraction is 100% confident
        metadata: {
          method: 'mammoth-docx',
          warnings: result.messages.map(m => m.message),
          processingTime: Date.now()
        }
      };
    }
    
    // For other document types, convert to PDF first then OCR
    // This would require LibreOffice or similar converter
    throw new Error('Document conversion not implemented yet');
    
  } catch (error) {
    console.error('❌ Document OCR failed:', error);
    throw error;
  }
}

function extractConfidenceFromStderr(stderr: string): number {
  // Tesseract outputs confidence information in stderr
  // Parse it to get an average confidence score
  const confidenceMatch = stderr.match(/confidence:\s*([\d.]+)/i);
  if (confidenceMatch) {
    return parseFloat(confidenceMatch[1]) / 100;
  }
  
  // Default confidence if we can't parse it
  return 0.85;
}

function extractPageCount(stderr: string): number {
  const pageMatch = stderr.match(/Page\s*(\d+)/i);
  if (pageMatch) {
    return parseInt(pageMatch[1]);
  }
  return 1;
}

async function storeOcrResult(evidenceId: string, result: OcrResult): Promise<void> {
  try {
    const bucketName = process.env.MINIO_EVIDENCE_BUCKET || 'evidence';
    const objectName = `${evidenceId}/ocr.json`;
    
    const resultJson = JSON.stringify(result, null, 2);
    
    await minioClient.putObject(
      bucketName,
      objectName,
      resultJson,
      {
        'Content-Type': 'application/json',
        'x-amz-meta-generated': new Date().toISOString(),
        'x-amz-meta-method': result.metadata?.method || 'unknown'
      }
    );
    
    console.log(`💾 Stored OCR result for evidence: ${evidenceId}`);
    
  } catch (error) {
    console.warn('⚠️ Failed to store OCR result:', error);
    // Don't throw - this is just caching
  }
}

// Health check for OCR service
export async function checkOcrHealth(): Promise<boolean> {
  try {
    // Test Tesseract availability
    return new Promise((resolve) => {
      const testProcess = spawn('tesseract', ['--version']);
      
      testProcess.on('close', (code) => {
        resolve(code === 0);
      });
      
      testProcess.on('error', () => {
        resolve(false);
      });
      
      // Timeout after 5 seconds
      setTimeout(() => resolve(false), 5000);
    });
    
  } catch (error) {
    return false;
  }
}

// Get OCR capabilities
export function getOcrCapabilities(): string[] {
  return [
    'pdf',
    'jpg', 'jpeg', 'png', 'tiff', 'bmp',
    'docx' // Limited support
  ];
}
