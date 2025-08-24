/**
 * MinIO File Storage Service Integration  
 * Production-ready object storage for documents and media files
 */

import { Client as MinIOClient, type BucketItem, type ItemBucketMetadata } from 'minio';
import { extname } from 'path';
import { randomUUID } from 'crypto';

// MinIO configuration
const MINIO_CONFIG = {
  endPoint: process.env.MINIO_ENDPOINT || 'localhost',
  port: parseInt(process.env.MINIO_PORT || '9000'),
  useSSL: process.env.MINIO_USE_SSL === 'true' || process.env.NODE_ENV === 'production',
  accessKey: process.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: process.env.MINIO_SECRET_KEY || 'minioadmin123',
  region: process.env.MINIO_REGION || 'us-east-1'
};

// Bucket configurations
const BUCKETS = {
  DOCUMENTS: 'legal-documents',
  EVIDENCE: 'evidence-files', 
  IMAGES: 'image-assets',
  THUMBNAILS: 'thumbnails',
  TEMP: 'temp-uploads',
  ARCHIVES: 'archived-files',
  BACKUPS: 'system-backups'
} as const;

export interface FileMetadata {
  originalName: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  fileType: string;
  bucket: string;
  uploadedBy?: number;
  caseId?: number;
  documentId?: number;
  description?: string;
  tags?: string[];
  uploadedAt: Date;
}

export interface UploadResult {
  success: boolean;
  fileId: string;
  fileName: string;
  bucket: string;
  size: number;
  url: string;
  metadata: FileMetadata;
  error?: string;
}

export class MinIOService {
  private static instance: MinIOService;
  private client: MinIOClient;
  private isInitialized = false;

  constructor() {
    this.client = new MinIOClient(MINIO_CONFIG);
  }

  static getInstance(): MinIOService {
    if (!MinIOService.instance) {
      MinIOService.instance = new MinIOService();
    }
    return MinIOService.instance;
  }

  async initialize(): Promise<boolean> {
    try {
      await this.client.listBuckets();
      await this.createBuckets();
      this.isInitialized = true;
      console.log('✅ MinIO service initialized');
      return true;
    } catch (error) {
      console.error('❌ MinIO initialization failed:', error);
      return false;
    }
  }

  private async createBuckets(): Promise<void> {
    for (const [name, bucket] of Object.entries(BUCKETS)) {
      try {
        const exists = await this.client.bucketExists(bucket);
        if (!exists) {
          await this.client.makeBucket(bucket, MINIO_CONFIG.region);
          console.log(`✅ Created bucket: ${bucket}`);
        }
      } catch (error) {
        console.error(`❌ Failed to create bucket ${bucket}:`, error);
      }
    }
  }

  async uploadFile(
    file: File | Buffer,
    originalName: string,
    options: { bucket?: string; caseId?: number; uploadedBy?: number } = {}
  ): Promise<UploadResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const fileExtension = extname(originalName).toLowerCase();
      const bucket = options.bucket || BUCKETS.DOCUMENTS;
      const fileId = randomUUID();
      const fileName = `${fileId}${fileExtension}`;

      let fileBuffer: Buffer;
      if (file instanceof File) {
        fileBuffer = Buffer.from(await file.arrayBuffer());
      } else {
        fileBuffer = file;
      }

      const metadata: FileMetadata = {
        originalName,
        fileName,
        fileSize: fileBuffer.length,
        mimeType: this.getMimeType(fileExtension),
        fileType: this.determineFileType(fileExtension),
        bucket,
        uploadedAt: new Date(),
        uploadedBy: options.uploadedBy,
        caseId: options.caseId
      };

      await this.client.putObject(
        bucket,
        fileName,
        fileBuffer,
        fileBuffer.length,
        {
          'Content-Type': metadata.mimeType,
          'X-Uploaded-By': String(options.uploadedBy || 'system'),
          'X-Case-Id': String(options.caseId || ''),
          'X-Original-Name': originalName
        }
      );

      const url = await this.getFileUrl(bucket, fileName);

      return {
        success: true,
        fileId,
        fileName,
        bucket,
        size: fileBuffer.length,
        url,
        metadata
      };

    } catch (error) {
      console.error('File upload error:', error);
      return {
        success: false,
        fileId: '',
        fileName: '',
        bucket: '',
        size: 0,
        url: '',
        metadata: {} as FileMetadata,
        error: error instanceof Error ? error.message : 'Upload failed'
      };
    }
  }

  async getFileUrl(bucket: string, fileName: string, expires: number = 24 * 60 * 60): Promise<string> {
    try {
      return await this.client.presignedGetObject(bucket, fileName, expires);
    } catch (error) {
      console.error('URL generation error:', error);
      return '';
    }
  }

  async deleteFile(bucket: string, fileName: string): Promise<boolean> {
    try {
      await this.client.removeObject(bucket, fileName);
      return true;
    } catch (error) {
      console.error('File deletion error:', error);
      return false;
    }
  }

  private determineFileType(extension: string): string {
    const documentTypes = ['.pdf', '.doc', '.docx', '.txt', '.rtf'];
    const imageTypes = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
    
    if (documentTypes.includes(extension)) return 'document';
    if (imageTypes.includes(extension)) return 'image';
    return 'other';
  }

  private getMimeType(extension: string): string {
    const mimeTypes: Record<string, string> = {
      '.pdf': 'application/pdf',
      '.doc': 'application/msword',
      '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      '.txt': 'text/plain',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.json': 'application/json'
    };
    
    return mimeTypes[extension] || 'application/octet-stream';
  }

  async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      const buckets = await this.client.listBuckets();
      
      return {
        status: 'healthy',
        details: {
          buckets: buckets.length,
          bucketNames: buckets.map(b => b.name),
          endpoint: MINIO_CONFIG.endPoint,
          initialized: this.isInitialized
        }
      };
      
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error instanceof Error ? error.message : 'Unknown error',
          initialized: this.isInitialized
        }
      };
    }
  }
}

// Singleton instance
export const minioService = MinIOService.getInstance();
export { BUCKETS };