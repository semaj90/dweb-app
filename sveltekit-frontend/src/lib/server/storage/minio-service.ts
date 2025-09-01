/**
 * MinIO File Storage Service Integration  
 * Production-ready object storage for documents and media files
 */

import { Client as MinIOClient, type BucketItem, type ItemBucketMetadata } from 'minio';
import { extname } from 'path';
// Dynamic import for crypto to prevent browser leakage
// import { randomUUID } from 'crypto';

// MinIO configuration - following Redis client pattern for authentication
const MINIO_CONFIG = {
  endPoint: import.meta.env.MINIO_ENDPOINT || import.meta.env.MINIO_HOST || 'localhost',
  port: parseInt(import.meta.env.MINIO_PORT || '9000'),
  useSSL: import.meta.env.MINIO_USE_SSL === 'true' || false, // Keep SSL off for local development
  accessKey: import.meta.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: import.meta.env.MINIO_SECRET_KEY || 'minioadmin',
  region: import.meta.env.MINIO_REGION || 'us-east-1',
  // Additional connection options following Redis client patterns
  requestTimeout: 30000, // 30 second timeout
  transportTimeout: 15000, // 15 second transport timeout
  partSize: 64 * 1024 * 1024 // 64MB part size for multipart uploads
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
  private client: InstanceType<typeof MinIOClient>;
  private isInitialized = false;

  constructor() {
    // Initialize MinIO client with robust error handling following Redis patterns
    try {
      this.client = new MinIOClient({
        endPoint: MINIO_CONFIG.endPoint,
        port: MINIO_CONFIG.port,
        useSSL: MINIO_CONFIG.useSSL,
        accessKey: MINIO_CONFIG.accessKey,
        secretKey: MINIO_CONFIG.secretKey,
        region: MINIO_CONFIG.region
      });
      console.log('📦 MinIO client initialized with endpoint:', `${MINIO_CONFIG.useSSL ? 'https' : 'http'}://${MINIO_CONFIG.endPoint}:${MINIO_CONFIG.port}`);
    } catch (error: any) {
      console.warn('⚠️ MinIO client initialization failed - file storage will be disabled:', error.message);
      // Don't throw error - make MinIO optional
      this.client = null as any;
    }
  }

  static getInstance(): MinIOService {
    if (!MinIOService.instance) {
      MinIOService.instance = new MinIOService();
    }
    return MinIOService.instance;
  }

  async initialize(): Promise<boolean> {
    try {
      // Test connection with authentication validation (similar to Redis client pattern)
      console.log('🔄 Testing MinIO connection and authentication...');
      
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('MinIO connection timeout after 5 seconds')), 5000);
      });
      
      // Test basic connectivity and credentials
      const buckets = await Promise.race([this.client.listBuckets(), timeoutPromise]);
      const bucketsArray = Array.isArray(buckets) ? buckets : [];
      console.log('✅ MinIO authentication successful, found', bucketsArray.length, 'buckets');
      
      await this.createBuckets();
      this.isInitialized = true;
      console.log('✅ MinIO service fully initialized with', bucketsArray.length, 'buckets');
      return true;
    } catch (error: any) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Enhanced error diagnosis
      if (errorMessage.includes('signature') || errorMessage.includes('InvalidAccessKeyId')) {
        console.error('🔐 MinIO authentication failed - check ACCESS_KEY and SECRET_KEY');
        console.error('   Current config: accessKey="' + MINIO_CONFIG.accessKey + '", endpoint=' + MINIO_CONFIG.endPoint + ':' + MINIO_CONFIG.port);
      } else if (errorMessage.includes('timeout') || errorMessage.includes('ECONNREFUSED')) {
        console.error('🌐 MinIO connection failed - check if MinIO server is running on', MINIO_CONFIG.endPoint + ':' + MINIO_CONFIG.port);
      } else {
        console.error('❌ MinIO initialization failed:', errorMessage);
      }
      
      console.warn('⚠️ MinIO not available - file storage disabled');
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
      } catch (error: any) {
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
      // Generate fileId using Date.now() and random number to avoid crypto issues
      const fileId = `${Date.now()}-${Math.random().toString(36).substring(2)}`;
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

    } catch (error: any) {
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
    } catch (error: any) {
      console.error('URL generation error:', error);
      return '';
    }
  }

  async deleteFile(bucket: string, fileName: string): Promise<boolean> {
    try {
      await this.client.removeObject(bucket, fileName);
      return true;
    } catch (error: any) {
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
      
    } catch (error: any) {
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