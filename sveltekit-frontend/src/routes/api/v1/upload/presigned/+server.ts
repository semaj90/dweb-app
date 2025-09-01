import { json } from '@sveltejs/kit';
import { z } from 'zod';
import { Client } from 'minio';
import { db } from '$lib/db/client';
import { documents, cases } from '$lib/db/schema/rag-integration';
import { eq } from 'drizzle-orm';
import { randomUUID } from 'crypto';
import type { RequestHandler } from './$types.js';
import { URL } from "url";


const presignedRequestSchema = z.object({
  filename: z.string().min(1).max(255),
  contentType: z.string().min(1).max(100),
  caseId: z.string().uuid()
});

// Initialize MinIO client
const minioClient = new Client({
  endPoint: 'localhost',
  port: 9000,
  useSSL: false,
  accessKey: import.meta.env.MINIO_ACCESS_KEY || 'minioadmin',
  secretKey: import.meta.env.MINIO_SECRET_KEY || 'minioadmin'
});

const BUCKET_NAME = 'legal-documents';
const UPLOAD_EXPIRY = 60 * 60; // 1 hour

export async function POST({ request }): Promise<any> {
  try {
    // Parse and validate request
    const body = await request.json();
    const { filename, contentType, caseId } = presignedRequestSchema.parse(body);

    // Verify case exists
    const [caseRecord] = await db
      .select()
      .from(cases)
      .where(eq(cases.uuid, caseId))
      .limit(1);

    if (!caseRecord) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    // Generate unique file ID and path
    const fileId = randomUUID();
    const fileExtension = filename.split('.').pop() || '';
    const uniqueFilename = `${fileId}.${fileExtension}`;
    const minioPath = `cases/${caseId}/documents/${uniqueFilename}`;

    // Ensure bucket exists
    try {
      const bucketExists = await minioClient.bucketExists(BUCKET_NAME);
      if (!bucketExists) {
        await minioClient.makeBucket(BUCKET_NAME, 'us-east-1');
        
        // Set bucket policy to allow uploads
        const policy = {
          Version: '2012-10-17',
          Statement: [
            {
              Effect: 'Allow',
              Principal: { AWS: ['*'] },
              Action: ['s3:GetObject'],
              Resource: [`arn:aws:s3:::${BUCKET_NAME}/*`]
            },
            {
              Effect: 'Allow',
              Principal: { AWS: ['*'] },
              Action: ['s3:PutObject'],
              Resource: [`arn:aws:s3:::${BUCKET_NAME}/*`]
            }
          ]
        };
        await minioClient.setBucketPolicy(BUCKET_NAME, JSON.stringify(policy));
      }
    } catch (bucketError) {
      console.error('MinIO bucket setup error:', bucketError);
      return json({ error: 'Storage initialization failed' }, { status: 500 });
    }

    // Generate pre-signed URL for upload
    const presignedUrl = await minioClient.presignedPutObject(
      BUCKET_NAME,
      minioPath,
      UPLOAD_EXPIRY,
      {
        'Content-Type': contentType,
        'x-amz-meta-original-name': filename,
        'x-amz-meta-file-id': fileId,
        'x-amz-meta-case-id': caseId
      }
    );

    // Create document record in database
    const [document] = await db
      .insert(documents)
      .values({
        uuid: fileId,
        caseId: caseRecord.id,
        filename: uniqueFilename,
        originalName: filename,
        contentType,
        fileSize: 0, // Will be updated after upload
        minioPath,
        processingStatus: 'pending',
        metadata: {
          uploadedAt: new Date().toISOString(),
          uploadMethod: 'presigned'
        }
      })
      .returning();

    return json({
      fileId,
      uploadUrl: presignedUrl,
      expiresIn: UPLOAD_EXPIRY,
      document: {
        id: document.id,
        uuid: document.uuid,
        filename: document.filename,
        originalName: document.originalName,
        status: document.processingStatus
      }
    });

  } catch (error: any) {
    console.error('Presigned URL generation error:', error);
    
    if (error instanceof z.ZodError) {
      return json({ error: 'Invalid request data', details: error.errors }, { status: 400 });
    }
    
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Optional: GET method to check upload status
export async function GET({ url }): Promise<any> {
  const fileId = url.searchParams.get('fileId');
  
  if (!fileId) {
    return json({ error: 'File ID required' }, { status: 400 });
  }

  try {
    const [document] = await db
      .select()
      .from(documents)
      .where(eq(documents.uuid, fileId))
      .limit(1);

    if (!document) {
      return json({ error: 'Document not found' }, { status: 404 });
    }

    // Check if file exists in MinIO
    let fileExists = false;
    let fileSize = 0;
    
    try {
      const stat = await minioClient.statObject(BUCKET_NAME, document.minioPath);
      fileExists = true;
      fileSize = stat.size;
      
      // Update file size in database if it changed
      if (document.fileSize !== fileSize) {
        await db
          .update(documents)
          .set({ fileSize })
          .where(eq(documents.id, document.id));
      }
    } catch (statError) {
      // File doesn't exist yet or access error
      console.warn(`File ${document.minioPath} not accessible:`, statError.message);
    }

    return json({
      document: {
        id: document.id,
        uuid: document.uuid,
        filename: document.filename,
        originalName: document.originalName,
        status: document.processingStatus,
        fileSize,
        fileExists,
        uploadedAt: document.createdAt
      }
    });

  } catch (error: any) {
    console.error('Upload status check error:', error);
    return json({ error: 'Internal server error' }, { status: 500 });
  }
}