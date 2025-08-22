import type { RequestHandler } from '@sveltejs/kit';
import { json, error } from "@sveltejs/kit";
import { writeFile, mkdir } from 'fs/promises';
import { randomUUID } from 'crypto';
import { join } from 'path';
import { existsSync } from 'fs';

// Document upload API endpoint
const config = {
  uploadDir: './uploads/documents',
  maxFileSize: 10 * 1024 * 1024, // 10MB
  allowedTypes: [
    'application/pdf',
    'text/plain',
    'text/markdown',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'image/jpeg',
    'image/png',
    'image/gif'
  ]
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const caseId = formData.get('caseId') as string;
    const userId = formData.get('userId') as string;
    const title = formData.get('title') as string || file.name;

    // Validation
    if (!file) {
      throw error(400, 'No file provided');
    }

    if (!caseId || !userId) {
      throw error(400, 'Missing required fields: caseId, userId');
    }

    if (file.size > config.maxFileSize) {
      throw error(400, `File too large. Max size: ${config.maxFileSize / (1024 * 1024)}MB`);
    }

    if (!config.allowedTypes.includes(file.type)) {
      throw error(400, `Unsupported file type: ${file.type}`);
    }

    // Create upload directory if it doesn't exist
    if (!existsSync(config.uploadDir)) {
      await mkdir(config.uploadDir, { recursive: true });
    }

    // Generate unique filename
    const fileId = randomUUID();
    const extension = file.name.split('.').pop();
    const filename = `${fileId}.${extension}`;
    const filePath = join(config.uploadDir, filename);

    // Save file
    const buffer = await file.arrayBuffer();
    await writeFile(filePath, new Uint8Array(buffer));

    console.log(`📁 Uploaded document: ${filename}`);

    return json({
      success: true,
      document: {
        id: fileId,
        filename: file.name,
        filePath,
        fileSize: file.size,
        mimeType: file.type,
        caseId,
        userId,
        title,
        uploadedAt: new Date().toISOString()
      },
      message: 'Document uploaded successfully'
    });

  } catch (err) {
    console.error('❌ Upload error:', err);
    if (err instanceof Error) {
      throw error(500, `Upload failed: ${err.message}`);
    }
    throw error(500, 'Unknown upload error');
  }
};

export const GET: RequestHandler = async () => {
  return json({
    status: 'healthy',
    config: {
      uploadDir: config.uploadDir,
      maxFileSize: `${config.maxFileSize / (1024 * 1024)}MB`,
      allowedTypes: config.allowedTypes
    }
  });
};