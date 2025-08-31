import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/server/database';
import { sql } from 'drizzle-orm';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { existsSync } from 'fs';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const caseId = formData.get('caseId') as string;
    const title = formData.get('title') as string;
    const description = formData.get('description') as string;
    
    if (!file) {
      return json({ error: 'No file provided' }, { status: 400 });
    }

    const userId = locals.user?.id || 'system';
    const evidenceNumber = `EV-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Create upload directory
    const uploadDir = join('C:\\LegalAI\\Evidence', userId);
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }
    
    // Save file
    const fileName = `${evidenceNumber}_${file.name}`;
    const filePath = join(uploadDir, fileName);
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);
    
    // Calculate hash
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const fileHash = Array.from(new Uint8Array(hashBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');

    // Insert into database
    const result = await db.execute(sql`
      INSERT INTO evidence (
        id, evidence_number, title, description, 
        file_name, file_size, mime_type, file_path, file_hash,
        case_id, user_id, type, processing_status,
        created_at, updated_at
      ) VALUES (
        ${crypto.randomUUID()}, ${evidenceNumber}, ${title}, ${description},
        ${file.name}, ${file.size}, ${file.type}, ${filePath}, ${fileHash},
        ${caseId}, ${userId}, 'document', 'pending',
        NOW(), NOW()
      ) RETURNING *
    `);

    // Queue for processing
    queueProcessing(result[0].id, filePath);

    return json({ 
      success: true, 
      message: 'Evidence uploaded successfully',
      data: result[0] 
    });
  } catch (error) {
    console.error('Error uploading evidence:', error);
    return json({ error: 'Failed to upload evidence' }, { status: 500 });
  }
};

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const evidenceId = url.searchParams.get('id');
    const caseId = url.searchParams.get('caseId');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    const userId = locals.user?.id || 'system';
    
    if (evidenceId) {
      const result = await db.execute(sql`
        SELECT * FROM evidence 
        WHERE id = ${evidenceId} AND user_id = ${userId}
        LIMIT 1
      `);
      
      if (result.length === 0) {
        return json({ error: 'Evidence not found' }, { status: 404 });
      }
      
      return json({ success: true, data: result[0] });
    }
    
    if (caseId) {
      const result = await db.execute(sql`
        SELECT * FROM evidence 
        WHERE case_id = ${caseId} AND user_id = ${userId}
        ORDER BY created_at DESC
        LIMIT ${limit}
      `);
      
      return json({ success: true, data: result });
    }
    
    const result = await db.execute(sql`
      SELECT * FROM evidence 
      WHERE user_id = ${userId}
      ORDER BY created_at DESC
      LIMIT ${limit}
    `);
    
    return json({ success: true, data: result });
  } catch (error) {
    console.error('Error fetching evidence:', error);
    return json({ error: 'Failed to fetch evidence' }, { status: 500 });
  }
};

async function queueProcessing(evidenceId: string, filePath: string) {
  try {
    await fetch('http://localhost:8094/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evidenceId,
        filePath,
        operations: ['ocr', 'auto-tag', 'embedding']
      })
    });
  } catch (err) {
    console.error('Failed to queue processing:', err);
  }
}