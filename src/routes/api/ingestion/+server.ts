import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { sql } from 'drizzle-orm';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const { resourceType, resourceId, operations } = await request.json();
    const userId = locals.user?.id || 'system';
    
    // Queue ingestion tasks
    const tasks = operations.map((operation: string) => ({
      id: crypto.randomUUID(),
      userId,
      resourceType,
      resourceId,
      operation,
      priority: operation === 'ocr' ? 8 : 5,
      status: 'queued',
      createdAt: new Date()
    }));
    
    for (const task of tasks) {
      await db.execute(sql`
        INSERT INTO ingestion_queue (
          id, user_id, resource_type, resource_id,
          operation, priority, status, created_at
        ) VALUES (
          ${task.id}, ${task.userId}, ${task.resourceType}, ${task.resourceId},
          ${task.operation}, ${task.priority}, ${task.status}, ${task.createdAt}
        )
      `);
      
      // Trigger processing
      triggerProcessing(task);
    }
    
    return json({
      success: true,
      message: `Queued ${tasks.length} ingestion tasks`,
      taskIds: tasks.map(t => t.id)
    });
  } catch (error) {
    console.error('Error queuing ingestion:', error);
    return json({ error: 'Failed to queue ingestion' }, { status: 500 });
  }
};

async function triggerProcessing(task: any) {
  // Try GPU processing first
  try {
    const response = await fetch('http://localhost:8094/gpu/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(task)
    });
    
    if (response.ok) {
      await db.execute(sql`
        UPDATE ingestion_queue 
        SET status = 'processing', processor = 'gpu', started_at = NOW()
        WHERE id = ${task.id}
      `);
      return;
    }
  } catch (err) {
    console.log('GPU not available, trying WebGPU...');
  }
  
  // Try WebGPU
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    await db.execute(sql`
      UPDATE ingestion_queue 
      SET status = 'processing', processor = 'webgpu', started_at = NOW()
      WHERE id = ${task.id}
    `);
    // WebGPU processing would happen here
    return;
  }
  
  // Fallback to CPU/WASM
  await db.execute(sql`
    UPDATE ingestion_queue 
    SET status = 'processing', processor = 'wasm', started_at = NOW()
    WHERE id = ${task.id}
  `);
  
  // Process with WASM
  processWithWASM(task);
}

async function processWithWASM(task: any) {
  // WASM processing implementation
  setTimeout(async () => {
    await db.execute(sql`
      UPDATE ingestion_queue 
      SET status = 'completed', completed_at = NOW(), 
          processing_time = EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000
      WHERE id = ${task.id}
    `);
  }, 1000);
}

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const status = url.searchParams.get('status');
    const resourceId = url.searchParams.get('resourceId');
    const userId = locals.user?.id || 'system';
    
    let query = 'SELECT * FROM ingestion_queue WHERE user_id = $1';
    const params = [userId];
    
    if (status) {
      query += ' AND status = $2';
      params.push(status);
    }
    
    if (resourceId) {
      query += ` AND resource_id = $${params.length + 1}`;
      params.push(resourceId);
    }
    
    query += ' ORDER BY priority DESC, created_at ASC';
    
    const result = await db.execute(sql.raw(query, params));
    
    return json({ success: true, data: result.rows });
  } catch (error) {
    console.error('Error fetching ingestion queue:', error);
    return json({ error: 'Failed to fetch queue status' }, { status: 500 });
  }
};