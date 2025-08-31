import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { sql } from 'drizzle-orm';

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const reportId = url.searchParams.get('id');
    const caseId = url.searchParams.get('caseId');
    const type = url.searchParams.get('type');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    
    let query = 'SELECT * FROM reports WHERE user_id = $1';
    const params = [locals.user?.id || 'system'];
    
    if (reportId) {
      query += ' AND id = $2';
      params.push(reportId);
    } else if (caseId) {
      query += ' AND case_id = $2';
      params.push(caseId);
    }
    
    if (type) {
      query += ` AND type = $${params.length + 1}`;
      params.push(type);
    }
    
    query += ' ORDER BY created_at DESC LIMIT $' + (params.length + 1);
    params.push(limit);
    
    const result = await db.execute(sql.raw(query, params));
    
    return json({ success: true, data: result.rows });
  } catch (error) {
    console.error('Error fetching reports:', error);
    return json({ error: 'Failed to fetch reports' }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    const { title, type, content, caseId, format = 'markdown', aiGenerated = false } = body;
    
    const reportId = crypto.randomUUID();
    const userId = locals.user?.id || 'system';
    
    // Generate AI content if requested
    let finalContent = content;
    if (aiGenerated && body.aiPrompt) {
      const aiResponse = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: body.aiPrompt,
          stream: false
        })
      });
      
      if (aiResponse.ok) {
        const result = await aiResponse.json();
        finalContent = result.response;
      }
    }
    
    // Generate embedding
    const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: finalContent.substring(0, 8000)
      })
    });
    
    let embedding = null;
    if (embeddingResponse.ok) {
      const embResult = await embeddingResponse.json();
      embedding = embResult.embedding;
    }
    
    // Save to filesystem
    const filePath = `C:\\LegalAI\\Reports\\${userId}\\${reportId}.${format}`;
    
    // Insert into database
    const result = await db.execute(sql`
      INSERT INTO reports (
        id, user_id, case_id, title, type, content,
        content_embedding, format, ai_generated, ai_model,
        file_path, file_size, created_at, updated_at
      ) VALUES (
        ${reportId}, ${userId}, ${caseId}, ${title}, ${type}, ${finalContent},
        ${embedding ? `[${embedding.join(',')}]::vector` : null}, ${format}, 
        ${aiGenerated}, ${aiGenerated ? 'gemma3-legal' : null},
        ${filePath}, ${Buffer.byteLength(finalContent, 'utf8')},
        NOW(), NOW()
      ) RETURNING *
    `);
    
    return json({ 
      success: true, 
      message: 'Report created successfully',
      data: result.rows[0] 
    });
  } catch (error) {
    console.error('Error creating report:', error);
    return json({ error: 'Failed to create report' }, { status: 500 });
  }
};

export const PUT: RequestHandler = async ({ request, url }) => {
  try {
    const reportId = url.searchParams.get('id');
    if (!reportId) {
      return json({ error: 'Report ID required' }, { status: 400 });
    }
    
    const body = await request.json();
    const { title, content, type } = body;
    
    const result = await db.execute(sql`
      UPDATE reports 
      SET title = ${title}, content = ${content}, type = ${type}, updated_at = NOW()
      WHERE id = ${reportId}
      RETURNING *
    `);
    
    if (result.rows.length === 0) {
      return json({ error: 'Report not found' }, { status: 404 });
    }
    
    return json({ success: true, data: result.rows[0] });
  } catch (error) {
    console.error('Error updating report:', error);
    return json({ error: 'Failed to update report' }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const reportId = url.searchParams.get('id');
    if (!reportId) {
      return json({ error: 'Report ID required' }, { status: 400 });
    }
    
    await db.execute(sql`DELETE FROM reports WHERE id = ${reportId}`);
    
    return json({ success: true, message: 'Report deleted' });
  } catch (error) {
    console.error('Error deleting report:', error);
    return json({ error: 'Failed to delete report' }, { status: 500 });
  }
};