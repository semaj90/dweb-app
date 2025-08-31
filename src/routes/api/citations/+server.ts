import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { sql } from 'drizzle-orm';

export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    const citationId = url.searchParams.get('id');
    const caseId = url.searchParams.get('caseId');
    const search = url.searchParams.get('search');
    const limit = parseInt(url.searchParams.get('limit') || '20');
    
    const userId = locals.user?.id || 'system';
    
    // Vector search if query provided
    if (search) {
      // Generate embedding for search
      const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: search
        })
      });
      
      if (embeddingResponse.ok) {
        const embResult = await embeddingResponse.json();
        const embedding = embResult.embedding;
        
        const result = await db.execute(sql`
          SELECT *, 1 - (embedding <-> $1::vector) as similarity
          FROM citations
          WHERE user_id = $2
          ${caseId ? sql`AND case_id = $3` : sql``}
          ORDER BY embedding <-> $1::vector
          LIMIT $4
        `, [
          `[${embedding.join(',')}]`,
          userId,
          ...(caseId ? [caseId] : []),
          limit
        ]);
        
        return json({ success: true, data: result.rows });
      }
    }
    
    // Regular query
    let query = 'SELECT * FROM citations WHERE user_id = $1';
    const params = [userId];
    
    if (citationId) {
      query += ' AND id = $2';
      params.push(citationId);
    } else if (caseId) {
      query += ' AND case_id = $2';
      params.push(caseId);
    }
    
    query += ' ORDER BY created_at DESC LIMIT $' + (params.length + 1);
    params.push(limit);
    
    const result = await db.execute(sql.raw(query, params));
    
    return json({ success: true, data: result.rows });
  } catch (error) {
    console.error('Error fetching citations:', error);
    return json({ error: 'Failed to fetch citations' }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const body = await request.json();
    const {
      caseId,
      reportId,
      caseNumber,
      caseName,
      citation,
      courtName,
      decisionDate,
      judges,
      precedentLevel,
      legalIssues,
      holding,
      reasoning,
      sourceUrl,
      notes
    } = body;
    
    const citationId = crypto.randomUUID();
    const userId = locals.user?.id || 'system';
    
    // Generate embedding
    const text = `${caseName} ${citation} ${holding || ''} ${reasoning || ''}`;
    const embeddingResponse = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text.substring(0, 8000)
      })
    });
    
    let embedding = null;
    if (embeddingResponse.ok) {
      const embResult = await embeddingResponse.json();
      embedding = embResult.embedding;
    }
    
    // Calculate relevance score
    let relevanceScore = 0.5;
    if (caseId && holding) {
      // AI-based relevance calculation
      const aiResponse = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: `Rate the relevance of this citation to the case on a scale of 0 to 1:
            Citation: ${citation}
            Holding: ${holding}
            Return only a decimal number between 0 and 1.`,
          stream: false
        })
      });
      
      if (aiResponse.ok) {
        const result = await aiResponse.json();
        relevanceScore = parseFloat(result.response) || 0.5;
      }
    }
    
    // Insert into database
    const result = await db.execute(sql`
      INSERT INTO citations (
        id, user_id, case_id, report_id,
        case_number, case_name, citation, court_name,
        decision_date, judges, precedent_level,
        legal_issues, holding, reasoning,
        relevance_score, embedding,
        source_url, notes,
        created_at, updated_at
      ) VALUES (
        ${citationId}, ${userId}, ${caseId}, ${reportId},
        ${caseNumber}, ${caseName}, ${citation}, ${courtName},
        ${decisionDate}, ${judges || []}, ${precedentLevel},
        ${legalIssues || []}, ${holding}, ${reasoning},
        ${relevanceScore}, ${embedding ? `[${embedding.join(',')}]::vector` : null},
        ${sourceUrl}, ${notes},
        NOW(), NOW()
      ) RETURNING *
    `);
    
    return json({ 
      success: true, 
      message: 'Citation saved successfully',
      data: result.rows[0] 
    });
  } catch (error) {
    console.error('Error creating citation:', error);
    return json({ error: 'Failed to create citation' }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ url }) => {
  try {
    const citationId = url.searchParams.get('id');
    if (!citationId) {
      return json({ error: 'Citation ID required' }, { status: 400 });
    }
    
    await db.execute(sql`DELETE FROM citations WHERE id = ${citationId}`);
    
    return json({ success: true, message: 'Citation deleted' });
  } catch (error) {
    console.error('Error deleting citation:', error);
    return json({ error: 'Failed to delete citation' }, { status: 500 });
  }
};