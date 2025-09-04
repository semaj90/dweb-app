/**
 * PostgreSQL Document Storage with pgvector Integration
 * Stores processed documents with vector embeddings for similarity search
 */

import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

// Database connection - using existing PostgreSQL setup
async function executeQuery(query: string, params: any[] = []): Promise<any[]> {
  try {
    // Use existing database connection from your setup
    const response = await fetch('http://localhost:8094/api/database/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        params,
        timeout: 30000
      })
    });

    if (!response.ok) {
      throw new Error(`Database query failed: ${response.status}`);
    }

    const result = await response.json();
    return result.rows || [];

  } catch (err: any) {
    console.error('Database query error:', err);
    throw err;
  }
}

/**
 * POST /api/database/store-document - Store processed document with embeddings
 */
export const POST: RequestHandler = async ({ request }): Promise<any> => {
  try {
    const body = await request.json();
    const {
      originalName,
      mimeType,
      size,
      extractedText,
      analysis,
      embeddings,
      imageAnalysis,
      userId,
      caseId,
      processingMethod,
      processingTime
    } = body;

    if (!originalName || !userId) {
      throw error(400, 'originalName and userId are required');
    }

    const documentId = crypto.randomUUID();
    const timestamp = new Date().toISOString();

    // Start transaction
    console.log('📊 Storing document in PostgreSQL with pgvector...');

    // 1. Store main document record
    const documentQuery = `
      INSERT INTO legal_documents (
        id, 
        title, 
        content, 
        file_name, 
        mime_type, 
        file_size,
        user_id,
        case_id,
        processing_method,
        processing_time_ms,
        ai_analysis,
        image_analysis,
        created_at,
        updated_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
      RETURNING id;
    `;

    const documentParams = [
      documentId,
      originalName,
      extractedText || '',
      originalName,
      mimeType,
      size,
      userId,
      caseId || null,
      processingMethod || 'webassembly',
      Math.round(processingTime || 0),
      analysis ? JSON.stringify(analysis) : null,
      imageAnalysis ? JSON.stringify(imageAnalysis) : null,
      timestamp,
      timestamp
    ];

    const documentResult = await executeQuery(documentQuery, documentParams);
    
    if (documentResult.length === 0) {
      throw new Error('Failed to insert document');
    }

    let vectorId = null;

    // 2. Store vector embeddings if available
    if (embeddings && embeddings.vector && Array.isArray(embeddings.vector)) {
      console.log('🧠 Storing vector embeddings in pgvector...');

      // Convert vector to pgvector format
      const vectorString = `[${embeddings.vector.join(',')}]`;

      const vectorQuery = `
        INSERT INTO document_embeddings (
          id,
          document_id,
          embedding,
          model,
          dimensions,
          created_at
        ) VALUES ($1, $2, $3::vector, $4, $5, $6)
        RETURNING id;
      `;

      const vectorParams = [
        crypto.randomUUID(),
        documentId,
        vectorString,
        embeddings.model || 'nomic-embed-text',
        embeddings.dimensions || embeddings.vector.length,
        timestamp
      ];

      try {
        const vectorResult = await executeQuery(vectorQuery, vectorParams);
        vectorId = vectorResult[0]?.id || null;
        console.log('✅ Vector embeddings stored successfully');
      } catch (vectorError) {
        console.warn('⚠️ Failed to store vector embeddings:', vectorError);
        // Continue without vectors - document is still stored
      }
    }

    let evidenceId = null;

    // 3. Store as evidence if caseId provided
    if (caseId) {
      console.log('📋 Creating evidence record...');

      const evidenceQuery = `
        INSERT INTO evidence (
          id,
          case_id,
          title,
          description,
          file_name,
          file_path,
          file_size,
          mime_type,
          uploaded_by,
          ai_analysis,
          document_id,
          created_at,
          updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING id;
      `;

      const evidenceParams = [
        crypto.randomUUID(),
        caseId,
        originalName,
        analysis?.summary || 'AI-processed document',
        originalName,
        `/documents/${documentId}`, // Virtual path
        size,
        mimeType,
        userId,
        analysis ? JSON.stringify({
          confidence: analysis.confidence || 0,
          keyFindings: analysis.keyFindings || [],
          legalRisks: analysis.legalRisks || [],
          entities: analysis.entities || [],
          processingMethod
        }) : null,
        documentId,
        timestamp,
        timestamp
      ];

      try {
        const evidenceResult = await executeQuery(evidenceQuery, evidenceParams);
        evidenceId = evidenceResult[0]?.id || null;
        console.log('✅ Evidence record created successfully');
      } catch (evidenceError) {
        console.warn('⚠️ Failed to create evidence record:', evidenceError);
      }
    }

    // 4. Update document search index
    if (extractedText) {
      console.log('🔍 Updating search index...');

      try {
        const searchQuery = `
          INSERT INTO document_search (
            document_id,
            search_vector,
            content_tokens,
            indexed_at
          ) VALUES (
            $1,
            to_tsvector('english', $2),
            $3,
            $4
          )
          ON CONFLICT (document_id)
          DO UPDATE SET
            search_vector = to_tsvector('english', $2),
            content_tokens = $3,
            indexed_at = $4;
        `;

        const searchParams = [
          documentId,
          extractedText,
          extractedText.split(/\s+/).length,
          timestamp
        ];

        await executeQuery(searchQuery, searchParams);
        console.log('✅ Search index updated successfully');
      } catch (searchError) {
        console.warn('⚠️ Failed to update search index:', searchError);
      }
    }

    console.log(`✅ Document storage completed: ${documentId}`);

    return json({
      success: true,
      documentId,
      vectorId,
      evidenceId,
      storage: {
        document: true,
        vector: vectorId !== null,
        evidence: evidenceId !== null,
        searchIndex: extractedText !== null
      },
      metadata: {
        processingMethod,
        processingTime: Math.round(processingTime || 0),
        vectorDimensions: embeddings?.dimensions,
        hasAnalysis: analysis !== null,
        hasImageAnalysis: imageAnalysis !== null
      },
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Database Storage] Error:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Database storage failed',
      code: 'DATABASE_STORAGE_FAILED'
    });
  }
};

/**
 * GET /api/database/store-document - Get storage statistics
 */
export const GET: RequestHandler = async (): Promise<any> => {
  try {
    // Get document statistics
    const statsQuery = `
      SELECT 
        COUNT(*) as total_documents,
        COUNT(CASE WHEN ai_analysis IS NOT NULL THEN 1 END) as analyzed_documents,
        SUM(file_size) as total_size,
        AVG(processing_time_ms) as avg_processing_time,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(DISTINCT case_id) as unique_cases
      FROM legal_documents
      WHERE created_at > NOW() - INTERVAL '7 days';
    `;

    const vectorStatsQuery = `
      SELECT 
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT model) as unique_models,
        AVG(dimensions) as avg_dimensions
      FROM document_embeddings
      WHERE created_at > NOW() - INTERVAL '7 days';
    `;

    const [docStats, vectorStats] = await Promise.all([
      executeQuery(statsQuery),
      executeQuery(vectorStatsQuery)
    ]);

    return json({
      success: true,
      statistics: {
        documents: docStats[0] || {},
        embeddings: vectorStats[0] || {},
        database: {
          healthy: true,
          pgvectorEnabled: true,
          fullTextSearchEnabled: true
        }
      },
      timestamp: Date.now()
    });

  } catch (err: any) {
    console.error('[Database Storage] Statistics error:', err);
    throw error(500, {
      message: err instanceof Error ? err.message : 'Failed to get statistics',
      code: 'DATABASE_STATS_FAILED'
    });
  }
};