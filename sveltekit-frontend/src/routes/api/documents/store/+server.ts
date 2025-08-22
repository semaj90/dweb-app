import type { RequestHandler } from '@sveltejs/kit';
import type { RequestHandler } from "./$types";
// Document storage API endpoint
import { db, documents, embeddings } from "$lib/server/database";
import { error, json } from "drizzle-orm";
export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('[Storage] Processing document storage request...');

    const body = await request.json();
    const {
      content,
      embedding,
      metadata = {},
      filename,
      originalContent,
      legalAnalysis,
      confidence
    } = body;

    if (!content) {
      throw error(400, 'Content is required');
    }

    console.log(`[Storage] Storing document: ${filename}`);

    // Store document in database
    const documentResult = await db.insert(documents).values({
      filename: filename || 'untitled',
      content,
      originalContent,
      metadata: JSON.stringify({
        ...metadata,
        stored_at: new Date().toISOString()
      }),
      legalAnalysis: legalAnalysis ? JSON.stringify(legalAnalysis) : null,
      confidence: confidence || null
    }).returning();

    const documentId = documentResult[0].id;

    // Store embedding if provided
    if (embedding && embedding.length > 0) {
      await db.insert(embeddings).values({
        documentId,
        content,
        embedding: embedding,
        metadata: {
          ...metadata,
          embedding_model: 'nomic-embed-text',
          stored_at: new Date().toISOString()
        }
      });

      console.log(`[Storage] Embedding stored for document ${documentId}`);
    }

    console.log(`[Storage] Document stored successfully with ID: ${documentId}`);

    return json({
      success: true,
      documentId,
      message: 'Document stored successfully',
      document: {
        id: documentId,
        filename: filename || 'untitled',
        contentLength: content.length,
        hasEmbedding: !!embedding,
        hasLegalAnalysis: !!legalAnalysis
      }
    });

  } catch (err: any) {
    console.error('[Storage] Error:', err);

    return json({
      success: false,
      error: err.message || 'Storage failed'
    }, { status: err.status || 500 });
  }
};

export const GET: RequestHandler = async () => {
  try {
    return json({
      status: 'healthy',
      service: 'Document Storage',
      message: 'POST to store documents'
    });
  } catch (err: any) {
    return json({
      status: 'unhealthy',
      error: err.message
    }, { status: 503 });
  }
};
