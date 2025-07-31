/**
 * Documents API Routes
 * CRUD operations for document management
 */

import express from 'express';
import { z } from 'zod';

const router = express.Router();

// Validation schemas
const documentQuerySchema = z.object({
  caseId: z.string().optional(),
  documentType: z.string().optional(),
  status: z.string().optional(),
  limit: z.number().int().min(1).max(100).optional().default(20),
  offset: z.number().int().min(0).optional().default(0),
  search: z.string().optional()
});

const documentUpdateSchema = z.object({
  title: z.string().min(1).max(255).optional(),
  description: z.string().optional(),
  documentType: z.string().optional(),
  metadata: z.record(z.any()).optional(),
  tags: z.array(z.string()).optional()
});

export function createDocumentRoutes(services, io) {
  const { database, vector, cache, documentProcessor } = services;

  /**
   * GET / - List documents with filtering
   */
  router.get('/', async (req, res) => {
    try {
      const validatedQuery = documentQuerySchema.parse(req.query);
      const { caseId, documentType, status, limit, offset, search } = validatedQuery;

      // Build filters
      const filters = {};
      if (caseId) filters.caseId = caseId;
      if (documentType) filters.documentType = documentType;
      if (status) filters.status = status;
      if (search) filters.search = search;
      filters.limit = limit;
      filters.offset = offset;

      // Check cache
      const cacheKey = `documents:list:${JSON.stringify(filters)}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          documents: cached.documents,
          total: cached.total,
          cached: true,
          pagination: {
            limit,
            offset,
            hasMore: cached.total > offset + limit
          }
        });
      }

      // Get documents from database
      const documents = await database.getDocuments(filters);
      
      // Get total count for pagination
      const totalFilters = { ...filters };
      delete totalFilters.limit;
      delete totalFilters.offset;
      const allDocuments = await database.getDocuments(totalFilters);
      const total = allDocuments.length;

      // Cache results
      await cache.set(cacheKey, { documents, total }, 300); // 5 minutes

      res.json({
        success: true,
        documents: documents.map(doc => ({
          id: doc.id,
          title: doc.title,
          documentType: doc.document_type,
          fileSize: doc.file_size,
          fileType: doc.file_type,
          caseId: doc.case_id,
          processingStatus: doc.processing_status,
          createdAt: doc.created_at,
          updatedAt: doc.updated_at,
          indexedAt: doc.indexed_at,
          metadata: doc.metadata
        })),
        total,
        cached: false,
        pagination: {
          limit,
          offset,
          hasMore: total > offset + limit
        }
      });

    } catch (error) {
      console.error('Document listing failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      res.status(500).json({
        success: false,
        error: 'Document listing failed',
        message: error.message
      });
    }
  });

  /**
   * GET /:id - Get document by ID
   */
  router.get('/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const { includeContent = false, includeChunks = false } = req.query;

      // Check cache
      const cacheKey = `document:${id}:${includeContent}:${includeChunks}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          document: cached,
          cached: true
        });
      }

      // Get document from database
      const document = await database.getDocument(id);
      
      if (!document) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Format response
      const response = {
        id: document.id,
        title: document.title,
        description: document.description,
        documentType: document.document_type,
        fileName: document.file_path ? document.file_path.split('/').pop() : null,
        fileSize: document.file_size,
        fileType: document.file_type,
        caseId: document.case_id,
        processingStatus: document.processing_status,
        createdAt: document.created_at,
        updatedAt: document.updated_at,
        indexedAt: document.indexed_at,
        metadata: document.metadata
      };

      // Include content if requested
      if (includeContent === 'true') {
        response.content = document.content;
      }

      // Include chunks if requested
      if (includeChunks === 'true') {
        const chunks = await database.pool.query(
          'SELECT * FROM rag_chunks WHERE document_id = $1 ORDER BY chunk_index',
          [id]
        );
        response.chunks = chunks.rows.map(chunk => ({
          id: chunk.id,
          chunkIndex: chunk.chunk_index,
          content: chunk.content,
          metadata: chunk.metadata,
          createdAt: chunk.created_at
        }));
      }

      // Cache response
      await cache.set(cacheKey, response, 1800); // 30 minutes

      res.json({
        success: true,
        document: response,
        cached: false
      });

    } catch (error) {
      console.error('Document retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Document retrieval failed',
        message: error.message
      });
    }
  });

  /**
   * PUT /:id - Update document
   */
  router.put('/:id', async (req, res) => {
    try {
      const { id } = req.params;
      const validatedData = documentUpdateSchema.parse(req.body);

      // Check if document exists
      const existing = await database.getDocument(id);
      if (!existing) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Update document in database
      const updateQuery = `
        UPDATE rag_documents 
        SET title = COALESCE($1, title),
            description = COALESCE($2, description),
            document_type = COALESCE($3, document_type),
            metadata = COALESCE($4, metadata),
            updated_at = NOW()
        WHERE id = $5
        RETURNING *
      `;

      const values = [
        validatedData.title,
        validatedData.description,
        validatedData.documentType,
        validatedData.metadata ? JSON.stringify(validatedData.metadata) : null,
        id
      ];

      const result = await database.pool.query(updateQuery, values);
      const updatedDocument = result.rows[0];

      // Clear related caches
      await cache.clearByPattern(`document:${id}:*`);
      await cache.clearByPattern(`documents:list:*`);

      // Emit real-time update
      if (io && updatedDocument.case_id) {
        io.to(`case-${updatedDocument.case_id}`).emit('document-updated', {
          documentId: id,
          title: updatedDocument.title,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        document: {
          id: updatedDocument.id,
          title: updatedDocument.title,
          description: updatedDocument.description,
          documentType: updatedDocument.document_type,
          metadata: updatedDocument.metadata,
          updatedAt: updatedDocument.updated_at
        }
      });

    } catch (error) {
      console.error('Document update failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      res.status(500).json({
        success: false,
        error: 'Document update failed',
        message: error.message
      });
    }
  });

  /**
   * DELETE /:id - Delete document
   */
  router.delete('/:id', async (req, res) => {
    try {
      const { id } = req.params;

      // Check if document exists
      const existing = await database.getDocument(id);
      if (!existing) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Delete document and related chunks (CASCADE)
      await database.pool.query('DELETE FROM rag_documents WHERE id = $1', [id]);

      // Clear related caches
      await cache.clearByPattern(`document:${id}:*`);
      await cache.clearByPattern(`documents:list:*`);
      await cache.clearByPattern(`search:*`); // Clear search caches as document is no longer available

      // Emit real-time update
      if (io && existing.case_id) {
        io.to(`case-${existing.case_id}`).emit('document-deleted', {
          documentId: id,
          title: existing.title,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        message: 'Document deleted successfully'
      });

    } catch (error) {
      console.error('Document deletion failed:', error);
      res.status(500).json({
        success: false,
        error: 'Document deletion failed',
        message: error.message
      });
    }
  });

  /**
   * POST /:id/reprocess - Reprocess document
   */
  router.post('/:id/reprocess', async (req, res) => {
    try {
      const { id } = req.params;
      const { options = {} } = req.body;

      // Check if document exists
      const existing = await database.getDocument(id);
      if (!existing) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Reprocess document
      const result = await documentProcessor.reprocessDocument(id, options);

      // Clear related caches
      await cache.clearByPattern(`document:${id}:*`);
      await cache.clearByPattern(`search:*`);

      // Emit real-time update
      if (io && existing.case_id) {
        io.to(`case-${existing.case_id}`).emit('document-reprocessed', {
          documentId: id,
          title: result.document.title,
          chunkCount: result.chunks.length,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        document: {
          id: result.document.id,
          title: result.document.title,
          chunkCount: result.chunks.length,
          aiAnalysis: result.aiAnalysis
        },
        stats: result.stats
      });

    } catch (error) {
      console.error('Document reprocessing failed:', error);
      res.status(500).json({
        success: false,
        error: 'Document reprocessing failed',
        message: error.message
      });
    }
  });

  /**
   * GET /:id/chunks - Get document chunks
   */
  router.get('/:id/chunks', async (req, res) => {
    try {
      const { id } = req.params;
      const { limit = 20, offset = 0 } = req.query;

      // Check cache
      const cacheKey = `document:${id}:chunks:${limit}:${offset}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          chunks: cached.chunks,
          total: cached.total,
          cached: true
        });
      }

      // Get chunks from database
      const chunksQuery = `
        SELECT * FROM rag_chunks 
        WHERE document_id = $1 
        ORDER BY chunk_index 
        LIMIT $2 OFFSET $3
      `;
      const chunksResult = await database.pool.query(chunksQuery, [id, parseInt(limit), parseInt(offset)]);

      // Get total count
      const countQuery = 'SELECT COUNT(*) FROM rag_chunks WHERE document_id = $1';
      const countResult = await database.pool.query(countQuery, [id]);
      const total = parseInt(countResult.rows[0].count);

      const chunks = chunksResult.rows.map(chunk => ({
        id: chunk.id,
        chunkIndex: chunk.chunk_index,
        content: chunk.content,
        metadata: chunk.metadata,
        createdAt: chunk.created_at
      }));

      // Cache results
      await cache.set(cacheKey, { chunks, total }, 1800); // 30 minutes

      res.json({
        success: true,
        chunks,
        total,
        cached: false,
        pagination: {
          limit: parseInt(limit),
          offset: parseInt(offset),
          hasMore: total > parseInt(offset) + parseInt(limit)
        }
      });

    } catch (error) {
      console.error('Chunks retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Chunks retrieval failed',
        message: error.message
      });
    }
  });

  /**
   * GET /:id/analysis - Get document AI analysis
   */
  router.get('/:id/analysis', async (req, res) => {
    try {
      const { id } = req.params;

      // Check cache
      const cacheKey = `document:${id}:analysis`;
      const cached = await cache.getCachedDocumentProcessing(id);
      
      if (cached && cached.aiAnalysis) {
        return res.json({
          success: true,
          analysis: cached.aiAnalysis,
          cached: true
        });
      }

      // Get document
      const document = await database.getDocument(id);
      if (!document) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Generate fresh analysis if not cached
      if (!document.content) {
        return res.status(400).json({
          success: false,
          error: 'Document has no content for analysis'
        });
      }

      const analysis = await documentProcessor.generateAIAnalysis(
        document.content,
        document.metadata || {}
      );

      // Cache the analysis
      await cache.set(cacheKey, analysis, 86400); // 24 hours

      res.json({
        success: true,
        analysis,
        cached: false
      });

    } catch (error) {
      console.error('Analysis retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Analysis retrieval failed',
        message: error.message
      });
    }
  });

  /**
   * GET /case/:caseId - Get documents for a case
   */
  router.get('/case/:caseId', async (req, res) => {
    try {
      const { caseId } = req.params;
      const { limit = 50, includeStats = false } = req.query;

      // Check cache
      const cacheKey = `documents:case:${caseId}:${limit}:${includeStats}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          ...cached,
          cached: true
        });
      }

      // Get documents for case
      const documents = await database.getDocuments({
        caseId,
        limit: parseInt(limit)
      });

      const response = {
        documents: documents.map(doc => ({
          id: doc.id,
          title: doc.title,
          documentType: doc.document_type,
          fileSize: doc.file_size,
          fileType: doc.file_type,
          processingStatus: doc.processing_status,
          createdAt: doc.created_at,
          indexedAt: doc.indexed_at
        })),
        total: documents.length
      };

      // Include statistics if requested
      if (includeStats === 'true') {
        const stats = {
          byType: {},
          byStatus: {},
          totalSize: 0,
          avgSize: 0
        };

        documents.forEach(doc => {
          // Count by type
          stats.byType[doc.document_type] = (stats.byType[doc.document_type] || 0) + 1;
          
          // Count by status
          stats.byStatus[doc.processing_status] = (stats.byStatus[doc.processing_status] || 0) + 1;
          
          // Sum file sizes
          if (doc.file_size) {
            stats.totalSize += doc.file_size;
          }
        });

        stats.avgSize = documents.length > 0 ? Math.round(stats.totalSize / documents.length) : 0;
        response.statistics = stats;
      }

      // Cache results
      await cache.set(cacheKey, response, 600); // 10 minutes

      res.json({
        success: true,
        ...response,
        cached: false
      });

    } catch (error) {
      console.error('Case documents retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Case documents retrieval failed',
        message: error.message
      });
    }
  });

  return router;
}