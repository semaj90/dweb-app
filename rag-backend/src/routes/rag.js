/**
 * RAG API Routes
 * Main API endpoints for Retrieval Augmented Generation operations
 */

import express from 'express';
import { z } from 'zod';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(process.cwd(), 'uploads');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain',
      'text/html',
      'image/jpeg',
      'image/png',
      'image/tiff'
    ];
    
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type: ${file.mimetype}`), false);
    }
  }
});

// Validation schemas
const querySchema = z.object({
  query: z.string().min(1, 'Query is required').max(1000, 'Query too long'),
  caseId: z.string().optional(),
  documentTypes: z.array(z.string()).optional().default([]),
  limit: z.number().int().min(1).max(50).optional().default(10),
  threshold: z.number().min(0).max(1).optional().default(0.7),
  includeContent: z.boolean().optional().default(true),
  searchType: z.enum(['vector', 'hybrid', 'chunks']).optional().default('hybrid')
});

const uploadMetadataSchema = z.object({
  title: z.string().optional(),
  documentType: z.string().optional().default('general'),
  caseId: z.string().optional(),
  metadata: z.record(z.any()).optional().default({})
});

export function createRAGRoutes(services, io) {
  const { database, vector, cache, ollama, documentProcessor, agentOrchestrator } = services;

  /**
   * POST /search - Semantic search across documents
   */
  router.post('/search', async (req, res) => {
    try {
      // Validate request
      const validatedData = querySchema.parse(req.body);
      const { query, caseId, documentTypes, limit, threshold, includeContent, searchType } = validatedData;

      // Check cache first
      const cacheKey = `search:${query}:${JSON.stringify({ caseId, documentTypes, limit, threshold, searchType })}`;
      const cached = await cache.getCachedSearchResults(query, { caseId, documentTypes, limit, threshold, searchType });
      
      if (cached) {
        return res.json({
          success: true,
          results: cached.results,
          cached: true,
          metadata: {
            query,
            resultCount: cached.results.length,
            searchType,
            timestamp: cached.timestamp
          }
        });
      }

      let results;
      const startTime = Date.now();

      // Perform search based on type
      switch (searchType) {
        case 'vector':
          results = await vector.searchDocuments(query, {
            limit,
            threshold,
            caseId,
            documentTypes,
            includeContent
          });
          break;
        
        case 'chunks':
          results = await vector.searchChunks(query, {
            limit,
            threshold,
            documentIds: caseId ? [caseId] : []
          });
          break;
        
        case 'hybrid':
        default:
          results = await vector.hybridSearch(query, {
            limit,
            caseId,
            documentTypes
          });
          break;
      }

      const processingTime = Date.now() - startTime;

      // Log query
      await database.logQuery({
        queryText: query,
        queryEmbedding: searchType === 'vector' ? await vector.generateEmbedding(query) : null,
        response: `Found ${results.length} results`,
        confidenceScore: results.length > 0 ? results[0].similarity_score || results[0].combined_score : 0,
        processingTimeMs: processingTime,
        sources: results.map(r => ({ id: r.id, title: r.title, type: r.document_type })),
        caseId
      });

      // Cache results
      await cache.cacheSearchResults(query, validatedData, results, 1800); // 30 minutes

      // Emit real-time update
      if (io && caseId) {
        io.to(`case-${caseId}`).emit('search-completed', {
          query,
          resultCount: results.length,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        results,
        cached: false,
        metadata: {
          query,
          resultCount: results.length,
          searchType,
          processingTime,
          timestamp: new Date().toISOString()
        }
      });

    } catch (error) {
      console.error('Search failed:', error);
      
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Validation error',
          details: error.errors
        });
      }

      res.status(500).json({
        success: false,
        error: 'Search failed',
        message: error.message
      });
    }
  });

  /**
   * POST /upload - Upload and process document
   */
  router.post('/upload', upload.single('document'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: 'No file uploaded'
        });
      }

      // Validate metadata
      const metadata = uploadMetadataSchema.parse(req.body);

      // Process document
      const result = await documentProcessor.processDocument(req.file.path, {
        ...metadata,
        originalName: req.file.originalname,
        mimeType: req.file.mimetype,
        uploadedAt: new Date().toISOString()
      });

      // Emit real-time update
      if (io && metadata.caseId) {
        io.to(`case-${metadata.caseId}`).emit('document-processed', {
          documentId: result.document.id,
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
          type: result.document.document_type,
          fileSize: result.document.file_size,
          chunkCount: result.chunks.length
        },
        aiAnalysis: result.aiAnalysis,
        stats: result.stats
      });

    } catch (error) {
      console.error('Upload failed:', error);
      
      // Clean up uploaded file on error
      if (req.file) {
        try {
          await fs.unlink(req.file.path);
        } catch (cleanupError) {
          console.error('Failed to cleanup uploaded file:', cleanupError);
        }
      }

      res.status(500).json({
        success: false,
        error: 'Upload failed',
        message: error.message
      });
    }
  });

  /**
   * POST /batch-upload - Upload and process multiple documents
   */
  router.post('/batch-upload', upload.array('documents', 10), async (req, res) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'No files uploaded'
        });
      }

      const metadata = uploadMetadataSchema.parse(req.body);
      const filePaths = req.files.map(file => file.path);

      // Process documents in batch
      const result = await documentProcessor.batchProcessDocuments(filePaths, metadata);

      // Emit real-time updates
      if (io && metadata.caseId) {
        io.to(`case-${metadata.caseId}`).emit('batch-processing-completed', {
          successful: result.stats.successful,
          failed: result.stats.failed,
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        results: result.results.map(r => ({
          filePath: path.basename(r.filePath),
          success: r.success,
          documentId: r.success ? r.result.document.id : null,
          error: r.error || null
        })),
        stats: result.stats
      });

    } catch (error) {
      console.error('Batch upload failed:', error);
      
      // Clean up uploaded files on error
      if (req.files) {
        for (const file of req.files) {
          try {
            await fs.unlink(file.path);
          } catch (cleanupError) {
            console.error('Failed to cleanup uploaded file:', cleanupError);
          }
        }
      }

      res.status(500).json({
        success: false,
        error: 'Batch upload failed',
        message: error.message
      });
    }
  });

  /**
   * POST /analyze - Analyze text with AI
   */
  router.post('/analyze', async (req, res) => {
    try {
      const { text, analysisType = 'general', options = {} } = req.body;

      if (!text || typeof text !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Text is required'
        });
      }

      if (text.length > 50000) {
        return res.status(400).json({
          success: false,
          error: 'Text is too long (max 50,000 characters)'
        });
      }

      // Check cache
      const cacheKey = `analyze:${analysisType}:${Buffer.from(text.substring(0, 1000)).toString('base64')}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          analysis: cached,
          cached: true
        });
      }

      // Perform AI analysis
      const result = await ollama.analyzeLegalDocument(text, analysisType, options);

      // Cache result
      await cache.set(cacheKey, result, 3600); // 1 hour

      res.json({
        success: true,
        analysis: result,
        cached: false
      });

    } catch (error) {
      console.error('Analysis failed:', error);
      res.status(500).json({
        success: false,
        error: 'Analysis failed',
        message: error.message
      });
    }
  });

  /**
   * POST /summarize - Summarize text
   */
  router.post('/summarize', async (req, res) => {
    try {
      const { text, length = 'medium', options = {} } = req.body;

      if (!text || typeof text !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Text is required'
        });
      }

      // Check cache
      const cacheKey = `summarize:${length}:${Buffer.from(text.substring(0, 1000)).toString('base64')}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          summary: cached,
          cached: true
        });
      }

      // Generate summary
      const result = await ollama.summarizeText(text, {
        summaryLength: length,
        ...options
      });

      // Cache result
      await cache.set(cacheKey, result, 3600);

      res.json({
        success: true,
        summary: result,
        cached: false
      });

    } catch (error) {
      console.error('Summarization failed:', error);
      res.status(500).json({
        success: false,
        error: 'Summarization failed',
        message: error.message
      });
    }
  });

  /**
   * POST /extract - Extract information from text
   */
  router.post('/extract', async (req, res) => {
    try {
      const { text, extractionType = 'entities', options = {} } = req.body;

      if (!text || typeof text !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Text is required'
        });
      }

      // Check cache
      const cacheKey = `extract:${extractionType}:${Buffer.from(text.substring(0, 1000)).toString('base64')}`;
      const cached = await cache.get(cacheKey);
      
      if (cached) {
        return res.json({
          success: true,
          extraction: cached,
          cached: true
        });
      }

      // Extract information
      const result = await ollama.extractKeyInfo(text, extractionType, options);

      // Cache result
      await cache.set(cacheKey, result, 3600);

      res.json({
        success: true,
        extraction: result,
        cached: false
      });

    } catch (error) {
      console.error('Extraction failed:', error);
      res.status(500).json({
        success: false,
        error: 'Extraction failed',
        message: error.message
      });
    }
  });

  /**
   * GET /similar/:documentId - Find similar documents
   */
  router.get('/similar/:documentId', async (req, res) => {
    try {
      const { documentId } = req.params;
      const { limit = 5, threshold = 0.8 } = req.query;

      const results = await vector.findSimilarDocuments(documentId, {
        limit: parseInt(limit),
        threshold: parseFloat(threshold)
      });

      res.json({
        success: true,
        results,
        metadata: {
          documentId,
          resultCount: results.length,
          threshold: parseFloat(threshold)
        }
      });

    } catch (error) {
      console.error('Similar documents search failed:', error);
      res.status(500).json({
        success: false,
        error: 'Similar documents search failed',
        message: error.message
      });
    }
  });

  /**
   * GET /stats - Get RAG system statistics
   */
  router.get('/stats', async (req, res) => {
    try {
      const dbStats = await database.getHealthStats();
      const processingStats = await documentProcessor.getProcessingStats();
      const cacheStats = await cache.getStats();

      res.json({
        success: true,
        stats: {
          documents: {
            total: dbStats.totalDocuments,
            indexed: dbStats.indexedDocuments,
            pendingProcessing: dbStats.totalDocuments - dbStats.indexedDocuments
          },
          chunks: {
            total: dbStats.totalChunks
          },
          queries: {
            last24h: dbStats.queriesLast24h,
            avgProcessingTime: dbStats.avgProcessingTime
          },
          jobs: {
            pending: dbStats.pendingJobs,
            running: dbStats.runningJobs
          },
          processing: processingStats,
          cache: cacheStats ? {
            connected: cacheStats.connected,
            memoryUsage: cacheStats.memory?.used_memory_human,
            hitRate: cacheStats.stats?.keyspace_hits / (cacheStats.stats?.keyspace_hits + cacheStats.stats?.keyspace_misses) * 100
          } : null
        },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Stats retrieval failed:', error);
      res.status(500).json({
        success: false,
        error: 'Stats retrieval failed',
        message: error.message
      });
    }
  });

  /**
   * DELETE /cache - Clear cache
   */
  router.delete('/cache', async (req, res) => {
    try {
      const { pattern = 'rag:*' } = req.query;
      const cleared = await cache.clearByPattern(pattern);

      res.json({
        success: true,
        message: `Cleared ${cleared} cache entries`,
        pattern
      });

    } catch (error) {
      console.error('Cache clear failed:', error);
      res.status(500).json({
        success: false,
        error: 'Cache clear failed',
        message: error.message
      });
    }
  });

  return router;
}