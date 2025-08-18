// API Endpoints Integration Tests
// Comprehensive testing for all REST API endpoints

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { json } from '@sveltejs/kit';

// Import API handlers
import { POST as documentsAnalyzePost, GET as documentsAnalyzeGet } from '../../src/routes/api/documents/analyze/+server.js';
import { POST as documentsSearchPost } from '../../src/routes/api/documents/search/+server.js';
import { POST as documentsStorePost } from '../../src/routes/api/documents/store/+server.js';
import { POST as aiEmbeddingsPost } from '../../src/routes/api/ai/embeddings/+server.js';
import { POST as aiSynthesizerPost } from '../../src/routes/api/ai-synthesizer/+server.js';
import { POST as legalResearchPost } from '../../src/routes/api/ai/legal-research/+server.js';

describe('API Endpoints Integration Tests', () => {
  beforeEach(() => {
    // Mock database connections
    vi.mock('../../src/lib/database/postgres.js', () => ({
      db: {
        query: {
          legalDocuments: {
            findFirst: vi.fn(),
            findMany: vi.fn()
          }
        },
        insert: vi.fn().mockReturnValue({
          values: vi.fn().mockReturnValue({
            returning: vi.fn().mockResolvedValue([testUtils.generateMockDocument()])
          })
        }),
        update: vi.fn().mockReturnValue({
          set: vi.fn().mockReturnValue({
            where: vi.fn().mockReturnValue({
              returning: vi.fn().mockResolvedValue([testUtils.generateMockDocument()])
            })
          })
        }),
        select: vi.fn().mockReturnValue({
          from: vi.fn().mockReturnValue({
            where: vi.fn().mockReturnValue({
              orderBy: vi.fn().mockReturnValue({
                limit: vi.fn().mockResolvedValue([testUtils.generateMockDocument()])
              })
            })
          })
        })
      }
    }));

    // Mock external services
    global.fetch = vi.fn().mockResolvedValue(testUtils.createMockResponse({
      embedding: testUtils.generateMockEmbedding()
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Documents Analysis API', () => {
    it('should analyze document successfully', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        title: 'Test Contract',
        content: 'This is a test contract content with liability and indemnity clauses.',
        documentType: 'contract',
        jurisdiction: 'federal',
        practiceArea: 'corporate'
      });

      const response = await documentsAnalyzePost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.documentId).toBeDefined();
      expect(responseData.analysis).toBeDefined();
      expect(responseData.embeddings).toBeDefined();
    });

    it('should return error for missing required fields', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({});

      const response = await documentsAnalyzePost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(400);
      expect(responseData.error).toBe('Content and title are required');
    });

    it('should retrieve document by ID', async () => {
      const mockDocument = testUtils.generateMockDocument();
      
      // Mock database query
      vi.mocked(global.fetch).mockResolvedValueOnce(testUtils.createMockResponse(mockDocument));

      const url = new URL('http://localhost:3000/api/documents/analyze?id=test-doc-123');
      const mockEvent = testUtils.createMockRequestEvent({}, 'GET');
      mockEvent.url = url;
      const response = await documentsAnalyzeGet(mockEvent);
      
      expect(response.status).toBe(200);
    });
  });

  describe('Documents Search API', () => {
    it('should perform semantic search successfully', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'contract breach liability',
        searchType: 'semantic',
        limit: 10
      });

      const response = await documentsSearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.results).toBeDefined();
      expect(Array.isArray(responseData.results)).toBe(true);
      expect(responseData.metadata).toBeDefined();
      expect(responseData.metadata.searchType).toBe('semantic');
    });

    it('should perform full-text search', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'intellectual property rights',
        searchType: 'full-text',
        limit: 5
      });

      const response = await documentsSearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.metadata.searchType).toBe('full-text');
    });

    it('should perform hybrid search', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'employment termination',
        searchType: 'hybrid',
        filters: {
          jurisdiction: 'federal',
          documentType: 'contract'
        }
      });

      const response = await documentsSearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.metadata.searchType).toBe('hybrid');
    });

    it('should return error for missing query', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        searchType: 'semantic'
      });

      const response = await documentsSearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(400);
      expect(responseData.error).toBe('Query is required');
    });
  });

  describe('Documents Store API', () => {
    it('should store document with embeddings', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        title: 'Legal Memorandum',
        content: 'This is a comprehensive legal memorandum discussing contract interpretation.',
        documentType: 'memorandum',
        jurisdiction: 'state',
        practiceArea: 'litigation',
        generateEmbeddings: true,
        chunkSize: 500
      });

      const response = await documentsStorePost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.documentId).toBeDefined();
      expect(responseData.embeddings).toBeDefined();
      expect(responseData.processing.embeddingsGenerated).toBe(true);
    });

    it('should detect duplicate documents', async () => {
      // Mock existing document
      const existingDoc = testUtils.generateMockDocument();
      vi.mocked(global.fetch).mockResolvedValueOnce(testUtils.createMockResponse(existingDoc));

      const mockRequestEvent = testUtils.createMockRequestEvent({
        title: 'Duplicate Document',
        content: 'This content should trigger duplicate detection.'
      });

      const response = await documentsStorePost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.duplicate).toBe(true);
    });
  });

  describe('AI Embeddings API', () => {
    it('should generate embeddings for text', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        text: 'This is a legal document about contract interpretation and liability.',
        model: 'nomic-embed-text'
      });

      const response = await aiEmbeddingsPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.embedding).toBeDefined();
      expect(Array.isArray(responseData.embedding)).toBe(true);
      expect(responseData.dimensions).toBe(384);
    });

    it('should save embeddings to database when documentId provided', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        text: 'Legal document content for embedding.',
        documentId: 'test-doc-123',
        contentType: 'document',
        metadata: { source: 'test' }
      });

      const response = await aiEmbeddingsPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.savedRecord).toBeDefined();
      expect(responseData.savedRecord.contentId).toBe('test-doc-123');
    });
  });

  describe('AI Synthesizer API', () => {
    it('should synthesize analysis from query and documents', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'What are the key liability risks in this contract?',
        documents: [testUtils.generateMockDocument()],
        synthesisType: 'analysis'
      });

      const response = await aiSynthesizerPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.result).toBeDefined();
      expect(responseData.result.synthesis).toBeDefined();
      expect(responseData.result.confidence).toBeGreaterThan(0);
    });

    it('should return metrics for system status', async () => {
      const response = await fetch('/api/ai-synthesizer', { method: 'GET' });
      // This would test the GET endpoint for metrics
    });
  });

  describe('Legal Research API', () => {
    it('should perform comprehensive legal research', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'contract breach damages precedent',
        jurisdiction: 'federal',
        practiceArea: 'commercial',
        documentTypes: ['case', 'statute'],
        limit: 15,
        enableReranking: true
      });

      const response = await legalResearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.success).toBe(true);
      expect(responseData.results).toBeDefined();
      expect(responseData.metadata).toBeDefined();
      expect(responseData.metadata.reranked).toBe(true);
      expect(responseData.suggestions).toBeDefined();
    });

    it('should filter results by jurisdiction and practice area', async () => {
      const mockRequestEvent = testUtils.createMockRequestEvent({
        query: 'employment discrimination',
        jurisdiction: 'california',
        practiceArea: 'employment',
        documentTypes: ['case']
      });

      const response = await legalResearchPost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(200);
      expect(responseData.metadata.jurisdiction).toBe('california');
      expect(responseData.metadata.practiceArea).toBe('employment');
    });
  });

  describe('Error Handling', () => {
    it('should handle database connection errors gracefully', async () => {
      // Mock database error
      vi.mocked(global.fetch).mockRejectedValueOnce(new Error('Database connection failed'));

      const mockRequestEvent = testUtils.createMockRequestEvent({
        title: 'Test Document',
        content: 'Test content'
      });

      const response = await documentsAnalyzePost(mockRequestEvent);
      const responseData = await response.json();

      expect(response.status).toBe(500);
      expect(responseData.error).toBeDefined();
    });

    it('should handle invalid JSON payloads', async () => {
      const mockRequest = testUtils.createMockRequestEvent({});
      mockRequest.request.json = vi.fn().mockRejectedValue(new Error('Invalid JSON'));

      const response = await documentsAnalyzePost(mockRequest);
      const responseData = await response.json();

      expect(response.status).toBe(500);
    });
  });

  describe('Performance Tests', () => {
    it('should handle large document analysis within reasonable time', async () => {
      const largeContent = 'Large document content. '.repeat(10000);
      
      const mockRequestEvent = testUtils.createMockRequestEvent({
        title: 'Large Document',
        content: largeContent,
        documentType: 'brief'
      });

      const startTime = Date.now();
      const response = await documentsAnalyzePost(mockRequestEvent);
      const endTime = Date.now();

      expect(response.status).toBe(200);
      expect(endTime - startTime).toBeLessThan(30000); // Should complete within 30 seconds
    });

    it('should handle concurrent requests efficiently', async () => {
      const requests = Array(10).fill(null).map((_, i) => 
        testUtils.createMockRequestEvent({
          query: `Test query ${i}`,
          searchType: 'semantic'
        })
      );

      const startTime = Date.now();
      const responses = await Promise.all(
        requests.map(request => documentsSearchPost(request))
      );
      const endTime = Date.now();

      expect(responses).toHaveLength(10);
      responses.forEach(response => {
        expect(response.status).toBe(200);
      });
      expect(endTime - startTime).toBeLessThan(60000); // Should complete within 60 seconds
    });
  });
});
