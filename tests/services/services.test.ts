// Service Layer Tests
// Comprehensive testing for core services and utilities

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EnhancedSentenceSplitter } from '../../shared/text/enhanced-sentence-splitter';
import { EnhancedLegalOrchestrator } from '../../src/lib/agents/orchestrator-enhanced.js';
import { EnhancedQdrantManager } from '../../src/lib/database/qdrant-enhanced.js';
import { RagPipelineIntegrator } from '../../src/lib/services/rag-pipeline-integrator.js';

describe('Service Layer Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Mock external dependencies
    global.fetch = vi.fn().mockResolvedValue(testUtils.createMockResponse({
      embedding: testUtils.generateMockEmbedding(),
      response: 'Mock AI response'
    }));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Enhanced Legal Orchestrator', () => {
    let orchestrator: EnhancedLegalOrchestrator;

    beforeEach(() => {
      orchestrator = new EnhancedLegalOrchestrator();
    });

    it('should orchestrate legal analysis successfully', async () => {
      const request = {
        query: 'Analyze the liability clauses in this contract',
        documentType: 'contract',
        jurisdiction: 'federal',
        urgency: 'medium' as const,
        requiresMultiAgent: true,
        enableStreaming: false,
        context: { practiceArea: 'corporate' }
      };

      const response = await orchestrator.orchestrate(request);

      expect(response).toBeDefined();
      expect(response.synthesizedConclusion).toContain('legal analysis');
      expect(response.confidence).toBeGreaterThan(0);
      expect(response.totalProcessingTime).toBeGreaterThan(0);
      expect(response.primaryResponse).toBeDefined();
      expect(response.primaryResponse.agentName).toBeDefined();
      expect(response.metadata).toBeDefined();
      expect(response.metadata.requestId).toBeDefined();
    });

    it('should handle multi-agent orchestration', async () => {
      const request = {
        query: 'Complex legal analysis requiring multiple perspectives',
        requiresMultiAgent: true,
        enableStreaming: false
      };

      const response = await orchestrator.orchestrate(request);

      expect(response.secondaryResponses).toBeDefined();
      expect(Array.isArray(response.secondaryResponses)).toBe(true);
    });

    it('should enable streaming when requested', async () => {
      const request = {
        query: 'Legal analysis with streaming response',
        enableStreaming: true
      };

      const response = await orchestrator.orchestrate(request);

      expect(response.streaming).toBeDefined();
      expect(response.streaming!.enabled).toBe(true);
      expect(response.streaming!.chunks).toBeDefined();
      expect(Array.isArray(response.streaming!.chunks)).toBe(true);
    });

    it('should analyze documents comprehensively', async () => {
      const document = {
        title: 'Software License Agreement',
        content: 'This agreement contains liability, indemnity, and termination clauses. The licensor shall not be liable for any damages.',
        type: 'contract'
      };

      const analysis = await orchestrator.analyze(document);

      expect(analysis).toBeDefined();
      expect(analysis.summary).toBeDefined();
      expect(analysis.keyTerms).toContain('liability');
      expect(analysis.risks).toBeDefined();
      expect(analysis.recommendations).toBeDefined();
      expect(analysis.confidence).toBeGreaterThan(0);
      expect(analysis.entities).toBeDefined();
      expect(analysis.citations).toBeDefined();
    });

    it('should search legal precedents effectively', async () => {
      const query = 'contract breach damages';
      const options = {
        jurisdiction: 'federal',
        documentType: 'case',
        limit: 5
      };

      const precedents = await orchestrator.searchPrecedents(query, options);

      expect(Array.isArray(precedents)).toBe(true);
      expect(precedents.length).toBeGreaterThan(0);

      precedents.forEach(precedent => {
        expect(precedent.id).toBeDefined();
        expect(precedent.title).toBeDefined();
        expect(precedent.relevance).toBeGreaterThan(0);
        expect(precedent.citation).toBeDefined();
        expect(precedent.keyHoldings).toBeDefined();
      });
    });

    it('should generate legal documents from templates', async () => {
      const contractParams = {
        party1: 'Acme Corp',
        party2: 'Beta LLC',
        scope: 'Software development services',
        compensation: '$50,000',
        startDate: '2024-01-01',
        jurisdiction: 'Delaware'
      };

      const contract = await orchestrator.generateDocument('contract', contractParams);

      expect(contract).toBeDefined();
      expect(contract).toContain('Acme Corp');
      expect(contract).toContain('Beta LLC');
      expect(contract).toContain('Software development services');
      expect(contract).toContain('$50,000');
    });

    it('should handle session management', async () => {
      const sessionId = 'test-session-123';
      const request = {
        query: 'Test query for session management',
        sessionId
      };

      await orchestrator.orchestrate(request);

      const session = orchestrator.getSession(sessionId);
      expect(session).toBeDefined();
      expect(session.history).toBeDefined();
      expect(session.history.length).toBeGreaterThan(0);

      orchestrator.clearSession(sessionId);
      const clearedSession = orchestrator.getSession(sessionId);
      expect(clearedSession).toBeNull();
    });

    it('should validate input parameters', async () => {
      await expect(orchestrator.orchestrate({} as any))
        .rejects.toThrow('Query is required and must be a string');

      await expect(orchestrator.generateDocument('', {}))
        .rejects.toThrow('Document type is required');

      await expect(orchestrator.generateDocument('unsupported-type', {}))
        .rejects.toThrow('Unsupported document type: unsupported-type');
    });
  });

  describe('Enhanced Sentence Splitter', () => {
    let splitter: EnhancedSentenceSplitter;

    beforeEach(() => {
      splitter = new EnhancedSentenceSplitter({
        minLength: 10,
        maxLength: 200,
  // preserveAbbreviations removed: handled implicitly by abbreviation protection
      });
    });

    it('should split legal text accurately', async () => {
      const legalText = 'The case Smith v. Jones was decided by the U.S. Supreme Court. The ruling established important precedent. It affects future contract interpretation.';

      const sentences = splitter.split(legalText);

      expect(sentences).toHaveLength(3);
      expect(sentences[0]).toContain('Smith v. Jones');
      expect(sentences[0]).toContain('U.S. Supreme Court');
      expect(sentences[1]).toContain('ruling established');
      expect(sentences[2]).toContain('future contract interpretation');
    });

    it('should preserve legal abbreviations', async () => {
      const textWithAbbreviations = 'The corporation, Inc. filed with the S.E.C. per federal regulations. The C.E.O. signed the documents.';

      const sentences = splitter.split(textWithAbbreviations);

      expect(sentences).toHaveLength(2);
      expect(sentences[0]).toContain('Inc.');
      expect(sentences[0]).toContain('S.E.C.');
      expect(sentences[1]).toContain('C.E.O.');
    });

    it('should handle complex legal citations', async () => {
      const citationText = 'In Brown v. Board of Education, 347 U.S. 483 (1954), the Court held that separate educational facilities are inherently unequal.';

      const sentences = splitter.split(citationText);

      expect(sentences).toHaveLength(1);
      expect(sentences[0]).toContain('347 U.S. 483 (1954)');
    });

    it('should respect length constraints', async () => {
      const shortSplitter = new EnhancedSentenceSplitter({
        minLength: 50,
        maxLength: 100
      });

      const text = 'Short. This is a longer sentence that meets the minimum length requirement and should be included.';

      const sentences = shortSplitter.split(text);

      expect(sentences).toHaveLength(1);
      expect(sentences[0].length).toBeGreaterThanOrEqual(50);
      expect(sentences[0].length).toBeLessThanOrEqual(100);
    });

    it('should add custom abbreviations dynamically', async () => {
      splitter.addAbbreviations(['Cust.', 'Spec.', 'Req.']);

      const text = 'The Cust. Agreement references Spec. Document 123. The Req. Analysis is complete.';

      const sentences = splitter.split(text);

      expect(sentences).toHaveLength(2);
      expect(sentences[0]).toContain('Cust. Agreement');
      expect(sentences[1]).toContain('Req. Analysis');
    });
  });

  describe('RAG Pipeline Integrator', () => {
    let ragPipeline: RagPipelineIntegrator;

    beforeEach(() => {
      ragPipeline = new RagPipelineIntegrator({
        embeddingModel: 'nomic-embed-text',
        retrievalLimit: 10,
        rerankingEnabled: true,
        vectorStore: 'qdrant'
      });
    });

    it('should process documents for RAG pipeline', async () => {
      const documentProgress = {
        documents: [
          {
            id: 'doc-1',
            title: 'Contract Analysis',
            content: 'This contract contains important liability clauses.',
            score: 0.9
          },
          {
            id: 'doc-2',
            title: 'Legal Precedent',
            content: 'Relevant case law for contract interpretation.',
            score: 0.8
          }
        ],
        totalProcessed: 2,
        totalToProcess: 2,
        currentStep: 'embedding_generation'
      };

      const results = await ragPipeline.processDocuments(documentProgress);

      expect(Array.isArray(results)).toBe(true);
      expect(results).toHaveLength(2);

      results.forEach(result => {
        expect(result.id).toBeDefined();
        expect(result.title).toBeDefined();
        expect(result.score).toBeGreaterThan(0);
        expect(result.document).toBeDefined();
      });
    });

    it('should retrieve relevant documents', async () => {
      const query = 'contract liability and indemnification';
      const filters = {
        documentType: 'contract',
        jurisdiction: 'federal'
      };

      const retrievalResult = await ragPipeline.retrieveDocuments(query, filters);

      expect(retrievalResult).toBeDefined();
      expect(retrievalResult.documents).toBeDefined();
      expect(Array.isArray(retrievalResult.documents)).toBe(true);
      expect(retrievalResult.metadata).toBeDefined();
      expect(retrievalResult.metadata.queryTime).toBeGreaterThan(0);
      expect(retrievalResult.metadata.model).toBe('nomic-embed-text');
    });

    it('should generate contextual responses', async () => {
      const query = 'What are the key risks in this contract?';
      const context = [
        {
          score: 0.9,
          rank: 1,
          id: 'doc-1',
          title: 'Contract Risk Analysis',
          excerpt: 'The contract contains several liability provisions that may expose the company to significant risk.',
          document: testUtils.generateMockDocument()
        }
      ];

      const response = await ragPipeline.generateResponse(query, context);

      expect(response).toBeDefined();
      expect(response.response).toBeDefined();
      expect(response.sources).toBeDefined();
      expect(response.confidence).toBeGreaterThan(0);
      expect(response.sources).toEqual(context);
    });

    it('should update configuration dynamically', async () => {
      const newOptions = {
        retrievalLimit: 20,
        rerankingEnabled: false,
        llmProvider: 'claude' as const
      };

      ragPipeline.updateOptions(newOptions);
      const updatedOptions = ragPipeline.getOptions();

      expect(updatedOptions.retrievalLimit).toBe(20);
      expect(updatedOptions.rerankingEnabled).toBe(false);
      expect(updatedOptions.llmProvider).toBe('claude');
    });
  });

  describe('Enhanced Qdrant Manager', () => {
    let qdrantManager: EnhancedQdrantManager;

    beforeEach(() => {
      qdrantManager = new EnhancedQdrantManager({
        baseUrl: 'http://localhost:6333',
        collection: 'test_collection'
      });

      // Mock fetch for Qdrant API calls
      global.fetch = vi.fn().mockImplementation((url: string, options?: unknown) => {
        if (url.includes('/collections')) {
          return Promise.resolve(testUtils.createMockResponse({ status: 'ok' }));
        }
        if (url.includes('/points/search')) {
          return Promise.resolve(testUtils.createMockResponse({
            result: [
              {
                id: 'doc-1',
                score: 0.95,
                payload: {
                  documentId: 'doc-1',
                  title: 'Test Document',
                  content: 'Test content'
                }
              }
            ]
          }));
        }
        return Promise.resolve(testUtils.createMockResponse({}));
      });
    });

    it('should connect to Qdrant successfully', async () => {
      const connected = await qdrantManager.connect();
      expect(connected).toBe(true);
    });

    it('should upsert documents with vectors', async () => {
      const documentRequest = {
        id: 'test-doc-1',
        vector: testUtils.generateMockEmbedding(),
        payload: {
          documentId: 'test-doc-1',
          title: 'Test Legal Document',
          documentType: 'contract',
          jurisdiction: 'federal',
          practiceArea: 'corporate',
          content: 'Test document content',
          metadata: { source: 'test' },
          timestamp: Date.now()
        }
      };

      const success = await qdrantManager.upsertDocument(documentRequest);
      expect(success).toBe(true);
    });

    it('should search for similar documents', async () => {
      await qdrantManager.connect();

      const queryVector = testUtils.generateMockEmbedding();
      const options = {
        limit: 5,
        documentType: 'contract',
        jurisdiction: 'federal',
        scoreThreshold: 0.7
      };

      const results = await qdrantManager.searchDocuments(queryVector, options);

      expect(Array.isArray(results)).toBe(true);
      results.forEach(result => {
        expect(result.documentId).toBeDefined();
        expect(result.title).toBeDefined();
        expect(result.score).toBeGreaterThan(0);
      });
    });

    it('should perform batch upsert operations', async () => {
      const documents = Array(5).fill(null).map((_, i) => ({
        id: `batch-doc-${i}`,
        vector: testUtils.generateMockEmbedding(),
        payload: {
          documentId: `batch-doc-${i}`,
          title: `Batch Document ${i}`,
          documentType: 'contract',
          jurisdiction: 'federal',
          practiceArea: 'corporate',
          content: `Batch document content ${i}`,
          metadata: {},
          timestamp: Date.now()
        }
      }));

      const results = await qdrantManager.batchUpsertDocuments(documents);

      expect(results.success).toBe(5);
      expect(results.failed).toBe(0);
      expect(results.errors).toHaveLength(0);
    });

    it('should get health status', async () => {
      const healthStatus = await qdrantManager.getHealthStatus();

      expect(healthStatus).toBeDefined();
      expect(healthStatus.collection).toBe('test_collection');
      expect(typeof healthStatus.connected).toBe('boolean');
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle network failures gracefully', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      const orchestrator = new EnhancedLegalOrchestrator();

      // Should not throw, but handle gracefully
      const response = await orchestrator.orchestrate({
        query: 'Test query during network failure'
      });

      expect(response).toBeDefined();
      expect(response.synthesizedConclusion).toBeDefined();
    });

    it('should handle malformed data inputs', async () => {
      const splitter = new EnhancedSentenceSplitter();

      expect(splitter.split(null as any)).toEqual([]);
      expect(splitter.split(undefined as any)).toEqual([]);
      expect(splitter.split('')).toEqual([]);
      expect(splitter.split('   ')).toEqual([]);
    });

    it('should handle service timeouts', async () => {
      global.fetch = vi.fn().mockImplementation(() =>
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      );

      const ragPipeline = new RagPipelineIntegrator();

      const result = await ragPipeline.retrieveDocuments('test query');

      // Should provide fallback behavior
      expect(result).toBeDefined();
      expect(result.documents).toBeDefined();
    });
  });

  describe('Performance and Load Testing', () => {
    it('should handle large text processing efficiently', async () => {
      const largeText = 'Legal sentence. '.repeat(10000);
      const splitter = new EnhancedSentenceSplitter();

      const startTime = Date.now();
      const sentences = splitter.split(largeText);
      const endTime = Date.now();

      expect(sentences.length).toBeGreaterThan(0);
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should handle concurrent orchestration requests', async () => {
      const orchestrator = new EnhancedLegalOrchestrator();

      const requests = Array(10).fill(null).map((_, i) => ({
        query: `Concurrent test query ${i}`,
        sessionId: `concurrent-session-${i}`
      }));

      const startTime = Date.now();
      const responses = await Promise.all(
        requests.map(req => orchestrator.orchestrate(req))
      );
      const endTime = Date.now();

      expect(responses).toHaveLength(10);
      responses.forEach(response => {
        expect(response.synthesizedConclusion).toBeDefined();
      });
      expect(endTime - startTime).toBeLessThan(30000); // Should complete within 30 seconds
    });
  });
});
