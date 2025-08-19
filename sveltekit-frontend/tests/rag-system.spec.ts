// @ts-nocheck
import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('RAG (Retrieval-Augmented Generation) System', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the case management or document page
    await page.goto('/dashboard/cases');
    
    // Login if needed
    const loginButton = page.locator('button:has-text("Login")');
    if (await loginButton.isVisible()) {
      await page.goto('/login');
      await page.fill('input[name="email"]', 'demo@example.com');
      await page.fill('input[name="password"]', 'demoPassword123');
      await page.click('button[type="submit"]');
      await page.waitForURL('/dashboard/**');
    }
  });

  test('should upload and index documents for RAG', async ({ page }) => {
    // Navigate to document upload
    await page.goto('/dashboard/documents/upload');
    
    // Upload a test document
    const fileInput = page.locator('input[type="file"]');
    const testFile = path.join(import.meta.url, 'fixtures', 'test-legal-document.pdf');
    
    // Create a test file if it doesn't exist
    await page.evaluate(() => {
      const file = new File(['Test legal document content about contract law'], 'test-legal-document.pdf', {
        type: 'application/pdf'
      });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (input) {
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    
    // Wait for upload to complete
    await page.waitForSelector('[data-testid="upload-success"]', { timeout: 30000 });
    
    // Verify document was indexed
    const response = await page.request.get('/api/documents/search', {
      params: {
        query: 'contract law'
      }
    });
    
    expect(response.status()).toBe(200);
    const results = await response.json();
    expect(results.documents).toHaveLength(1);
  });

  test('should perform semantic search on indexed documents', async ({ page }) => {
    // Perform a semantic search
    const response = await page.request.post('/api/rag/search', {
      data: {
        query: 'What are the key elements of a valid contract?',
        limit: 5,
        threshold: 0.7
      }
    });
    
    expect(response.status()).toBe(200);
    
    const results = await response.json();
    expect(results).toHaveProperty('chunks');
    expect(Array.isArray(results.chunks)).toBe(true);
    
    // Check each result has required fields
    results.chunks.forEach((chunk: any) => {
      expect(chunk).toHaveProperty('content');
      expect(chunk).toHaveProperty('metadata');
      expect(chunk).toHaveProperty('similarity_score');
      expect(chunk.similarity_score).toBeGreaterThanOrEqual(0.7);
    });
  });

  test('should generate augmented responses with context', async ({ page }) => {
    // First, ensure we have some documents indexed
    await page.request.post('/api/documents/index-sample');
    
    // Test RAG query
    const response = await page.request.post('/api/rag/query', {
      data: {
        question: 'What are the legal requirements for forming a corporation?',
        use_context: true,
        model: 'llama3.2'
      }
    });
    
    expect(response.status()).toBe(200);
    
    const result = await response.json();
    expect(result).toHaveProperty('answer');
    expect(result).toHaveProperty('sources');
    expect(result).toHaveProperty('context_used');
    
    // Verify the answer includes context from documents
    expect(result.context_used).toBe(true);
    expect(Array.isArray(result.sources)).toBe(true);
    expect(result.sources.length).toBeGreaterThan(0);
    
    // Check that sources have proper structure
    result.sources.forEach((source: any) => {
      expect(source).toHaveProperty('document_id');
      expect(source).toHaveProperty('chunk_text');
      expect(source).toHaveProperty('relevance_score');
    });
  });

  test('should handle vector embeddings with pgvector', async ({ page }) => {
    const testText = 'This is a test legal document about intellectual property rights';
    
    // Generate embeddings
    const embeddingResponse = await page.request.post('/api/rag/embed', {
      data: {
        text: testText,
        model: 'nomic-embed-text'
      }
    });
    
    expect(embeddingResponse.status()).toBe(200);
    
    const embedding = await embeddingResponse.json();
    expect(embedding).toHaveProperty('vector');
    expect(embedding).toHaveProperty('dimension');
    expect(embedding.dimension).toBe(768); // nomic-embed-text dimension
    
    // Store in pgvector
    const storeResponse = await page.request.post('/api/rag/store-embedding', {
      data: {
        content: testText,
        embedding: embedding.vector,
        metadata: {
          type: 'legal_document',
          category: 'intellectual_property'
        }
      }
    });
    
    expect(storeResponse.status()).toBe(201);
    
    // Search using vector similarity
    const searchResponse = await page.request.post('/api/rag/vector-search', {
      data: {
        query_text: 'patent and trademark law',
        limit: 5
      }
    });
    
    expect(searchResponse.status()).toBe(200);
    const searchResults = await searchResponse.json();
    expect(searchResults.results).toBeDefined();
    expect(searchResults.results.length).toBeGreaterThan(0);
  });

  test('should use RAG in chat conversations', async ({ page }) => {
    // Navigate to AI chat with RAG enabled
    await page.goto('/dashboard/ai-assistant');
    
    // Enable RAG mode
    const ragToggle = page.locator('[data-testid="rag-toggle"]');
    if (await ragToggle.isVisible()) {
      await ragToggle.click();
    }
    
    // Ask a question that requires document context
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill('What does our company policy say about remote work?');
    await page.locator('[data-testid="send-button"]').click();
    
    // Wait for RAG processing indicator
    await page.waitForSelector('[data-testid="rag-processing"]', { timeout: 5000 });
    
    // Wait for response with sources
    await page.waitForSelector('[data-testid="ai-response-with-sources"]', { timeout: 30000 });
    
    // Verify sources are displayed
    const sources = page.locator('[data-testid="source-citation"]');
    const sourceCount = await sources.count();
    expect(sourceCount).toBeGreaterThan(0);
    
    // Check that each source can be expanded
    const firstSource = sources.first();
    await firstSource.click();
    await page.waitForSelector('[data-testid="source-detail"]');
  });

  test('should update vector index when documents are modified', async ({ page }) => {
    // Create a document
    const createResponse = await page.request.post('/api/documents/create', {
      data: {
        title: 'Test Legal Brief',
        content: 'Original content about contract formation',
        type: 'legal_brief'
      }
    });
    
    expect(createResponse.status()).toBe(201);
    const document = await createResponse.json();
    
    // Search for original content
    const searchResponse1 = await page.request.post('/api/rag/search', {
      data: {
        query: 'contract formation',
        limit: 5
      }
    });
    
    const results1 = await searchResponse1.json();
    expect(results1.chunks.some((c: any) => c.document_id === document.id)).toBe(true);
    
    // Update the document
    const updateResponse = await page.request.put(`/api/documents/${document.id}`, {
      data: {
        content: 'Updated content about tort liability'
      }
    });
    
    expect(updateResponse.status()).toBe(200);
    
    // Wait for re-indexing
    await page.waitForTimeout(2000);
    
    // Search for updated content
    const searchResponse2 = await page.request.post('/api/rag/search', {
      data: {
        query: 'tort liability',
        limit: 5
      }
    });
    
    const results2 = await searchResponse2.json();
    expect(results2.chunks.some((c: any) => c.document_id === document.id)).toBe(true);
    
    // Verify old content is not found
    const searchResponse3 = await page.request.post('/api/rag/search', {
      data: {
        query: 'contract formation',
        limit: 5
      }
    });
    
    const results3 = await searchResponse3.json();
    const docChunks = results3.chunks.filter((c: any) => c.document_id === document.id);
    expect(docChunks.length).toBe(0);
  });

  test('should chunk large documents appropriately', async ({ page }) => {
    // Create a large document
    const largeContent = Array(100).fill('This is a paragraph about legal principles. ').join('\n\n');
    
    const response = await page.request.post('/api/documents/create', {
      data: {
        title: 'Large Legal Document',
        content: largeContent,
        type: 'legal_document'
      }
    });
    
    expect(response.status()).toBe(201);
    const document = await response.json();
    
    // Check chunking results
    const chunksResponse = await page.request.get(`/api/documents/${document.id}/chunks`);
    expect(chunksResponse.status()).toBe(200);
    
    const chunks = await chunksResponse.json();
    expect(chunks.chunks).toBeDefined();
    expect(chunks.chunks.length).toBeGreaterThan(1);
    
    // Verify chunk properties
    chunks.chunks.forEach((chunk: any) => {
      expect(chunk).toHaveProperty('content');
      expect(chunk).toHaveProperty('chunk_index');
      expect(chunk).toHaveProperty('embedding');
      expect(chunk.content.length).toBeLessThanOrEqual(1000); // Max chunk size
    });
  });

  test('should filter RAG results by metadata', async ({ page }) => {
    // Search with metadata filters
    const response = await page.request.post('/api/rag/search', {
      data: {
        query: 'legal precedent',
        filters: {
          document_type: 'case_law',
          jurisdiction: 'federal',
          date_after: '2020-01-01'
        },
        limit: 10
      }
    });
    
    expect(response.status()).toBe(200);
    
    const results = await response.json();
    
    // Verify all results match filters
    results.chunks.forEach((chunk: any) => {
      expect(chunk.metadata.document_type).toBe('case_law');
      expect(chunk.metadata.jurisdiction).toBe('federal');
      expect(new Date(chunk.metadata.date)).toBeGreaterThan(new Date('2020-01-01'));
    });
  });

  test('should handle hybrid search (keyword + semantic)', async ({ page }) => {
    const response = await page.request.post('/api/rag/hybrid-search', {
      data: {
        query: 'negligence tort damages',
        semantic_weight: 0.7,
        keyword_weight: 0.3,
        limit: 10
      }
    });
    
    expect(response.status()).toBe(200);
    
    const results = await response.json();
    expect(results).toHaveProperty('results');
    expect(results).toHaveProperty('search_method');
    expect(results.search_method).toBe('hybrid');
    
    // Check that results have both scores
    results.results.forEach((result: any) => {
      expect(result).toHaveProperty('semantic_score');
      expect(result).toHaveProperty('keyword_score');
      expect(result).toHaveProperty('combined_score');
    });
  });
});