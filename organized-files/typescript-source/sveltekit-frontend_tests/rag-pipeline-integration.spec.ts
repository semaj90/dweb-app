// @ts-nocheck
import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('Complete RAG Pipeline Integration', () => {
  let testCaseId: string;
  let testUserId: string;
  
  test.beforeAll(async ({ request }) => {
    // Setup test user and case
    const userResponse = await request.post('/api/test/setup-user', {
      data: {
        email: `rag-test-${Date.now()}@example.com`,
        password: 'TestPass123!',
        name: 'RAG Test User'
      }
    });
    
    const userData = await userResponse.json();
    testUserId = userData.userId;
    
    const caseResponse = await request.post('/api/cases', {
      data: {
        title: 'RAG Integration Test Case',
        description: 'Testing complete RAG pipeline',
        user_id: testUserId
      }
    });
    
    const caseData = await caseResponse.json();
    testCaseId = caseData.id;
  });

  test.afterAll(async ({ request }) => {
    // Cleanup
    if (testUserId) {
      await request.delete(`/api/test/cleanup-user/${testUserId}`);
    }
  });

  test('should complete full RAG pipeline: upload, process, embed, and query', async ({ page }) => {
    // Step 1: Upload multiple legal documents
    const documents = [
      {
        title: 'Contract Law Basics',
        content: `
          A contract is a legally binding agreement between two or more parties. 
          Essential elements include offer, acceptance, consideration, and mutual intent.
          Breach of contract occurs when one party fails to fulfill their obligations.
          Remedies may include damages, specific performance, or rescission.
        `
      },
      {
        title: 'Tort Law Principles',
        content: `
          Tort law addresses civil wrongs that cause harm or loss. 
          Negligence requires duty of care, breach of duty, causation, and damages.
          Intentional torts include assault, battery, and defamation.
          Strict liability applies regardless of intent or negligence.
        `
      },
      {
        title: 'Criminal Procedure',
        content: `
          Criminal procedure governs the process of criminal prosecution.
          Constitutional protections include right to counsel and due process.
          The burden of proof is beyond a reasonable doubt.
          Defendants are presumed innocent until proven guilty.
        `
      }
    ];
    
    // Upload documents
    for (const doc of documents) {
      const response = await page.request.post('/api/documents', {
        data: {
          case_id: testCaseId,
          title: doc.title,
          content: doc.content,
          type: 'legal_reference'
        }
      });
      
      expect(response.status()).toBe(201);
    }
    
    // Step 2: Process documents and generate embeddings
    const processResponse = await page.request.post('/api/rag/process-case-documents', {
      data: {
        case_id: testCaseId,
        chunk_size: 500,
        chunk_overlap: 50
      }
    });
    
    expect(processResponse.status()).toBe(200);
    const processResult = await processResponse.json();
    
    expect(processResult).toHaveProperty('documents_processed');
    expect(processResult).toHaveProperty('chunks_created');
    expect(processResult).toHaveProperty('embeddings_generated');
    
    expect(processResult.documents_processed).toBe(3);
    expect(processResult.chunks_created).toBeGreaterThan(3);
    expect(processResult.embeddings_generated).toBe(processResult.chunks_created);
    
    // Step 3: Test semantic search
    const searchQueries = [
      {
        query: 'What are the remedies for breach of contract?',
        expected_topic: 'contract'
      },
      {
        query: 'Explain negligence in tort law',
        expected_topic: 'tort'
      },
      {
        query: 'What constitutional rights do defendants have?',
        expected_topic: 'criminal'
      }
    ];
    
    for (const testQuery of searchQueries) {
      const searchResponse = await page.request.post('/api/rag/search', {
        data: {
          query: testQuery.query,
          case_id: testCaseId,
          limit: 3,
          threshold: 0.7
        }
      });
      
      expect(searchResponse.status()).toBe(200);
      const searchResults = await searchResponse.json();
      
      expect(searchResults.chunks).toBeDefined();
      expect(searchResults.chunks.length).toBeGreaterThan(0);
      
      // Verify relevance
      const topResult = searchResults.chunks[0];
      expect(topResult.content.toLowerCase()).toContain(testQuery.expected_topic);
      expect(topResult.similarity_score).toBeGreaterThan(0.7);
    }
    
    // Step 4: Test RAG-enhanced question answering
    const qaResponse = await page.request.post('/api/rag/ask', {
      data: {
        question: 'Compare the burden of proof in civil and criminal cases',
        case_id: testCaseId,
        model: 'llama3.2',
        use_context: true,
        max_context_chunks: 5
      }
    });
    
    expect(qaResponse.status()).toBe(200);
    const qaResult = await qaResponse.json();
    
    expect(qaResult).toHaveProperty('answer');
    expect(qaResult).toHaveProperty('context_used');
    expect(qaResult).toHaveProperty('sources');
    expect(qaResult).toHaveProperty('confidence_score');
    
    // Verify answer quality
    expect(qaResult.answer).toContain('criminal');
    expect(qaResult.answer).toContain('civil');
    expect(qaResult.context_used).toBe(true);
    expect(qaResult.sources.length).toBeGreaterThan(0);
    expect(qaResult.confidence_score).toBeGreaterThan(0.7);
  });

  test('should handle multi-turn conversations with context', async ({ page }) => {
    // Start a conversation
    const conversationId = `conv-${Date.now()}`;
    
    const questions = [
      'What is negligence?',
      'What are the four elements?', // Refers to previous context
      'How does it differ from strict liability?' // Builds on conversation
    ];
    
    let conversationContext = [];
    
    for (const question of questions) {
      const response = await page.request.post('/api/rag/chat', {
        data: {
          message: question,
          conversation_id: conversationId,
          case_id: testCaseId,
          use_rag: true,
          conversation_history: conversationContext
        }
      });
      
      expect(response.status()).toBe(200);
      const result = await response.json();
      
      expect(result).toHaveProperty('response');
      expect(result).toHaveProperty('sources_used');
      
      // Add to conversation history
      conversationContext.push({
        role: 'user',
        content: question
      });
      conversationContext.push({
        role: 'assistant',
        content: result.response
      });
    }
    
    // Verify conversation coherence
    const lastResponse = conversationContext[conversationContext.length - 1].content;
    expect(lastResponse).toContain('strict liability');
    expect(lastResponse).toContain('negligence');
  });

  test('should update embeddings when documents are modified', async ({ page }) => {
    // Create a document
    const createResponse = await page.request.post('/api/documents', {
      data: {
        case_id: testCaseId,
        title: 'Dynamic Update Test',
        content: 'Original content about property law and ownership rights',
        type: 'legal_document'
      }
    });
    
    const document = await createResponse.json();
    
    // Process and embed
    await page.request.post('/api/rag/process-document', {
      data: { document_id: document.id }
    });
    
    // Search for original content
    const search1Response = await page.request.post('/api/rag/search', {
      data: {
        query: 'property ownership rights',
        case_id: testCaseId
      }
    });
    
    const search1Results = await search1Response.json();
    const originalFound = search1Results.chunks.some(
      (c: any) => c.document_id === document.id
    );
    expect(originalFound).toBe(true);
    
    // Update document
    await page.request.patch(`/api/documents/${document.id}`, {
      data: {
        content: 'Updated content about employment law and workplace regulations'
      }
    });
    
    // Re-process document
    await page.request.post('/api/rag/process-document', {
      data: { document_id: document.id }
    });
    
    // Search for new content
    const search2Response = await page.request.post('/api/rag/search', {
      data: {
        query: 'employment workplace regulations',
        case_id: testCaseId
      }
    });
    
    const search2Results = await search2Response.json();
    const updatedFound = search2Results.chunks.some(
      (c: any) => c.document_id === document.id
    );
    expect(updatedFound).toBe(true);
    
    // Verify old content is not found
    const search3Response = await page.request.post('/api/rag/search', {
      data: {
        query: 'property ownership rights',
        case_id: testCaseId
      }
    });
    
    const search3Results = await search3Response.json();
    const oldContentFound = search3Results.chunks.some(
      (c: any) => c.document_id === document.id && 
      c.content.includes('property')
    );
    expect(oldContentFound).toBe(false);
  });

  test('should handle large document processing efficiently', async ({ page }) => {
    // Create a large document
    const largeContent = Array(50).fill(null).map((_, i) => `
      Section ${i + 1}: Legal Principles
      
      This section discusses various aspects of law including contracts, torts, 
      criminal law, property law, and constitutional law. Each area has unique 
      characteristics and requirements that must be understood by legal practitioners.
      
      Key points include jurisdiction, precedent, statutory interpretation, and 
      the role of common law in shaping legal decisions. Understanding these 
      principles is essential for effective legal practice.
    `).join('\n\n');
    
    const startTime = Date.now();
    
    // Upload large document
    const uploadResponse = await page.request.post('/api/documents', {
      data: {
        case_id: testCaseId,
        title: 'Large Legal Compendium',
        content: largeContent,
        type: 'reference'
      }
    });
    
    const document = await uploadResponse.json();
    
    // Process with RAG
    const processResponse = await page.request.post('/api/rag/process-document', {
      data: {
        document_id: document.id,
        chunk_size: 1000,
        chunk_overlap: 100
      }
    });
    
    const endTime = Date.now();
    const processingTime = endTime - startTime;
    
    expect(processResponse.status()).toBe(200);
    const result = await processResponse.json();
    
    // Should complete within reasonable time
    expect(processingTime).toBeLessThan(30000); // 30 seconds
    
    expect(result.chunks_created).toBeGreaterThan(40);
    expect(result.embeddings_generated).toBe(result.chunks_created);
    
    // Test query performance on large dataset
    const queryStart = Date.now();
    
    const queryResponse = await page.request.post('/api/rag/search', {
      data: {
        query: 'statutory interpretation principles',
        case_id: testCaseId,
        limit: 10
      }
    });
    
    const queryEnd = Date.now();
    const queryTime = queryEnd - queryStart;
    
    expect(queryResponse.status()).toBe(200);
    expect(queryTime).toBeLessThan(2000); // Should be fast
  });

  test('should integrate with UI chat interface', async ({ page }) => {
    // Navigate to chat interface
    await page.goto(`/dashboard/cases/${testCaseId}/chat`);
    
    // Wait for chat to load
    await page.waitForSelector('[data-testid="case-chat-interface"]');
    
    // Enable RAG mode
    const ragToggle = page.locator('[data-testid="enable-rag"]');
    if (await ragToggle.isVisible()) {
      await ragToggle.click();
    }
    
    // Type a question
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill('Summarize all the legal principles in this case');
    
    // Send message
    await page.locator('[data-testid="send-button"]').click();
    
    // Wait for RAG processing
    await page.waitForSelector('[data-testid="rag-indicator"]');
    
    // Wait for response
    await page.waitForSelector('[data-testid="ai-response"]', { timeout: 30000 });
    
    // Verify response includes sources
    const sources = page.locator('[data-testid="response-sources"]');
    await expect(sources).toBeVisible();
    
    const sourceItems = page.locator('[data-testid="source-item"]');
    const sourceCount = await sourceItems.count();
    expect(sourceCount).toBeGreaterThan(0);
    
    // Click to view source
    await sourceItems.first().click();
    
    // Verify source modal/expansion
    await page.waitForSelector('[data-testid="source-content"]');
    const sourceContent = await page.locator('[data-testid="source-content"]').textContent();
    expect(sourceContent).toBeTruthy();
  });

  test('should export RAG conversation with sources', async ({ page }) => {
    // Create a conversation with multiple turns
    const conversationId = `export-test-${Date.now()}`;
    
    const exchanges = [
      { question: 'What is a contract?', expectedInAnswer: 'agreement' },
      { question: 'What constitutes negligence?', expectedInAnswer: 'duty' },
      { question: 'Explain criminal procedure', expectedInAnswer: 'prosecution' }
    ];
    
    for (const exchange of exchanges) {
      await page.request.post('/api/rag/chat', {
        data: {
          message: exchange.question,
          conversation_id: conversationId,
          case_id: testCaseId,
          use_rag: true
        }
      });
    }
    
    // Export conversation
    const exportResponse = await page.request.get(`/api/rag/export-conversation/${conversationId}`);
    expect(exportResponse.status()).toBe(200);
    
    const exportData = await exportResponse.json();
    
    expect(exportData).toHaveProperty('conversation_id');
    expect(exportData).toHaveProperty('case_id');
    expect(exportData).toHaveProperty('messages');
    expect(exportData).toHaveProperty('sources_used');
    expect(exportData).toHaveProperty('metadata');
    
    // Verify export completeness
    expect(exportData.messages).toHaveLength(exchanges.length * 2); // Q&A pairs
    expect(exportData.sources_used.length).toBeGreaterThan(0);
    
    // Each message should have associated sources
    const assistantMessages = exportData.messages.filter((m: any) => m.role === 'assistant');
    assistantMessages.forEach((msg: any) => {
      expect(msg.sources).toBeDefined();
      expect(Array.isArray(msg.sources)).toBe(true);
    });
  });
});