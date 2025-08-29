import { test, expect } from '@playwright/test';
import { randomUUID } from 'crypto';

/**
 * Comprehensive RAG Integration Flow Test
 * 
 * Tests the complete pipeline:
 * 1. Document upload via MinIO pre-signed URLs
 * 2. Redis job queue processing
 * 3. OCR + NLP + embedding generation
 * 4. Vector storage in PostgreSQL + pgvector
 * 5. RAG queries with Gemma3 LLM
 * 6. Real-time status updates
 */

test.describe('RAG Integration Full Pipeline', () => {
  let testCaseId: string;
  let testFileId: string;
  let testJobId: string;

  test.beforeEach(async ({ page }) => {
    testCaseId = randomUUID();
    
    // Navigate to RAG demo page
    await page.goto('/demo/rag-integration');
    await expect(page).toHaveTitle(/RAG Integration Demo/);
    
    // Wait for page to initialize with demo case
    await expect(page.locator('[data-testid="demo-case-id"]')).toBeVisible();
  });

  test('Complete document ingestion and RAG query pipeline', async ({ page }) => {
    // Step 1: Test file upload flow
    test.step('File Upload', async () => {
      // Prepare test file
      const testContent = `
        LEGAL DOCUMENT - CONTRACT ANALYSIS
        
        This is a sample legal document for testing the RAG integration pipeline.
        
        PARTIES:
        - Client: ACME Corporation
        - Attorney: John Doe, Esq.
        - Matter: Contract Review #2024-001
        
        TERMS:
        - Payment: $50,000 retainer
        - Duration: 6 months
        - Scope: Commercial contract analysis and negotiation
        
        OBLIGATIONS:
        1. Attorney shall review all contract provisions
        2. Client shall provide all relevant documentation
        3. Both parties shall maintain confidentiality
        
        GOVERNING LAW: This agreement shall be governed by California law.
      `;
      
      // Create test file
      const testFile = new File([testContent], 'test-contract.txt', { type: 'text/plain' });
      
      // Upload file
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles({
        name: 'test-contract.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from(testContent)
      });
      
      // Verify upload progress
      await expect(page.locator('[data-testid="upload-status"]')).toContainText('uploading');
      await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
      
      // Wait for MinIO upload completion
      await expect(page.locator('[data-testid="upload-status"]')).toContainText('processing', { timeout: 30000 });
      
      // Extract file ID for later use
      const fileIdElement = page.locator('[data-testid="file-id"]');
      testFileId = await fileIdElement.textContent() || '';
      expect(testFileId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    });

    // Step 2: Test job queue processing
    test.step('Job Queue Processing', async () => {
      // Verify job creation
      await expect(page.locator('[data-testid="job-id"]')).toBeVisible();
      testJobId = await page.locator('[data-testid="job-id"]').textContent() || '';
      
      // Monitor processing steps
      const processingSteps = [
        'download',
        'ocr',
        'nlp', 
        'chunk',
        'embed',
        'store',
        'index',
        'notify'
      ];
      
      for (const step of processingSteps) {
        await expect(page.locator(`[data-testid="current-step"]`)).toContainText(step, { timeout: 10000 });
        await expect(page.locator('[data-testid="progress-percentage"]')).not.toContainText('0%');
      }
      
      // Wait for completion
      await expect(page.locator('[data-testid="upload-status"]')).toContainText('completed', { timeout: 60000 });
      await expect(page.locator('[data-testid="progress-percentage"]')).toContainText('100%');
      
      // Verify processing results
      await expect(page.locator('[data-testid="extracted-text-length"]')).not.toContainText('0');
      await expect(page.locator('[data-testid="embeddings-count"]')).not.toContainText('0');
    });

    // Step 3: Test RAG query functionality
    test.step('RAG Query Processing', async () => {
      // Test query 1: Entity extraction
      const query1 = "Who are the parties involved in this contract?";
      await page.fill('[data-testid="rag-query-input"]', query1);
      await page.click('[data-testid="submit-query"]');
      
      // Wait for response
      await expect(page.locator('[data-testid="rag-response"]')).toBeVisible({ timeout: 30000 });
      await expect(page.locator('[data-testid="rag-answer"]')).toContainText('ACME Corporation');
      await expect(page.locator('[data-testid="rag-answer"]')).toContainText('John Doe');
      
      // Verify sources
      await expect(page.locator('[data-testid="rag-sources"]')).toBeVisible();
      await expect(page.locator('[data-testid="source-similarity"]')).not.toContainText('0%');
      
      // Test query 2: Financial terms
      const query2 = "What is the payment amount mentioned in the contract?";
      await page.fill('[data-testid="rag-query-input"]', query2);
      await page.click('[data-testid="submit-query"]');
      
      await expect(page.locator('[data-testid="rag-answer"]')).toContainText('$50,000', { timeout: 30000 });
      
      // Test query 3: Legal jurisdiction
      const query3 = "What law governs this agreement?";
      await page.fill('[data-testid="rag-query-input"]', query3);
      await page.click('[data-testid="submit-query"]');
      
      await expect(page.locator('[data-testid="rag-answer"]')).toContainText('California', { timeout: 30000 });
      
      // Verify metadata
      await expect(page.locator('[data-testid="model-name"]')).toContainText('gemma3-legal');
      await expect(page.locator('[data-testid="processing-time"]')).not.toContainText('0ms');
      await expect(page.locator('[data-testid="tokens-used"]')).not.toContainText('0');
    });

    // Step 4: Test error handling
    test.step('Error Handling', async () => {
      // Test with empty query
      await page.fill('[data-testid="rag-query-input"]', '');
      await page.click('[data-testid="submit-query"]');
      await expect(page.locator('[data-testid="query-error"]')).toBeVisible();
      
      // Test with very long query
      const longQuery = 'x'.repeat(3000);
      await page.fill('[data-testid="rag-query-input"]', longQuery);
      await page.click('[data-testid="submit-query"]');
      await expect(page.locator('[data-testid="query-error"]')).toContainText('too long');
    });
  });

  test('API endpoint integration tests', async ({ request }) => {
    // Test case creation
    const caseResponse = await request.post('/api/v1/cases', {
      data: {
        id: testCaseId,
        title: 'RAG Integration Test Case',
        description: 'Automated test case for RAG pipeline',
        status: 'active'
      }
    });
    expect(caseResponse.ok()).toBeTruthy();
    
    // Test pre-signed URL generation
    const presignedResponse = await request.post('/api/v1/upload/presigned', {
      data: {
        filename: 'test-document.txt',
        contentType: 'text/plain',
        caseId: testCaseId
      }
    });
    expect(presignedResponse.ok()).toBeTruthy();
    
    const presignedData = await presignedResponse.json();
    expect(presignedData.fileId).toBeDefined();
    expect(presignedData.uploadUrl).toBeDefined();
    testFileId = presignedData.fileId;
    
    // Test job creation
    const jobResponse = await request.post('/api/v1/jobs/ingest', {
      data: {
        fileId: testFileId,
        caseId: testCaseId,
        filename: 'test-document.txt',
        contentType: 'text/plain',
        priority: 'high'
      }
    });
    expect(jobResponse.ok()).toBeTruthy();
    
    const jobData = await jobResponse.json();
    expect(jobData.jobId).toBeDefined();
    testJobId = jobData.jobId;
    
    // Test job status polling
    let attempts = 0;
    let jobCompleted = false;
    
    while (attempts < 30 && !jobCompleted) {
      const statusResponse = await request.get(`/api/v1/jobs/${testJobId}`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const statusData = await statusResponse.json();
      jobCompleted = statusData.status === 'completed' || statusData.status === 'failed';
      
      if (!jobCompleted) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        attempts++;
      }
    }
    
    // Test AI query (mock since we don't have actual processed data)
    const queryResponse = await request.post('/api/v1/ai/query', {
      data: {
        query: 'What is this document about?',
        caseId: testCaseId,
        maxResults: 5,
        minSimilarity: 0.5
      }
    });
    
    // Should return response even if no matching documents
    expect(queryResponse.ok()).toBeTruthy();
  });

  test('Performance benchmarks', async ({ page }) => {
    // Test upload performance
    const uploadStartTime = Date.now();
    
    const smallTestContent = 'Small test document for performance testing.';
    await page.locator('input[type="file"]').setInputFiles({
      name: 'small-test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(smallTestContent)
    });
    
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('processing', { timeout: 10000 });
    const uploadTime = Date.now() - uploadStartTime;
    
    // Upload should complete within 10 seconds
    expect(uploadTime).toBeLessThan(10000);
    
    // Test query performance
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('completed', { timeout: 30000 });
    
    const queryStartTime = Date.now();
    await page.fill('[data-testid="rag-query-input"]', 'What is this document about?');
    await page.click('[data-testid="submit-query"]');
    
    await expect(page.locator('[data-testid="rag-response"]')).toBeVisible({ timeout: 15000 });
    const queryTime = Date.now() - queryStartTime;
    
    // Query should complete within 15 seconds
    expect(queryTime).toBeLessThan(15000);
    
    // Verify processing time is logged
    const processingTimeElement = page.locator('[data-testid="processing-time"]');
    const processingTime = await processingTimeElement.textContent();
    expect(processingTime).toMatch(/\d+ms/);
  });

  test('Real-time updates via WebSocket', async ({ page }) => {
    // Start upload
    const testContent = 'WebSocket test document content.';
    await page.locator('input[type="file"]').setInputFiles({
      name: 'websocket-test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(testContent)
    });
    
    // Track status changes
    const statusChanges: string[] = [];
    
    await page.locator('[data-testid="upload-status"]').evaluate((element) => {
      const observer = new MutationObserver(() => {
        (window as any).statusChanges = (window as any).statusChanges || [];
        (window as any).statusChanges.push(element.textContent);
      });
      observer.observe(element, { childList: true, subtree: true });
    });
    
    // Wait for completion
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('completed', { timeout: 60000 });
    
    // Verify we received multiple status updates
    const finalStatusChanges = await page.evaluate(() => (window as any).statusChanges || []);
    expect(finalStatusChanges.length).toBeGreaterThan(2);
    expect(finalStatusChanges).toContain('uploading');
    expect(finalStatusChanges).toContain('processing');
    expect(finalStatusChanges).toContain('completed');
  });

  test.afterEach(async ({ request }) => {
    // Cleanup test data
    try {
      if (testJobId) {
        await request.delete(`/api/v1/jobs/${testJobId}`);
      }
      
      if (testCaseId) {
        await request.delete(`/api/v1/cases?id=${testCaseId}&hard=true`);
      }
    } catch (error: any) {
      console.warn('Cleanup error:', error);
    }
  });
});

test.describe('RAG Integration Error Scenarios', () => {
  test('handles service failures gracefully', async ({ page }) => {
    await page.goto('/demo/rag-integration');
    
    // Test with unavailable service (mock network failure)
    await page.route('/api/v1/upload/presigned', route => {
      route.fulfill({ status: 500, body: JSON.stringify({ error: 'Service unavailable' }) });
    });
    
    const testContent = 'Error test document';
    await page.locator('input[type="file"]').setInputFiles({
      name: 'error-test.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(testContent)
    });
    
    await expect(page.locator('[data-testid="upload-status"]')).toContainText('error', { timeout: 10000 });
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
  });

  test('validates file types and sizes', async ({ page }) => {
    await page.goto('/demo/rag-integration');
    
    // Test invalid file type
    const invalidContent = 'Invalid file content';
    await page.locator('input[type="file"]').setInputFiles({
      name: 'invalid.xyz',
      mimeType: 'application/unknown',
      buffer: Buffer.from(invalidContent)
    });
    
    await expect(page.locator('[data-testid="file-error"]')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('[data-testid="file-error"]')).toContainText('not supported');
  });

  test('handles timeout scenarios', async ({ page }) => {
    await page.goto('/demo/rag-integration');
    
    // Mock slow processing
    await page.route('/api/v1/jobs/**', route => {
      if (route.request().method() === 'GET') {
        setTimeout(() => {
          route.fulfill({
            status: 200,
            body: JSON.stringify({ status: 'processing', progress: 50 })
          });
        }, 5000);
      } else {
        route.continue();
      }
    });
    
    // Start upload and verify timeout handling
    const testContent = 'Timeout test document';
    await page.locator('input[type="file"]').setInputFiles({
      name: 'timeout-test.txt',
      mimeType: 'text/plain',  
      buffer: Buffer.from(testContent)
    });
    
    // Should show timeout warning after extended processing
    await expect(page.locator('[data-testid="processing-warning"]')).toBeVisible({ timeout: 35000 });
  });
});