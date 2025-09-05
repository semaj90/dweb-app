// tests/api/upload.spec.js
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test.describe('Document Upload API Tests', () => {
  let testFilePath;
  
  test.beforeAll(async () => {
    // Create a test document
    testFilePath = path.join(__dirname, '../../uploads/playwright-test.txt');
    const testContent = `
PLAYWRIGHT TEST LEGAL DOCUMENT

This is a test legal contract for automated testing purposes.

PARTIES:
- Party A: Test Company Inc.
- Party B: Test Individual

TERMS:
1. This is a test contract clause
2. Payment terms: Net 30 days
3. Jurisdiction: United States

SIGNATURES:
Test Company Inc. - Date: ${new Date().toISOString()}
Test Individual - Date: ${new Date().toISOString()}
    `.trim();
    
    fs.writeFileSync(testFilePath, testContent);
  });

  test.afterAll(async () => {
    // Cleanup test file
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  test('should successfully upload and process text document', async ({ request }) => {
    const response = await request.post('/api/upload', {
      multipart: {
        file: {
          name: 'playwright-test.txt',
          mimeType: 'text/plain',
          buffer: fs.readFileSync(testFilePath),
        },
        document_type: 'contract',
        case_id: 'PLAYWRIGHT-TEST-001',
        practice_area: 'contract_law',
        jurisdiction: 'US',
        enable_ocr: 'false',
        enable_embedding: 'true',
      }
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    
    // Validate response structure
    expect(result.document_id).toBeDefined();
    expect(result.original_name).toBe('playwright-test.txt');
    expect(result.file_type).toBe('text/plain');
    expect(result.extracted_text).toContain('PLAYWRIGHT TEST LEGAL DOCUMENT');
    expect(result.text_length).toBeGreaterThan(0);
    expect(result.chunks).toBeDefined();
    expect(Array.isArray(result.chunks)).toBeTruthy();
    expect(result.summary).toBeDefined();
    expect(result.key_points).toBeDefined();
    expect(Array.isArray(result.key_points)).toBeTruthy();
    expect(result.embeddings).toBeDefined();
    expect(Array.isArray(result.embeddings)).toBeTruthy();
    expect(result.metadata).toBeDefined();
    expect(result.performance).toBeDefined();
    
    // Validate metadata
    expect(result.metadata.document_type).toBe('contract');
    expect(result.metadata.case_id).toBe('PLAYWRIGHT-TEST-001');
    expect(result.metadata.practice_area).toBe('contract_law');
    expect(result.metadata.jurisdiction).toBe('US');
    expect(result.metadata.processor_version).toBe('2.0.0-gpu-simd');
    
    // Validate performance metrics
    expect(result.performance.concurrent_tasks).toBeGreaterThan(0);
    expect(result.performance.cpu_cores).toBeGreaterThan(0);
    expect(result.performance.simd_accelerated).toBe(true);
    expect(result.performance.gpu_accelerated).toBe(true);
    
    // Validate embeddings
    if (result.embeddings.length > 0) {
      expect(result.embeddings[0].dimension).toBe(384);
      expect(result.embeddings[0].model).toBe('nomic-embed-text');
      expect(Array.isArray(result.embeddings[0].embedding)).toBeTruthy();
      expect(result.embeddings[0].embedding.length).toBeGreaterThan(0); // Accept any valid embedding size
    }
  });

  test('should handle missing file error', async ({ request }) => {
    const response = await request.post('/api/upload', {
      multipart: {
        document_type: 'contract',
        case_id: 'TEST-NO-FILE',
      }
    });

    expect(response.status()).toBe(400);
    const result = await response.json();
    expect(result.error).toBeDefined();
  });

  test('should validate document types', async ({ request }) => {
    const response = await request.post('/api/upload', {
      multipart: {
        file: {
          name: 'test.txt',
          mimeType: 'text/plain',
          buffer: Buffer.from('test content'),
        },
        document_type: 'legal',
        enable_ocr: 'false',
        enable_embedding: 'false',
      }
    });

    expect(response.ok()).toBeTruthy();
    const result = await response.json();
    expect(result.metadata.document_type).toBe('legal');
  });
});