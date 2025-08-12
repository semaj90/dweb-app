// tests/ui/document-simulator.spec.js
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test.describe('Document Upload Simulator Tests', () => {
  let testFilePath;
  
  test.beforeAll(async () => {
    // Create a test document
    testFilePath = path.join(__dirname, '../../uploads/simulator-test.txt');
    const testContent = `
SIMULATOR TEST LEGAL DOCUMENT

This is a test document for the AI Document Processing Simulator.

CASE DETAILS:
- Case ID: SIM-TEST-001
- Type: Contract Dispute
- Priority: High

CONTENT:
This document contains legal text that will be processed through the complete AI pipeline:
1. OCR extraction (for PDFs)
2. AI summarization using Ollama
3. Vector embedding generation using Nomic-Embed-Text
4. Storage in PostgreSQL and local cache

EXPECTED PROCESSING:
- Text extraction should capture all content
- AI summary should identify key legal concepts
- Embeddings should be 768-dimensional vectors
- Files under 10MB should be cached locally

END OF TEST DOCUMENT
    `.trim();
    
    // Ensure uploads directory exists
    const uploadsDir = path.dirname(testFilePath);
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }
    
    fs.writeFileSync(testFilePath, testContent);
  });

  test.afterAll(async () => {
    // Cleanup test file
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  test('should load the document simulator interface', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Check page loads correctly
    await expect(page).toHaveTitle(/AI Document Processing Demo/);
    
    // Check main heading
    await expect(page.locator('h1')).toContainText('AI Document Processing Demo');
    
    // Check upload simulator is present
    await expect(page.locator('.document-upload-simulator')).toBeVisible();
    
    // Check upload area
    await expect(page.locator('.upload-area')).toBeVisible();
    await expect(page.locator('.upload-area')).toContainText('Drop PDFs or text files here');
  });

  test('should display system status correctly', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Check system status section
    await expect(page.locator('h2:has-text("System Status")')).toBeVisible();
    
    // Check service status indicators
    await expect(page.locator('text=Ollama LLM')).toBeVisible();
    await expect(page.locator('text=PostgreSQL')).toBeVisible();
    await expect(page.locator('text=AI Summarizer')).toBeVisible();
    await expect(page.locator('text=Embeddings')).toBeVisible();
    
    // Test refresh status button
    await page.locator('button:has-text("Refresh Status")').click();
    await page.waitForTimeout(1000); // Allow status check to complete
  });

  test('should upload and process document successfully', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Wait for page to be ready
    await page.waitForSelector('.upload-area');
    
    // Upload test file
    await page.locator('input[type="file"]').setInputFiles(testFilePath);
    
    // Wait for upload item to appear
    await page.waitForSelector('.upload-item', { timeout: 5000 });
    
    // Check upload item is displayed
    await expect(page.locator('.upload-item')).toBeVisible();
    await expect(page.locator('.upload-item')).toContainText('simulator-test.txt');
    
    // Check file size is displayed
    await expect(page.locator('.upload-item')).toContainText('KB');
    
    // Check local storage indicator (files under 10MB)
    await expect(page.locator('.upload-item')).toContainText('Local Storage');
    
    // Wait for processing to complete (up to 2 minutes)
    await page.waitForSelector('text=Completed ✅', { timeout: 120000 });
    
    // Verify processing stages completed
    await expect(page.locator('.upload-item')).toContainText('Completed ✅');
    
    // Check extracted text is displayed
    await expect(page.locator('h4:has-text("Extracted Text")')).toBeVisible();
    await expect(page.locator('.bg-gray-900')).toContainText('SIMULATOR TEST LEGAL DOCUMENT');
    
    // Check AI summary is displayed
    await expect(page.locator('h4:has-text("AI Summary")')).toBeVisible();
    
    // Check embeddings info is displayed
    await expect(page.locator('h4:has-text("Vector Embeddings")')).toBeVisible();
    await expect(page.locator('text=Generated')).toBeVisible();
    await expect(page.locator('text=Nomic-Embed-Text')).toBeVisible();
    
    // Check download button is present
    await expect(page.locator('button:has-text("Download JSON")')).toBeVisible();
    
    // Check local cache indicator
    await expect(page.locator('text=Cached Locally')).toBeVisible();
  });

  test('should handle file drag and drop', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Create a DataTransfer object for drag and drop
    const fileContent = 'Test drag and drop content';
    
    // Simulate drag enter (should highlight upload area)
    await page.locator('.upload-area').dispatchEvent('dragenter');
    
    // Upload area should show drag state (visual feedback)
    // Note: Visual state changes might not be easily testable, but we can check the upload still works
    
    // Use file input as fallback test (drag/drop simulation is complex in Playwright)
    await page.locator('input[type="file"]').setInputFiles(testFilePath);
    await page.waitForSelector('.upload-item', { timeout: 5000 });
    await expect(page.locator('.upload-item')).toBeVisible();
  });

  test('should allow removing uploaded documents', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Upload a test file
    await page.locator('input[type="file"]').setInputFiles(testFilePath);
    await page.waitForSelector('.upload-item', { timeout: 5000 });
    
    // Check upload item exists
    await expect(page.locator('.upload-item')).toBeVisible();
    
    // Click remove button (X button)
    await page.locator('.upload-item button:has-text("✕")').click();
    
    // Upload item should be removed
    await expect(page.locator('.upload-item')).not.toBeVisible();
    
    // Should show empty state
    await expect(page.locator('text=No documents uploaded yet')).toBeVisible();
  });

  test('should show processing progress correctly', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Upload test file
    await page.locator('input[type="file"]').setInputFiles(testFilePath);
    await page.waitForSelector('.upload-item', { timeout: 5000 });
    
    // Check initial status
    await expect(page.locator('.upload-item')).toContainText('Uploading to PostgreSQL');
    
    // Check progress bar is present and progressing
    await expect(page.locator('.upload-item .w-full.bg-gray-700')).toBeVisible();
    
    // Wait for different processing stages
    await page.waitForSelector('text=AI Processing', { timeout: 30000 });
    await expect(page.locator('.upload-item')).toContainText('AI Processing');
    
    await page.waitForSelector('text=Generating Embeddings', { timeout: 60000 });
    await expect(page.locator('.upload-item')).toContainText('Generating Embeddings');
    
    // Final completion
    await page.waitForSelector('text=Completed ✅', { timeout: 120000 });
    await expect(page.locator('.upload-item')).toContainText('Completed ✅');
  });

  test('should test download functionality', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Upload and wait for processing
    await page.locator('input[type="file"]').setInputFiles(testFilePath);
    await page.waitForSelector('text=Completed ✅', { timeout: 120000 });
    
    // Set up download promise before clicking
    const downloadPromise = page.waitForEvent('download');
    
    // Click download button
    await page.locator('button:has-text("Download JSON")').click();
    
    // Wait for download to complete
    const download = await downloadPromise;
    
    // Check download filename
    expect(download.suggestedFilename()).toContain('simulator-test.txt_processed.json');
    
    // Optionally save and verify download content
    const downloadPath = path.join(__dirname, '../../downloads', download.suggestedFilename());
    await download.saveAs(downloadPath);
    
    // Verify file was downloaded
    expect(fs.existsSync(downloadPath)).toBeTruthy();
    
    // Clean up download
    if (fs.existsSync(downloadPath)) {
      fs.unlinkSync(downloadPath);
    }
  });

  test('should display processing pipeline information', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Check processing pipeline section
    await expect(page.locator('h2:has-text("Processing Pipeline")')).toBeVisible();
    
    // Check all pipeline stages are displayed
    await expect(page.locator('text=1. Upload & OCR')).toBeVisible();
    await expect(page.locator('text=2. AI Summary')).toBeVisible();
    await expect(page.locator('text=3. Embeddings')).toBeVisible();
    await expect(page.locator('text=4. Storage')).toBeVisible();
    
    // Check technology mentions
    await expect(page.locator('text=PDF → Text extraction')).toBeVisible();
    await expect(page.locator('text=Ollama + Go-Llama')).toBeVisible();
    await expect(page.locator('text=Nomic-Embed-Text')).toBeVisible();
    await expect(page.locator('text=PostgreSQL + Local')).toBeVisible();
  });

  test('should handle multiple file uploads', async ({ page }) => {
    await page.goto('/demo/document-ai');
    
    // Create a second test file
    const secondTestFile = path.join(__dirname, '../../uploads/simulator-test-2.txt');
    fs.writeFileSync(secondTestFile, 'Second test document content');
    
    try {
      // Upload first file
      await page.locator('input[type="file"]').setInputFiles(testFilePath);
      await page.waitForSelector('.upload-item', { timeout: 5000 });
      
      // Upload second file
      await page.locator('input[type="file"]').setInputFiles(secondTestFile);
      
      // Should have two upload items
      await expect(page.locator('.upload-item')).toHaveCount(2);
      
      // Check both files are listed
      await expect(page.locator('text=simulator-test.txt')).toBeVisible();
      await expect(page.locator('text=simulator-test-2.txt')).toBeVisible();
      
    } finally {
      // Clean up second test file
      if (fs.existsSync(secondTestFile)) {
        fs.unlinkSync(secondTestFile);
      }
    }
  });
});