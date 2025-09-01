import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

test.describe('File Upload Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to file upload page
    await page.goto('/ai-assistant-demo');
    await page.waitForLoadState('networkidle');
  });

  test('file upload interface is accessible', async ({ page }) => {
    // Check that file upload component is present
    await expect(page.locator('[data-testid="file-upload"]')).toBeVisible({ timeout: 10000 });
    
    // Check for drag and drop area
    const dropZone = page.locator('.drag-drop-zone, [data-testid="drop-zone"]');
    if (await dropZone.count() > 0) {
      await expect(dropZone.first()).toBeVisible();
    }
    
    // Check for file input
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeAttached();
  });

  test('can select file through file picker', async ({ page }) => {
    // Create a test file
    const testFilePath = path.join(import.meta.url, 'test-files', 'sample.txt');
    const testDir = path.dirname(testFilePath);
    
    // Ensure test directory exists
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    // Create test file
    fs.writeFileSync(testFilePath, 'This is a test legal document for upload testing.');
    
    // Find file input
    const fileInput = page.locator('input[type="file"]');
    
    // Upload file
    await fileInput.setInputFiles(testFilePath);
    
    // Check that file is selected/displayed
    await expect(page.locator('text=sample.txt')).toBeVisible({ timeout: 5000 });
    
    // Clean up
    fs.unlinkSync(testFilePath);
  });

  test('displays upload progress', async ({ page }) => {
    // Create test file
    const testFilePath = path.join(import.meta.url, 'test-files', 'progress-test.pdf');
    const testDir = path.dirname(testFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    // Create a larger test file to see progress
    const largeContent = 'PDF content '.repeat(10000);
    fs.writeFileSync(testFilePath, largeContent);
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    // Look for progress indicators
    const progressBar = page.locator('.progress-bar, [data-testid="progress"], .upload-progress');
    const progressText = page.locator('text=/\d+%/, text=Uploading, text=Processing');
    
    // Should show progress elements (at least one)
    const hasProgress = await Promise.race([
      progressBar.first().waitFor({ timeout: 3000 }).then(() => true).catch(() => false),
      progressText.first().waitFor({ timeout: 3000 }).then(() => true).catch(() => false)
    ]);
    
    expect(hasProgress).toBe(true);
    
    fs.unlinkSync(testFilePath);
  });

  test('handles AI processing workflow', async ({ page }) => {
    // Mock the upload and AI processing
    await page.route('**/api/upload', async route => {
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate processing time
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          documentId: 'test-doc-123',
          filename: 'test.pdf',
          aiProcessing: true
        })
      });
    });
    
    await page.route('**/api/ai/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          analysis: 'This is a legal contract with standard terms.',
          confidence: 0.95,
          entities: ['Contract', 'Legal Agreement'],
          embedding: new Array(768).fill(0.1) // Mock embedding
        })
      });
    });
    
    const testFilePath = path.join(import.meta.url, 'test-files', 'ai-test.txt');
    const testDir = path.dirname(testFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    fs.writeFileSync(testFilePath, 'Legal contract content for AI analysis testing.');
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    // Look for AI processing indicators
    await expect(page.locator('text=AI Processing, text=Analyzing, text=Processing')).toBeVisible({ timeout: 10000 });
    
    // Wait for completion
    await expect(page.locator('text=Complete, text=Done, text=Analysis Complete')).toBeVisible({ timeout: 15000 });
    
    fs.unlinkSync(testFilePath);
  });

  test('shows error handling for invalid files', async ({ page }) => {
    const invalidFilePath = path.join(import.meta.url, 'test-files', 'invalid.xyz');
    const testDir = path.dirname(invalidFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    fs.writeFileSync(invalidFilePath, 'Invalid file type content');
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(invalidFilePath);
    
    // Should show error message
    await expect(page.locator('text=/not supported|invalid|error/i')).toBeVisible({ timeout: 5000 });
    
    fs.unlinkSync(invalidFilePath);
  });

  test('drag and drop functionality', async ({ page }) => {
    const dropZone = page.locator('.drag-drop-zone, [data-testid="drop-zone"]');
    
    if (await dropZone.count() === 0) {
      test.skip(true, 'Drag and drop zone not found - component may not support D&D');
    }
    
    // Create test file
    const testFilePath = path.join(import.meta.url, 'test-files', 'drag-test.txt');
    const testDir = path.dirname(testFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    fs.writeFileSync(testFilePath, 'Drag and drop test content');
    
    // Simulate drag and drop
    const fileBuffer = fs.readFileSync(testFilePath);
    
    await dropZone.first().dispatchEvent('drop', {
      dataTransfer: {
        files: [{
          name: 'drag-test.txt',
          type: 'text/plain',
          size: fileBuffer.length,
          stream: () => new ReadableStream({
            start(controller) {
              controller.enqueue(fileBuffer);
              controller.close();
            }
          })
        }]
      }
    });
    
    // Should show file was dropped
    await expect(page.locator('text=drag-test.txt')).toBeVisible({ timeout: 5000 });
    
    fs.unlinkSync(testFilePath);
  });

  test('WebSocket/SSE progress updates', async ({ page }) => {
    // Mock WebSocket/SSE connections for real-time updates
    await page.route('**/api/evidence/stream/**', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: 'data: {"type":"upload-progress","progress":50}\n\ndata: {"type":"processing-step","step":"AI Analysis"}\n\ndata: {"type":"processing-complete","result":{"success":true}}\n\n'
      });
    });
    
    const testFilePath = path.join(import.meta.url, 'test-files', 'stream-test.txt');
    const testDir = path.dirname(testFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    fs.writeFileSync(testFilePath, 'Stream test content for real-time updates.');
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(testFilePath);
    
    // Should show real-time updates
    await expect(page.locator('text=50%')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=AI Analysis')).toBeVisible({ timeout: 8000 });
    
    fs.unlinkSync(testFilePath);
  });

  test('file size limits are enforced', async ({ page }) => {
    // Create a large test file
    const largeFilePath = path.join(import.meta.url, 'test-files', 'large.txt');
    const testDir = path.dirname(largeFilePath);
    
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    // Create 20MB file (assuming 10MB limit)
    const largeContent = 'A'.repeat(20 * 1024 * 1024);
    fs.writeFileSync(largeFilePath, largeContent);
    
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(largeFilePath);
    
    // Should show file size error
    await expect(page.locator('text=/too large|size limit|exceeds/i')).toBeVisible({ timeout: 5000 });
    
    fs.unlinkSync(largeFilePath);
  });

  test('multiple file upload support', async ({ page }) => {
    // Check if multiple file upload is supported
    const fileInput = page.locator('input[type="file"]');
    const hasMultiple = await fileInput.getAttribute('multiple');
    
    if (!hasMultiple) {
      test.skip(true, 'Multiple file upload not supported by component');
    }
    
    // Create multiple test files
    const testDir = path.join(import.meta.url, 'test-files');
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    
    const file1Path = path.join(testDir, 'multi1.txt');
    const file2Path = path.join(testDir, 'multi2.txt');
    
    fs.writeFileSync(file1Path, 'First file content');
    fs.writeFileSync(file2Path, 'Second file content');
    
    await fileInput.setInputFiles([file1Path, file2Path]);
    
    // Should show both files
    await expect(page.locator('text=multi1.txt')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('text=multi2.txt')).toBeVisible({ timeout: 5000 });
    
    fs.unlinkSync(file1Path);
    fs.unlinkSync(file2Path);
  });

  test.afterEach(async () => {
    // Clean up test files directory
    const testDir = path.join(import.meta.url, 'test-files');
    if (fs.existsSync(testDir)) {
      try {
        fs.rmSync(testDir, { recursive: true, force: true });
      } catch (error: any) {
        console.warn('Could not clean up test files:', error);
      }
    }
  });
});