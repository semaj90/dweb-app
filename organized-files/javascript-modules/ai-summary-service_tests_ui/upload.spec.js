// tests/ui/upload.spec.js
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test.describe('Document Upload UI Tests', () => {
  let testFilePath;
  
  test.beforeAll(async () => {
    // Create a test document
    testFilePath = path.join(__dirname, '../../uploads/ui-test-document.txt');
    const testContent = `
UI TEST LEGAL CONTRACT

This is a test legal contract for UI automation testing.

PARTIES:
- Playwright Test Company
- Automated Test User

CONTRACT TERMS:
1. Test automation clause
2. UI validation requirements
3. Browser compatibility testing

EFFECTIVE DATE: ${new Date().toDateString()}
    `.trim();
    
    fs.writeFileSync(testFilePath, testContent);
  });

  test.afterAll(async () => {
    // Cleanup test file
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }
  });

  test('should load the test interface correctly', async ({ page }) => {
    await page.goto('/test');
    
    // Check page title
    await expect(page).toHaveTitle(/AI Document Processing Test Interface/);
    
    // Check main heading
    await expect(page.locator('h1')).toContainText('AI Document Processing Test Interface');
    
    // Check tabs are present
    await expect(page.locator('.tab')).toHaveCount(3);
    await expect(page.locator('.tab').first()).toContainText('Single Upload');
    await expect(page.locator('.tab').nth(1)).toContainText('Batch Upload');
    await expect(page.locator('.tab').nth(2)).toContainText('Health Check');
  });

  test('should switch between tabs correctly', async ({ page }) => {
    await page.goto('/test');
    
    // Initially, Single Upload tab should be active
    await expect(page.locator('#upload-tab')).toHaveClass(/active/);
    await expect(page.locator('#batch-tab')).not.toHaveClass(/active/);
    await expect(page.locator('#health-tab')).not.toHaveClass(/active/);
    
    // Click on Batch Upload tab
    await page.locator('.tab').nth(1).click();
    await expect(page.locator('#batch-tab')).toHaveClass(/active/);
    await expect(page.locator('#upload-tab')).not.toHaveClass(/active/);
    
    // Click on Health Check tab
    await page.locator('.tab').nth(2).click();
    await expect(page.locator('#health-tab')).toHaveClass(/active/);
    await expect(page.locator('#upload-tab')).not.toHaveClass(/active/);
  });

  test('should perform health check successfully', async ({ page }) => {
    await page.goto('/test');
    
    // Switch to Health Check tab
    await page.locator('.tab').nth(2).click();
    
    // Click health check button
    await page.locator('button:has-text("Check System Health")').click();
    
    // Wait for results
    await page.waitForSelector('#healthResults .status-indicator', { timeout: 10000 });
    
    // Check that health indicators are present
    const statusIndicators = page.locator('#healthResults .status-indicator');
    await expect(statusIndicators).toHaveCount(3); // Expect 3 status indicators
    
    // Check JSON output is displayed
    await expect(page.locator('#healthResults .json-output')).toBeVisible();
  });

  test('should upload and process document successfully', async ({ page }) => {
    await page.goto('/test');
    
    // Fill out the form
    await page.locator('#file').setInputFiles(testFilePath);
    await page.locator('#document_type').selectOption('contract');
    await page.locator('#case_id').fill('UI-TEST-2025-001');
    await page.locator('#practice_area').selectOption('contract_law');
    await page.locator('#jurisdiction').fill('US');
    
    // Ensure checkboxes are checked
    await page.locator('#enable_ocr').check();
    await page.locator('#enable_embedding').check();
    
    // Submit the form
    await page.locator('#uploadBtn').click();
    
    // Wait for processing to complete (up to 2 minutes)
    await page.waitForSelector('#results.show', { timeout: 120000 });
    
    // Check for success message
    await expect(page.locator('.success')).toBeVisible();
    await expect(page.locator('.success')).toContainText('Document processed successfully');
    
    // Check JSON output is displayed
    await expect(page.locator('.json-output')).toBeVisible();
    
    // Verify the JSON contains expected fields
    const jsonText = await page.locator('.json-output').textContent();
    expect(jsonText).toContain('document_id');
    expect(jsonText).toContain('original_name');
    expect(jsonText).toContain('extracted_text');
    expect(jsonText).toContain('summary');
    expect(jsonText).toContain('embeddings');
    expect(jsonText).toContain('performance');
  });

  test('should handle form validation errors', async ({ page }) => {
    await page.goto('/test');
    
    // Try to submit without selecting a file
    await page.locator('#uploadBtn').click();
    
    // Check for browser validation (file input required)
    const fileInput = page.locator('#file');
    const validationMessage = await fileInput.evaluate(el => el.validationMessage);
    expect(validationMessage).toBeTruthy();
  });

  test('should show loading state during processing', async ({ page }) => {
    await page.goto('/test');
    
    // Fill out the form
    await page.locator('#file').setInputFiles(testFilePath);
    await page.locator('#document_type').selectOption('legal');
    
    // Submit the form
    await page.locator('#uploadBtn').click();
    
    // Check loading state
    await expect(page.locator('#uploadBtn')).toBeDisabled();
    await expect(page.locator('#uploadBtn')).toContainText('Processing...');
    await expect(page.locator('.loading')).toBeVisible();
    
    // Wait for completion
    await page.waitForSelector('#results.show', { timeout: 120000 });
    
    // Check button returns to normal state
    await expect(page.locator('#uploadBtn')).toBeEnabled();
    await expect(page.locator('#uploadBtn')).toContainText('Process Document with AI');
  });

  test('should test batch upload interface', async ({ page }) => {
    await page.goto('/test');
    
    // Switch to Batch Upload tab
    await page.locator('.tab').nth(1).click();
    
    // Check batch upload form elements
    await expect(page.locator('#batch_files')).toBeVisible();
    await expect(page.locator('#batch_document_type')).toBeVisible();
    await expect(page.locator('#batch_enable_ocr')).toBeVisible();
    await expect(page.locator('#batch_enable_embedding')).toBeVisible();
    await expect(page.locator('#batchBtn')).toBeVisible();
    
    // Verify multiple attribute on file input
    const multipleAttr = await page.locator('#batch_files').getAttribute('multiple');
    expect(multipleAttr).toBeDefined();
  });

  test('should handle network errors gracefully', async ({ page }) => {
    await page.goto('/test');
    
    // Mock a network error
    await page.route('/api/upload', route => {
      route.abort('internetdisconnected');
    });
    
    // Fill out and submit form
    await page.locator('#file').setInputFiles(testFilePath);
    await page.locator('#uploadBtn').click();
    
    // Wait for error message
    await page.waitForSelector('.error', { timeout: 10000 });
    
    // Check error is displayed
    await expect(page.locator('.error')).toBeVisible();
    await expect(page.locator('.error')).toContainText('Network Error');
  });

  test('should display processing performance metrics', async ({ page }) => {
    await page.goto('/test');
    
    // Fill out the form
    await page.locator('#file').setInputFiles(testFilePath);
    await page.locator('#document_type').selectOption('contract');
    await page.locator('#enable_embedding').check();
    
    // Submit the form
    await page.locator('#uploadBtn').click();
    
    // Wait for processing to complete
    await page.waitForSelector('#results.show', { timeout: 120000 });
    
    // Check that performance metrics are included in JSON output
    const jsonText = await page.locator('.json-output').textContent();
    expect(jsonText).toContain('performance');
    expect(jsonText).toContain('concurrent_tasks');
    expect(jsonText).toContain('cpu_cores');
    expect(jsonText).toContain('simd_accelerated');
    expect(jsonText).toContain('gpu_accelerated');
  });
});