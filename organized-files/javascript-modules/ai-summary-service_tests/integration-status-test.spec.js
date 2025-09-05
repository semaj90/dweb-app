// tests/integration-status-test.spec.js
const { test, expect } = require('@playwright/test');

test.describe('Integration Test HTML - Ollama Status Detection', () => {
  
  test('should correctly detect Ollama status via Go service health endpoint', async ({ page }) => {
    // Navigate to the integration test HTML page
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for the page to load and status checks to run
    await page.waitForTimeout(3000);
    
    // Check that all service status elements are present
    await expect(page.locator('#sveltekit-status')).toBeVisible();
    await expect(page.locator('#go-service-status')).toBeVisible();
    await expect(page.locator('#health-status')).toBeVisible();
    await expect(page.locator('#ollama-status')).toBeVisible();
    
    // Wait for status updates to complete
    await page.waitForTimeout(5000);
    
    // Test Ollama status detection specifically
    const ollamaStatus = await page.locator('#ollama-status').textContent();
    console.log('Ollama Status Detected:', ollamaStatus);
    
    // Ollama should be detected as running since we know it's operational
    expect(ollamaStatus).toMatch(/Running|healthy/i);
    
    // Verify the status has the correct CSS class
    const ollamaStatusClass = await page.locator('#ollama-status').getAttribute('class');
    expect(ollamaStatusClass).toContain('healthy');
  });
  
  test('should verify Go service integration with Ollama', async ({ page }) => {
    // Navigate to the integration test page
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for initial load
    await page.waitForTimeout(2000);
    
    // Click "Check All Services" button to refresh status
    await page.locator('button:has-text("Check All Services")').click();
    
    // Wait for status updates
    await page.waitForTimeout(5000);
    
    // Check Go service status
    const goServiceStatus = await page.locator('#go-service-status').textContent();
    expect(goServiceStatus).toMatch(/Running|healthy/i);
    
    // Check health check status
    const healthStatus = await page.locator('#health-status').textContent();
    expect(healthStatus).toMatch(/Running|healthy/i);
    
    // Check that Ollama is correctly detected via Go service
    const ollamaStatus = await page.locator('#ollama-status').textContent();
    expect(ollamaStatus).toMatch(/Running|healthy/i);
  });
  
  test('should test API endpoints through integration interface', async ({ page }) => {
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for page load
    await page.waitForTimeout(2000);
    
    // Test Health API
    await page.locator('button:has-text("Test Health API")').click();
    await page.waitForTimeout(3000);
    
    // Check if API results container becomes visible
    const apiResults = page.locator('#apiResults');
    await expect(apiResults).toBeVisible();
    
    // Check if JSON output contains expected health data
    const apiOutput = await page.locator('#apiOutput').textContent();
    expect(apiOutput).toContain('healthy');
    expect(apiOutput).toContain('ollama');
    
    // Test Embedding API
    await page.locator('button:has-text("Test Embedding API")').click();
    await page.waitForTimeout(5000);
    
    // Check embedding results
    const embeddingOutput = await page.locator('#apiOutput').textContent();
    expect(embeddingOutput).toMatch(/embedding|dimension/i);
  });
  
  test('should verify export functionality works', async ({ page }) => {
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for page load and status checks
    await page.waitForTimeout(3000);
    
    // Check that export button is visible
    await expect(page.locator('button:has-text("Export Results")')).toBeVisible();
    
    // Set up download promise before clicking
    const downloadPromise = page.waitForEvent('download');
    
    // Click export button
    await page.locator('button:has-text("Export Results")').click();
    
    // Wait for download
    const download = await downloadPromise;
    
    // Verify filename pattern
    expect(download.suggestedFilename()).toMatch(/ai-integration-test-results-.*\.json/);
    
    // Optionally save and verify content
    const downloadPath = `C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-export-${Date.now()}.json`;
    await download.saveAs(downloadPath);
    
    console.log('Export test completed, file saved to:', downloadPath);
  });
  
  test('should validate navigation between all interfaces', async ({ page }) => {
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for page load
    await page.waitForTimeout(2000);
    
    // Check all navigation buttons are present and properly configured
    const svelteKitButton = page.locator('a[href*="localhost:5175"]');
    await expect(svelteKitButton).toBeVisible();
    await expect(svelteKitButton).toContainText('SvelteKit Demo');
    
    const goServiceButton = page.locator('a[href*="localhost:8081/test"]');
    await expect(goServiceButton).toBeVisible();
    await expect(goServiceButton).toContainText('Go Service Test');
    
    const healthCheckButton = page.locator('a[href*="localhost:8081/api/health"]');
    await expect(healthCheckButton).toBeVisible();
    await expect(healthCheckButton).toContainText('Health Check');
    
    const ollamaButton = page.locator('a[href*="localhost:11434"]');
    await expect(ollamaButton).toBeVisible();
    await expect(ollamaButton).toContainText('Ollama Server');
    
    console.log('All navigation buttons validated successfully');
  });
  
  test('should handle document upload simulation', async ({ page }) => {
    await page.goto('file:///C:/Users/james/Desktop/deeds-web/deeds-web-app/ai-summary-service/test-integration.html');
    
    // Wait for page load
    await page.waitForTimeout(2000);
    
    // Check upload area is present
    await expect(page.locator('.upload-area, #uploadArea')).toBeVisible();
    
    // Check file input is present
    await expect(page.locator('input[type="file"]')).toBeVisible();
    
    // Check Select Files button
    await expect(page.locator('button:has-text("Select Files")')).toBeVisible();
    
    console.log('Document upload interface validated');
  });
});

test.describe('Service Status Verification', () => {
  
  test('should verify actual Ollama service is running', async ({ request }) => {
    // Direct API test to confirm Ollama is actually running
    try {
      const healthResponse = await request.get('http://localhost:8081/api/health');
      expect(healthResponse.ok()).toBeTruthy();
      
      const healthData = await healthResponse.json();
      console.log('Go Service Health Response:', healthData);
      
      expect(healthData.services.ollama).toBe('healthy');
      expect(healthData.status).toBe('healthy');
      
    } catch (error) {
      console.error('Failed to verify service status:', error);
      throw error;
    }
  });
  
  test('should test Ollama embedding endpoint directly', async ({ request }) => {
    const response = await request.post('http://localhost:8081/api/embed', {
      data: {
        text: 'Playwright test for Ollama integration'
      }
    });
    
    expect(response.ok()).toBeTruthy();
    
    const data = await response.json();
    expect(data).toHaveProperty('embedding');
    expect(data).toHaveProperty('dimension');
    expect(data.dimension).toBe(768);
    
    console.log('Ollama embedding test successful, dimension:', data.dimension);
  });
});