import { test, expect } from '@playwright/test';

test.describe('Legal AI API Endpoints Tests', () => {
  const API_BASE_URL = 'http://localhost:8093';
  
  test('Go service health endpoint should respond', async ({ page }) => {
    const response = await page.goto(`${API_BASE_URL}/health`);
    expect(response?.status()).toBe(200);
    
    const healthData = await response?.json();
    expect(healthData).toHaveProperty('status');
    expect(healthData).toHaveProperty('ecosystem', 'kratos');
    expect(healthData).toHaveProperty('framework', 'gin');
    
    console.log('Health check response:', healthData);
  });

  test('should verify database connectivity', async ({ page }) => {
    const response = await page.goto(`${API_BASE_URL}/health`);
    const healthData = await response?.json();
    
    expect(healthData?.db).toBe(true);
    console.log('Database connected:', healthData?.db);
  });

  test('should verify MinIO connectivity', async ({ page }) => {
    const response = await page.goto(`${API_BASE_URL}/health`);
    const healthData = await response?.json();
    
    expect(healthData?.minio).toBe(true);
    console.log('MinIO connected:', healthData?.minio);
  });

  test('should handle CORS properly', async ({ page }) => {
    // Test CORS by making a request from the frontend origin
    page.setExtraHTTPHeaders({
      'Origin': 'http://localhost:5173'
    });
    
    const response = await page.goto(`${API_BASE_URL}/health`);
    expect(response?.status()).toBe(200);
    
    const headers = response?.headers();
    console.log('CORS headers present:', {
      'access-control-allow-origin': headers?.['access-control-allow-origin'],
      'access-control-allow-methods': headers?.['access-control-allow-methods'],
    });
  });

  test('should handle 404 errors gracefully', async ({ page }) => {
    const response = await page.goto(`${API_BASE_URL}/non-existent-endpoint`, {
      failOnStatusCode: false
    });
    
    expect(response?.status()).toBe(404);
    console.log('404 handling works correctly');
  });

  test('should respond within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    const response = await page.goto(`${API_BASE_URL}/health`);
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    
    expect(response?.status()).toBe(200);
    expect(responseTime).toBeLessThan(2000); // Less than 2 seconds
    
    console.log(`API response time: ${responseTime}ms`);
  });

  test('should return valid JSON responses', async ({ page }) => {
    const response = await page.goto(`${API_BASE_URL}/health`);
    
    expect(response?.headers()['content-type']).toContain('application/json');
    
    const data = await response?.json();
    expect(typeof data).toBe('object');
    expect(data).not.toBeNull();
    
    console.log('Valid JSON response received');
  });
});