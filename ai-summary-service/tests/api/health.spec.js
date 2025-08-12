// tests/api/health.spec.js
const { test, expect } = require('@playwright/test');

test.describe('Health API Tests', () => {
  test('should return healthy status', async ({ request }) => {
    const response = await request.get('/api/health');
    expect(response.ok()).toBeTruthy();
    
    const health = await response.json();
    expect(health.status).toBe('healthy');
    expect(health.version).toBe('2.0.0-enhanced');
    expect(health.services).toBeDefined();
    expect(health.services.ollama).toBeDefined();
    expect(health.services.qdrant).toBeDefined();
    expect(health.services.gpu).toBeDefined();
  });

  test('should have proper timestamp format', async ({ request }) => {
    const response = await request.get('/api/health');
    const health = await response.json();
    
    expect(health.timestamp).toBeDefined();
    expect(new Date(health.timestamp)).toBeInstanceOf(Date);
  });

  test('should return service status details', async ({ request }) => {
    const response = await request.get('/api/health');
    const health = await response.json();
    
    // Check Ollama service
    expect(['healthy', 'unhealthy']).toContain(health.services.ollama);
    
    // Check GPU status
    expect(['enabled', 'disabled']).toContain(health.services.gpu);
    
    // Check Qdrant status
    expect(['healthy', 'unhealthy']).toContain(health.services.qdrant);
  });
});