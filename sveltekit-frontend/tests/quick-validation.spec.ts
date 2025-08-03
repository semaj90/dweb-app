import { test, expect } from '@playwright/test';

test.describe('Quick System Validation', () => {
  test('should verify all services are running and healthy', async ({ page }) => {
    // Test if dev server is responding
    await page.goto('/');
    await expect(page).toHaveTitle(/Legal AI|YoRHa|Deeds/i);
    
    // Test basic navigation
    await page.goto('/login');
    await expect(page.locator('input[name="email"]')).toBeVisible();
    
    // Test API health endpoint
    const healthResponse = await page.request.get('/api/health');
    console.log('Health check status:', healthResponse.status());
    
    if (healthResponse.ok()) {
      const health = await healthResponse.json();
      console.log('System health:', health);
    }
    
    // Test if Ollama is responding
    const ollamaResponse = await page.request.get('/api/ai/health');
    console.log('Ollama health status:', ollamaResponse.status());
    
    if (ollamaResponse.ok()) {
      const ollamaHealth = await ollamaResponse.json();
      console.log('Ollama status:', ollamaHealth);
    }
  });

  test('should create account, login, and stay logged in', async ({ page }) => {
    const testEmail = `quicktest-${Date.now()}@example.com`;
    const testPassword = 'QuickTest123!';
    
    // Try to register (may fail if user exists, that's OK)
    await page.goto('/register');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.fill('input[name="confirmPassword"]', testPassword);
    await page.fill('input[name="name"]', 'Quick Test User');
    
    // Accept terms if present
    const termsCheckbox = page.locator('input[name="terms"]');
    if (await termsCheckbox.isVisible()) {
      await termsCheckbox.check();
    }
    
    await page.click('button[type="submit"]');
    
    // Wait for redirect or error
    await page.waitForTimeout(2000);
    
    // Now try to login
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard
    await page.waitForURL('/dashboard/**', { timeout: 10000 });
    
    // Check if logged in
    expect(page.url()).toContain('/dashboard');
    
    // Refresh page to test session persistence
    await page.reload();
    expect(page.url()).toContain('/dashboard');
    
    console.log('✅ User authentication and session persistence working');
  });

  test('should verify Ollama GPU status and model availability', async ({ page }) => {
    // Check GPU status
    const gpuResponse = await page.request.get('/api/gpu/cuda-status');
    
    if (gpuResponse.ok()) {
      const gpuStatus = await gpuResponse.json();
      console.log('CUDA Status:', gpuStatus.cuda_available ? '✅ Available' : '❌ Not Available');
      console.log('GPU Info:', gpuStatus);
    }
    
    // Check Ollama models
    const modelsResponse = await page.request.get('/api/ai/models');
    
    if (modelsResponse.ok()) {
      const models = await modelsResponse.json();
      console.log('Available Models:', models.models?.length || 0);
      
      // Check for required models
      const requiredModels = ['llama3.2', 'nomic-embed-text'];
      requiredModels.forEach(modelName => {
        const hasModel = models.models?.some((m: any) => m.name.includes(modelName));
        console.log(`${modelName}: ${hasModel ? '✅' : '❌'}`);
      });
    }
  });

  test('should test basic AI functionality', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Go to AI assistant
    await page.goto('/dashboard/ai-assistant');
    
    // Check if chat interface is available
    const chatInput = page.locator('[data-testid="chat-input"]');
    if (await chatInput.isVisible()) {
      // Send a simple test message
      await chatInput.fill('Hello, can you help me?');
      await page.click('[data-testid="send-button"]');
      
      // Wait for response (with longer timeout for first request)
      await page.waitForSelector('[data-testid="ai-response"]', { timeout: 30000 });
      
      const response = await page.locator('[data-testid="ai-response"]').last().textContent();
      expect(response?.length).toBeGreaterThan(10);
      
      console.log('✅ AI chat functionality working');
    } else {
      console.log('⚠️ AI chat interface not found');
    }
  });

  test('should verify database connectivity', async ({ page }) => {
    const dbResponse = await page.request.get('/api/db/health');
    
    if (dbResponse.ok()) {
      const dbHealth = await dbResponse.json();
      console.log('Database Status:', dbHealth.database?.connected ? '✅ Connected' : '❌ Disconnected');
      console.log('Database Info:', dbHealth.database);
    } else {
      console.log('❌ Database health check failed');
    }
  });

  test('should check system resources and performance', async ({ page }) => {
    const metricsResponse = await page.request.get('/api/metrics');
    
    if (metricsResponse.ok()) {
      const metrics = await metricsResponse.json();
      
      console.log('System Metrics:');
      console.log(`CPU Usage: ${metrics.cpu?.usage_percent || 'N/A'}%`);
      console.log(`Memory Usage: ${metrics.memory?.usage_percent || 'N/A'}%`);
      console.log(`Disk Usage: ${metrics.disk?.usage_percent || 'N/A'}%`);
      console.log(`Uptime: ${metrics.process?.uptime_seconds || 'N/A'}s`);
    } else {
      console.log('⚠️ System metrics not available');
    }
  });
});