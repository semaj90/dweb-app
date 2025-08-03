import { test, expect } from '@playwright/test';

test.describe('Ollama Integration', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the AI demo page
    await page.goto('/ai-demo');
  });

  test('should check Ollama service health', async ({ page }) => {
    // Check if the Ollama service is running
    const response = await page.request.get('/api/ai/health');
    expect(response.status()).toBe(200);
    
    const health = await response.json();
    expect(health).toHaveProperty('status');
    expect(health.status).toBe('healthy');
    expect(health).toHaveProperty('ollama');
    expect(health.ollama.available).toBe(true);
  });

  test('should list available Ollama models', async ({ page }) => {
    const response = await page.request.get('/api/ai/models');
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty('models');
    expect(Array.isArray(data.models)).toBe(true);
    
    // Check if specific models are available
    const modelNames = data.models.map((m: any) => m.name);
    expect(modelNames).toContain('llama3.2');
  });

  test('should interact with Ollama chat interface', async ({ page }) => {
    // Wait for the chat interface to load
    await page.waitForSelector('[data-testid="ollama-chat-interface"]', { timeout: 10000 });
    
    // Type a message
    const inputField = page.locator('[data-testid="chat-input"]');
    await inputField.fill('Hello, can you help me with legal research?');
    
    // Send the message
    await page.locator('[data-testid="send-button"]').click();
    
    // Wait for response
    await page.waitForSelector('[data-testid="ai-response"]', { timeout: 30000 });
    
    // Verify response exists
    const response = await page.locator('[data-testid="ai-response"]').last().textContent();
    expect(response).toBeTruthy();
    expect(response?.length).toBeGreaterThan(10);
  });

  test('should handle streaming responses', async ({ page }) => {
    await page.waitForSelector('[data-testid="ollama-chat-interface"]');
    
    // Enable streaming mode if there's a toggle
    const streamToggle = page.locator('[data-testid="stream-toggle"]');
    if (await streamToggle.isVisible()) {
      await streamToggle.click();
    }
    
    // Send a message
    const inputField = page.locator('[data-testid="chat-input"]');
    await inputField.fill('Explain the concept of precedent in law');
    await page.locator('[data-testid="send-button"]').click();
    
    // Check for streaming indicator
    await page.waitForSelector('[data-testid="streaming-indicator"]', { timeout: 5000 });
    
    // Wait for streaming to complete
    await page.waitForSelector('[data-testid="streaming-indicator"]', { 
      state: 'hidden',
      timeout: 30000 
    });
    
    // Verify complete response
    const response = await page.locator('[data-testid="ai-response"]').last().textContent();
    expect(response).toContain('precedent');
  });

  test('should handle GPU acceleration status', async ({ page }) => {
    const response = await page.request.get('/api/ai/gpu-status');
    expect(response.status()).toBe(200);
    
    const gpuStatus = await response.json();
    expect(gpuStatus).toHaveProperty('cuda_available');
    expect(gpuStatus).toHaveProperty('gpu_info');
    
    // If CUDA is available, check GPU details
    if (gpuStatus.cuda_available) {
      expect(gpuStatus.gpu_info).toHaveProperty('name');
      expect(gpuStatus.gpu_info).toHaveProperty('memory');
    }
  });

  test('should generate embeddings with Ollama', async ({ page }) => {
    const testText = 'This is a legal document about contract law';
    
    const response = await page.request.post('/api/ai/embeddings', {
      data: {
        text: testText,
        model: 'nomic-embed-text'
      }
    });
    
    expect(response.status()).toBe(200);
    
    const result = await response.json();
    expect(result).toHaveProperty('embedding');
    expect(Array.isArray(result.embedding)).toBe(true);
    expect(result.embedding.length).toBeGreaterThan(0);
    
    // Check embedding dimensions (nomic-embed-text typically returns 768 dimensions)
    expect(result.embedding.length).toBe(768);
  });

  test('should handle multiple concurrent requests', async ({ page }) => {
    const requests = [
      'What is tort law?',
      'Explain criminal procedure',
      'Define intellectual property'
    ];
    
    // Send all requests concurrently
    const promises = requests.map(async (question) => {
      return page.request.post('/api/ai/chat', {
        data: {
          messages: [{ role: 'user', content: question }],
          model: 'llama3.2'
        }
      });
    });
    
    const responses = await Promise.all(promises);
    
    // Verify all responses are successful
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });
    
    // Check response content
    const results = await Promise.all(responses.map(r => r.json()));
    results.forEach((result, index) => {
      expect(result).toHaveProperty('response');
      expect(result.response).toBeTruthy();
      expect(result.response.length).toBeGreaterThan(20);
    });
  });

  test('should respect token limits', async ({ page }) => {
    // Create a very long prompt
    const longPrompt = 'Explain the following legal concepts in detail: ' + 
      Array(100).fill('contract law, tort law, criminal law').join(', ');
    
    const response = await page.request.post('/api/ai/chat', {
      data: {
        messages: [{ role: 'user', content: longPrompt }],
        model: 'llama3.2',
        options: {
          max_tokens: 100
        }
      }
    });
    
    expect(response.status()).toBe(200);
    
    const result = await response.json();
    expect(result).toHaveProperty('response');
    expect(result).toHaveProperty('token_count');
    expect(result.token_count).toBeLessThanOrEqual(100);
  });

  test('should handle model switching', async ({ page }) => {
    await page.waitForSelector('[data-testid="ollama-chat-interface"]');
    
    // Check if model selector exists
    const modelSelector = page.locator('[data-testid="model-selector"]');
    if (await modelSelector.isVisible()) {
      // Get available models
      const models = await modelSelector.locator('option').allTextContents();
      expect(models.length).toBeGreaterThan(0);
      
      // Switch to a different model if available
      if (models.length > 1) {
        await modelSelector.selectOption({ index: 1 });
        
        // Send a test message with the new model
        const inputField = page.locator('[data-testid="chat-input"]');
        await inputField.fill('Test with different model');
        await page.locator('[data-testid="send-button"]').click();
        
        // Wait for response
        await page.waitForSelector('[data-testid="ai-response"]', { timeout: 30000 });
      }
    }
  });

  test('should handle error scenarios gracefully', async ({ page }) => {
    // Test with invalid model
    const response = await page.request.post('/api/ai/chat', {
      data: {
        messages: [{ role: 'user', content: 'Test' }],
        model: 'non-existent-model'
      }
    });
    
    expect(response.status()).toBe(400);
    
    const error = await response.json();
    expect(error).toHaveProperty('error');
    expect(error.error).toContain('model');
  });
});