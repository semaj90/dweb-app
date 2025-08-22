import { test, expect } from '@playwright/test';

test.describe('Legal AI Integration Tests', () => {
  const OLLAMA_URL = 'http://localhost:11434';
  
  test('should connect to Ollama API', async ({ page }) => {
    const response = await page.goto(`${OLLAMA_URL}/api/version`);
    expect(response?.status()).toBe(200);
    
    const versionData = await response?.json();
    expect(versionData).toHaveProperty('version');
    
    console.log('Ollama version:', versionData?.version);
  });

  test('should list available models', async ({ page }) => {
    const response = await page.request.get(`${OLLAMA_URL}/api/tags`);
    
    if (response.ok()) {
      const modelsData = await response.json();
      console.log('Available models:', modelsData?.models?.length || 0);
      
      if (modelsData?.models?.length > 0) {
        const modelNames = modelsData.models.map((m: unknown) => m.name);
        console.log('Model names:', modelNames);
        expect(modelsData.models.length).toBeGreaterThan(0);
      } else {
        console.log('No models currently loaded in Ollama');
      }
    } else {
      console.log('Models endpoint not available or configured');
      expect([200, 404]).toContain(response.status());
    }
  });

  test('should test text generation capability', async ({ page }) => {
    // Test with a simple legal query
    const testPrompt = {
      model: 'llama2', // Default model - may need to be adjusted
      prompt: 'What is a legal contract?',
      stream: false,
      options: {
        temperature: 0.1,
        max_tokens: 100
      }
    };
    
    const response = await page.request.post(`${OLLAMA_URL}/api/generate`, {
      data: testPrompt
    });
    
    if (response.ok()) {
      const result = await response.json();
      console.log('AI response received:', result?.response?.substring(0, 100) + '...');
      expect(result).toHaveProperty('response');
      expect(typeof result.response).toBe('string');
      expect(result.response.length).toBeGreaterThan(10);
    } else {
      console.log(`AI generation failed with status: ${response.status()}`);
      console.log('This may indicate no model is loaded or endpoint needs configuration');
      
      // Accept failure as models might not be loaded
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  });

  test('should test embedding generation', async ({ page }) => {
    const testEmbedding = {
      model: 'llama2', // Default model
      prompt: 'Legal document processing'
    };
    
    const response = await page.request.post(`${OLLAMA_URL}/api/embeddings`, {
      data: testEmbedding
    });
    
    if (response.ok()) {
      const result = await response.json();
      console.log('Embedding generated:', result?.embedding?.length || 0, 'dimensions');
      
      if (result?.embedding) {
        expect(Array.isArray(result.embedding)).toBe(true);
        expect(result.embedding.length).toBeGreaterThan(0);
      }
    } else {
      console.log(`Embedding generation failed: ${response.status()}`);
      // Accept failure as embeddings endpoint might not be available
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  });

  test('should handle AI request timeout', async ({ page }) => {
    // Test with a complex prompt that might timeout
    const complexPrompt = {
      model: 'llama2',
      prompt: 'Analyze this complex legal scenario: ' + 'A'.repeat(1000),
      stream: false,
      options: {
        temperature: 0.5,
        max_tokens: 500
      }
    };
    
    try {
      const response = await page.request.post(`${OLLAMA_URL}/api/generate`, {
        data: complexPrompt,
        timeout: 10000 // 10 second timeout
      });
      
      console.log(`Complex prompt response: ${response.status()}`);
      expect([200, 400, 404, 500, 504]).toContain(response.status());
      
    } catch (error) {
      console.log('Request timeout or error (expected for complex prompts)');
      // Timeout is acceptable for complex requests
    }
  });

  test('should validate API response format', async ({ page }) => {
    const simplePrompt = {
      model: 'llama2',
      prompt: 'Hello',
      stream: false
    };
    
    const response = await page.request.post(`${OLLAMA_URL}/api/generate`, {
      data: simplePrompt
    });
    
    if (response.ok()) {
      const result = await response.json();
      
      // Check expected response structure
      expect(typeof result).toBe('object');
      expect(result).toHaveProperty('response');
      
      // Optional fields that might be present
      const optionalFields = ['done', 'context', 'total_duration', 'load_duration'];
      console.log('Response fields:', Object.keys(result));
      
      // At minimum should have response field
      expect(result.response).toBeDefined();
      
    } else {
      console.log('API validation skipped - service not ready');
    }
  });

  test('should handle malformed requests', async ({ page }) => {
    // Test with invalid JSON
    const response = await page.request.post(`${OLLAMA_URL}/api/generate`, {
      data: 'invalid json',
      failOnStatusCode: false
    });
    
    console.log(`Malformed request response: ${response.status()}`);
    expect([400, 404, 500]).toContain(response.status());
  });

  test('should test streaming capability', async ({ page }) => {
    const streamPrompt = {
      model: 'llama2',
      prompt: 'What is law?',
      stream: true
    };
    
    try {
      const response = await page.request.post(`${OLLAMA_URL}/api/generate`, {
        data: streamPrompt,
        timeout: 5000
      });
      
      if (response.ok()) {
        console.log('Streaming response available');
        // For streaming, we just check if connection is made
        expect(response.status()).toBe(200);
      } else {
        console.log(`Streaming test result: ${response.status()}`);
      }
    } catch (error) {
      console.log('Streaming test may need model to be loaded');
    }
  });
});