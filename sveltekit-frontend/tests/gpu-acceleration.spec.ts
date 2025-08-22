
import { test, expect } from '@playwright/test';

test.describe('GPU Acceleration and NVIDIA CUDA', () => {
  test('should detect CUDA availability', async ({ page }) => {
    const response = await page.request.get('/api/gpu/cuda-status');
    expect(response.status()).toBe(200);
    
    const cudaStatus = await response.json();
    
    expect(cudaStatus).toHaveProperty('cuda_available');
    expect(cudaStatus).toHaveProperty('cuda_version');
    expect(cudaStatus).toHaveProperty('driver_version');
    
    // Log CUDA status for debugging
    console.log('CUDA Status:', cudaStatus);
    
    if (cudaStatus.cuda_available) {
      expect(cudaStatus.cuda_version).toMatch(/\d+\.\d+/);
      expect(cudaStatus.driver_version).toBeDefined();
    }
  });

  test('should list available GPUs', async ({ page }) => {
    const response = await page.request.get('/api/gpu/devices');
    expect(response.status()).toBe(200);
    
    const gpuList = await response.json();
    
    expect(gpuList).toHaveProperty('devices');
    expect(Array.isArray(gpuList.devices)).toBe(true);
    
    if (gpuList.devices.length > 0) {
      gpuList.devices.forEach((gpu: unknown) => {
        expect(gpu).toHaveProperty('id');
        expect(gpu).toHaveProperty('name');
        expect(gpu).toHaveProperty('memory_total');
        expect(gpu).toHaveProperty('memory_used');
        expect(gpu).toHaveProperty('memory_free');
        expect(gpu).toHaveProperty('compute_capability');
        expect(gpu).toHaveProperty('temperature');
        expect(gpu).toHaveProperty('utilization');
      });
    }
  });

  test('should verify Ollama GPU usage', async ({ page }) => {
    const response = await page.request.get('/api/ollama/gpu-config');
    expect(response.status()).toBe(200);
    
    const ollamaGpuConfig = await response.json();
    
    expect(ollamaGpuConfig).toHaveProperty('gpu_enabled');
    expect(ollamaGpuConfig).toHaveProperty('gpu_layers');
    expect(ollamaGpuConfig).toHaveProperty('gpu_memory_fraction');
    
    if (ollamaGpuConfig.gpu_enabled) {
      expect(ollamaGpuConfig.gpu_layers).toBeGreaterThan(0);
      expect(ollamaGpuConfig.gpu_memory_fraction).toBeGreaterThan(0);
      expect(ollamaGpuConfig.gpu_memory_fraction).toBeLessThanOrEqual(1);
    }
  });

  test('should benchmark GPU vs CPU inference', async ({ page }) => {
    const testPrompt = 'Explain the concept of legal precedent in 100 words.';
    
    // Test CPU inference
    console.log('Testing CPU inference...');
    const cpuStartTime = Date.now();
    
    const cpuResponse = await page.request.post('/api/ai/chat', {
      data: {
        messages: [{ role: 'user', content: testPrompt }],
        model: 'llama3.2',
        options: {
          gpu_layers: 0 // Force CPU
        }
      }
    });
    
    const cpuEndTime = Date.now();
    const cpuTime = cpuEndTime - cpuStartTime;
    
    expect(cpuResponse.status()).toBe(200);
    const cpuResult = await cpuResponse.json();
    
    // Test GPU inference (if available)
    const gpuStatusResponse = await page.request.get('/api/gpu/cuda-status');
    const { cuda_available } = await gpuStatusResponse.json();
    
    if (cuda_available) {
      console.log('Testing GPU inference...');
      const gpuStartTime = Date.now();
      
      const gpuResponse = await page.request.post('/api/ai/chat', {
        data: {
          messages: [{ role: 'user', content: testPrompt }],
          model: 'llama3.2',
          options: {
            gpu_layers: -1 // Use all available GPU layers
          }
        }
      });
      
      const gpuEndTime = Date.now();
      const gpuTime = gpuEndTime - gpuStartTime;
      
      expect(gpuResponse.status()).toBe(200);
      const gpuResult = await gpuResponse.json();
      
      // Compare performance
      console.log(`CPU Time: ${cpuTime}ms`);
      console.log(`GPU Time: ${gpuTime}ms`);
      console.log(`Speedup: ${(cpuTime / gpuTime).toFixed(2)}x`);
      
      // GPU should be faster for inference
      expect(gpuTime).toBeLessThan(cpuTime);
      
      // Store benchmark results
      await page.request.post('/api/gpu/benchmark-result', {
        data: {
          test_name: 'inference_comparison',
          cpu_time: cpuTime,
          gpu_time: gpuTime,
          speedup: cpuTime / gpuTime,
          model: 'llama3.2',
          prompt_length: testPrompt.length
        }
      });
    }
  });

  test('should monitor GPU memory during embedding generation', async ({ page }) => {
    // Get initial GPU memory state
    const initialMemResponse = await page.request.get('/api/gpu/memory-status');
    const initialMemory = await initialMemResponse.json();
    
    // Generate embeddings for multiple texts
    const texts = Array(10).fill(null).map((_, i) => 
      `This is test document ${i} for GPU memory monitoring during embedding generation.`
    );
    
    const memorySnapshots = [];
    
    for (const text of texts) {
      // Generate embedding
      await page.request.post('/api/ai/embeddings', {
        data: {
          text: text,
          model: 'nomic-embed-text'
        }
      });
      
      // Capture memory snapshot
      const memResponse = await page.request.get('/api/gpu/memory-status');
      const memStatus = await memResponse.json();
      memorySnapshots.push(memStatus);
    }
    
    // Analyze memory usage
    if (initialMemory.gpu_available && memorySnapshots.length > 0) {
      const maxMemoryUsed = Math.max(...memorySnapshots.map((s: any) => s.memory_used || 0));
      const avgMemoryUsed = memorySnapshots.reduce((sum, s) => sum + (s.memory_used || 0), 0) / memorySnapshots.length;
      
      console.log(`Initial Memory: ${initialMemory.memory_used || 0} MB`);
      console.log(`Max Memory Used: ${maxMemoryUsed} MB`);
      console.log(`Average Memory Used: ${avgMemoryUsed.toFixed(2)} MB`);
      
      // Memory usage should be reasonable
      const memoryIncrease = maxMemoryUsed - (initialMemory.memory_used || 0);
      expect(memoryIncrease).toBeLessThan(2000); // Less than 2GB increase
    }
  });

  test('should handle GPU out-of-memory gracefully', async ({ page }) => {
    // Skip if no GPU available
    const gpuStatusResponse = await page.request.get('/api/gpu/cuda-status');
    const { cuda_available } = await gpuStatusResponse.json();
    
    if (!cuda_available) {
      test.skip();
      return;
    }
    
    // Try to load a model that might exceed GPU memory
    const response = await page.request.post('/api/ai/load-model', {
      data: {
        model: 'llama3.2:70b', // Large model
        gpu_layers: -1 // Try to load all layers on GPU
      }
    });
    
    // Should either succeed or fail gracefully
    if (response.status() === 200) {
      const result = await response.json();
      expect(result).toHaveProperty('loaded');
      expect(result).toHaveProperty('gpu_layers_loaded');
    } else if (response.status() === 507) { // Insufficient Storage
      const error = await response.json();
      expect(error.error).toContain('memory');
      expect(error).toHaveProperty('fallback');
      expect(error.fallback).toBe('cpu');
    } else {
      // Other errors should still be handled gracefully
      expect([400, 500]).toContain(response.status());
    }
  });

  test('should optimize batch processing on GPU', async ({ page }) => {
    const batchSizes = [1, 5, 10, 20];
    const results = [];
    
    for (const batchSize of batchSizes) {
      const texts = Array(batchSize).fill(null).map((_, i) => 
        `Batch processing test document ${i} for GPU optimization testing.`
      );
      
      const startTime = Date.now();
      
      // Process batch
      const response = await page.request.post('/api/ai/batch-embeddings', {
        data: {
          texts: texts,
          model: 'nomic-embed-text',
          use_gpu: true
        }
      });
      
      const endTime = Date.now();
      
      expect(response.status()).toBe(200);
      const result = await response.json();
      
      const timePerItem = (endTime - startTime) / batchSize;
      
      results.push({
        batch_size: batchSize,
        total_time: endTime - startTime,
        time_per_item: timePerItem
      });
      
      console.log(`Batch size ${batchSize}: ${timePerItem.toFixed(2)}ms per item`);
    }
    
    // Larger batches should be more efficient (lower time per item)
    expect(results[3].time_per_item).toBeLessThan(results[0].time_per_item);
  });

  test('should verify GPU acceleration for vector operations', async ({ page }) => {
    // Generate test vectors
    const dimension = 768;
    const numVectors = 1000;
    
    const vectors = Array(numVectors).fill(null).map(() => 
      Array(dimension).fill(0).map(() => Math.random())
    );
    
    // Test GPU-accelerated similarity computation
    const startTime = Date.now();
    
    const response = await page.request.post('/api/vectors/gpu-similarity-matrix', {
      data: {
        vectors: vectors,
        metric: 'cosine',
        use_gpu: true
      }
    });
    
    const endTime = Date.now();
    const gpuTime = endTime - startTime;
    
    expect(response.status()).toBe(200);
    const result = await response.json();
    
    expect(result).toHaveProperty('computation_time');
    expect(result).toHaveProperty('gpu_used');
    expect(result.gpu_used).toBe(true);
    
    console.log(`GPU similarity matrix computation: ${gpuTime}ms for ${numVectors} vectors`);
    
    // Should complete quickly with GPU
    expect(gpuTime).toBeLessThan(5000); // Under 5 seconds for 1000 vectors
  });

  test('should monitor GPU temperature and throttling', async ({ page }) => {
    // Run a stress test while monitoring temperature
    const monitoringDuration = 30000; // 30 seconds
    const checkInterval = 5000; // Check every 5 seconds
    
    const temperatureReadings = [];
    const startTime = Date.now();
    
    // Start stress test
    const stressPromise = page.request.post('/api/gpu/stress-test', {
      data: {
        duration: monitoringDuration,
        intensity: 'medium'
      }
    });
    
    // Monitor temperature
    const monitoringInterval = setInterval(async () => {
      const response = await page.request.get('/api/gpu/temperature');
      if (response.ok()) {
        const data = await response.json();
        temperatureReadings.push({
          timestamp: Date.now() - startTime,
          temperature: data.temperature,
          throttling: data.throttling
        });
      }
    }, checkInterval);
    
    // Wait for stress test to complete
    const stressResponse = await stressPromise;
    clearInterval(monitoringInterval);
    
    expect(stressResponse.status()).toBe(200);
    
    // Analyze temperature data
    if (temperatureReadings.length > 0) {
      const maxTemp = Math.max(...temperatureReadings.map((r: any) => r.temperature || 0));
      const avgTemp = temperatureReadings.reduce((sum, r) => sum + (r.temperature || 0), 0) / temperatureReadings.length;
      const throttlingOccurred = temperatureReadings.some((r: any) => r.throttling);
      
      console.log(`Max Temperature: ${maxTemp}°C`);
      console.log(`Average Temperature: ${avgTemp.toFixed(1)}°C`);
      console.log(`Throttling Occurred: ${throttlingOccurred}`);
      
      // Temperature should stay within safe limits
      expect(maxTemp).toBeLessThan(85); // Most GPUs throttle around 83-85°C
    }
  });

  test('should validate GPU setup and configuration', async ({ page }) => {
    const response = await page.request.get('/api/gpu/validate-setup');
    expect(response.status()).toBe(200);
    
    const validation = await response.json();
    
    expect(validation).toHaveProperty('cuda_available');
    expect(validation).toHaveProperty('ollama_gpu_enabled');
    expect(validation).toHaveProperty('pytorch_cuda_available');
    expect(validation).toHaveProperty('gpu_compute_capability');
    expect(validation).toHaveProperty('recommended_settings');
    
    // Log validation results
    console.log('GPU Setup Validation:', JSON.stringify(validation, null, 2));
    
    // If GPU is available, check compatibility
    if (validation.cuda_available) {
      expect(validation.gpu_compute_capability).toBeGreaterThanOrEqual(3.5); // Minimum for most models
      
      // Verify recommended settings are applied
      if (validation.recommended_settings) {
        expect(validation.recommended_settings).toHaveProperty('gpu_layers');
        expect(validation.recommended_settings).toHaveProperty('memory_fraction');
      }
    }
  });
});