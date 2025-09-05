import { test, expect, type Page } from "@playwright/test";

/**
 * Ollama GPU Acceleration Tests
 * Tests NVIDIA CUDA integration and GPU performance
 */

test.describe("Ollama GPU Acceleration Tests", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("/ai-demo");

    // Check GPU availability
    const gpuResponse = await page.request.get("/api/system/gpu");
    const gpuData = await gpuResponse.json();

    test.skip(!gpuData.cuda_available, "NVIDIA CUDA not available");
  });

  test("should detect NVIDIA CUDA availability", async () => {
    const gpuResponse = await page.request.get("/api/system/gpu");
    const gpuData = await gpuResponse.json();

    expect(gpuData.cuda_available).toBeTruthy();
    expect(gpuData.gpu_count).toBeGreaterThan(0);
    expect(gpuData.driver_version).toBeTruthy();
  });

  test("should start Ollama with GPU acceleration", async () => {
    // Test GPU-enabled Ollama startup
    const startResponse = await page.request.post("/api/ollama/start", {
      data: { gpu: true },
    });

    expect(startResponse.ok()).toBeTruthy();

    // Wait for service to be ready
    await page.waitForTimeout(10000);

    // Verify GPU is being used
    const statusResponse = await page.request.get("/api/ollama/status");
    const statusData = await statusResponse.json();

    expect(statusData.gpu_enabled).toBeTruthy();
    expect(statusData.gpu_memory_used).toBeGreaterThan(0);
  });

  test("should compare CPU vs GPU inference performance", async () => {
    const testPrompt =
      "Analyze the legal implications of contract breach in commercial law";

    // CPU inference
    const cpuStartTime = Date.now();
    const cpuResponse = await page.request.post("/api/ai/chat", {
      data: {
        message: testPrompt,
        model: "gemma3-legal",
        useGPU: false,
        max_tokens: 500,
      },
    });
    const cpuEndTime = Date.now();
    const cpuData = await cpuResponse.json();

    // GPU inference
    const gpuStartTime = Date.now();
    const gpuResponse = await page.request.post("/api/ai/chat", {
      data: {
        message: testPrompt,
        model: "gemma3-legal",
        useGPU: true,
        max_tokens: 500,
      },
    });
    const gpuEndTime = Date.now();
    const gpuData = await gpuResponse.json();

    const cpuTime = cpuEndTime - cpuStartTime;
    const gpuTime = gpuEndTime - gpuStartTime;

    console.log(`CPU inference time: ${cpuTime}ms`);
    console.log(`GPU inference time: ${gpuTime}ms`);
    console.log(`CPU tokens/sec: ${cpuData.performance.tokens_per_second}`);
    console.log(`GPU tokens/sec: ${gpuData.performance.tokens_per_second}`);

    // GPU should be faster
    expect(gpuData.performance.tokens_per_second).toBeGreaterThan(
      cpuData.performance.tokens_per_second
    );

    // Log performance improvement
    const speedup =
      gpuData.performance.tokens_per_second /
      cpuData.performance.tokens_per_second;
    console.log(`GPU speedup: ${speedup.toFixed(2)}x`);
  });

  test("should monitor GPU memory usage during inference", async () => {
    // Get initial GPU memory
    const initialMemoryResponse = await page.request.get(
      "/api/system/gpu/memory"
    );
    const initialMemory = await initialMemoryResponse.json();

    // Perform multiple concurrent inferences
    const promises = Array.from({ length: 3 }, (_, i) =>
      page.request.post("/api/ai/chat", {
        data: {
          message: `Legal question ${i + 1}: What are the elements of negligence?`,
          model: "gemma3-legal",
          useGPU: true,
          max_tokens: 300,
        },
      })
    );

    await Promise.all(promises);

    // Check final GPU memory
    const finalMemoryResponse = await page.request.get(
      "/api/system/gpu/memory"
    );
    const finalMemory = await finalMemoryResponse.json();

    expect(finalMemory.used_mb).toBeGreaterThanOrEqual(initialMemory.used_mb);
    expect(finalMemory.free_mb).toBeLessThanOrEqual(initialMemory.free_mb);

    console.log(
      `Initial GPU memory: ${initialMemory.used_mb}MB used, ${initialMemory.free_mb}MB free`
    );
    console.log(
      `Final GPU memory: ${finalMemory.used_mb}MB used, ${finalMemory.free_mb}MB free`
    );
  });

  test("should handle GPU temperature and throttling", async () => {
    const tempResponse = await page.request.get("/api/system/gpu/temperature");
    const tempData = await tempResponse.json();

    expect(tempData.temperature_c).toBeGreaterThan(0);
    expect(tempData.temperature_c).toBeLessThan(100); // Should not overheat

    // Check for thermal throttling
    if (tempData.temperature_c > 80) {
      console.warn(`GPU temperature high: ${tempData.temperature_c}°C`);
    }

    expect(tempData.throttled).toBeFalsy();
  });

  test("should validate CUDA version compatibility", async () => {
    const cudaResponse = await page.request.get("/api/system/cuda/version");
    const cudaData = await cudaResponse.json();

    expect(cudaData.cuda_version).toBeTruthy();
    expect(cudaData.driver_version).toBeTruthy();
    expect(cudaData.runtime_version).toBeTruthy();

    // Log versions for debugging
    console.log(`CUDA Driver Version: ${cudaData.driver_version}`);
    console.log(`CUDA Runtime Version: ${cudaData.runtime_version}`);

    // Check minimum version requirements
    const majorVersion = parseInt(cudaData.cuda_version.split(".")[0]);
    expect(majorVersion).toBeGreaterThanOrEqual(11); // CUDA 11.0+
  });

  test("should test model loading with GPU optimization", async () => {
    const modelName = "gemma3-legal";

    // Load model with GPU optimization
    const loadResponse = await page.request.post("/api/ollama/models/load", {
      data: {
        model: modelName,
        gpu_layers: 35, // Use GPU for most layers
        context_length: 8192,
      },
    });

    expect(loadResponse.ok()).toBeTruthy();

    // Wait for model to load
    await page.waitForTimeout(30000);

    // Verify model is loaded and using GPU
    const modelStatusResponse = await page.request.get(
      `/api/ollama/models/${modelName}/status`
    );
    const modelStatus = await modelStatusResponse.json();

    expect(modelStatus.loaded).toBeTruthy();
    expect(modelStatus.gpu_layers_loaded).toBeGreaterThan(0);
    expect(modelStatus.vram_usage_mb).toBeGreaterThan(0);
  });

  test("should benchmark different model sizes on GPU", async () => {
    const models = ["gemma3:2b", "gemma3:7b", "gemma3:13b"];
    const benchmarkResults: Array<{
      model: string;
      inference_time: number;
      tokens_per_second: number;
      vram_usage: number;
    }> = [];

    for (const model of models) {
      // Check if model is available
      const modelsResponse = await page.request.get("/api/ollama/models");
      const modelsData = await modelsResponse.json();

      if (
        !modelsData.models.some((m: any) =>
          m.name.includes(model.split(":")[0])
        )
      ) {
        console.log(`Skipping ${model} - not available`);
        continue;
      }

      const startTime = Date.now();
      const response = await page.request.post("/api/ai/chat", {
        data: {
          message: "What is the statute of limitations for contract disputes?",
          model: model,
          useGPU: true,
          max_tokens: 200,
        },
      });
      const endTime = Date.now();

      if (response.ok()) {
        const data = await response.json();
        benchmarkResults.push({
          model,
          inference_time: endTime - startTime,
          tokens_per_second: data.performance.tokens_per_second,
          vram_usage: data.performance.vram_usage_mb || 0,
        });
      }
    }

    // Log benchmark results
    console.log("GPU Benchmark Results:");
    benchmarkResults.forEach((result: any) => {
      console.log(
        `${result.model}: ${result.tokens_per_second} tokens/s, ${result.vram_usage}MB VRAM`
      );
    });

    expect(benchmarkResults.length).toBeGreaterThan(0);
  });

  test("should handle GPU memory cleanup", async () => {
    // Perform inference to use GPU memory
    await page.request.post("/api/ai/chat", {
      data: {
        message: "Test GPU memory allocation",
        model: "gemma3-legal",
        useGPU: true,
        max_tokens: 100,
      },
    });

    // Get memory usage before cleanup
    const beforeCleanupResponse = await page.request.get(
      "/api/system/gpu/memory"
    );
    const beforeCleanup = await beforeCleanupResponse.json();

    // Trigger GPU memory cleanup
    const cleanupResponse = await page.request.post("/api/system/gpu/cleanup");
    expect(cleanupResponse.ok()).toBeTruthy();

    // Wait for cleanup to complete
    await page.waitForTimeout(5000);

    // Check memory after cleanup
    const afterCleanupResponse = await page.request.get(
      "/api/system/gpu/memory"
    );
    const afterCleanup = await afterCleanupResponse.json();

    // Memory should be freed (allowing for some variance)
    expect(afterCleanup.free_mb).toBeGreaterThanOrEqual(
      beforeCleanup.free_mb - 100
    );

    console.log(`Memory before cleanup: ${beforeCleanup.used_mb}MB used`);
    console.log(`Memory after cleanup: ${afterCleanup.used_mb}MB used`);
  });
});
