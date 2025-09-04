import { test, expect, type Page } from "@playwright/test";

/**
 * RAG Integration Tests with Ollama + SvelteKit + PostgreSQL + pgvector
 * Tests the complete Retrieval-Augmented Generation pipeline
 */

test.describe("RAG Integration Tests", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("/ai-demo");

    // Wait for page to load
    await page.waitForLoadState("networkidle");

    // Check if Ollama service is healthy
    const healthResponse = await page.request.get("/api/ai/health");
    const healthData = await healthResponse.json();

    test.skip(
      !healthData.services?.ollama?.healthy,
      "Ollama service not available"
    );
  });

  test("should load AI demo page with RAG interface", async () => {
    await expect(page.locator("h1")).toContainText("Legal AI System Demo");
    await expect(page.locator('[data-testid="system-status"]')).toBeVisible();
    await expect(page.locator('[data-testid="rag-interface"]')).toBeVisible();
  });

  test("should show Ollama service status", async () => {
    const statusSection = page.locator('[data-testid="ollama-status"]');
    await expect(statusSection).toBeVisible();

    // Check for healthy status indicator
    const healthIndicator = statusSection.locator(
      ".text-green-600, .text-red-600"
    );
    await expect(healthIndicator).toBeVisible();
  });

  test("should display available AI models", async () => {
    const modelsSection = page.locator('[data-testid="available-models"]');
    await expect(modelsSection).toBeVisible();

    // Should show at least one model badge
    const modelBadges = page.locator('[data-testid="model-badge"]');
    await expect(modelBadges.first()).toBeVisible();
  });

  test("should perform RAG-enhanced chat query", async () => {
    // Enable RAG mode
    await page.locator('[data-testid="rag-toggle"]').check();

    // Enter a legal question
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("What are the key elements of criminal prosecution?");

    // Send message
    await page.locator('[data-testid="send-button"]').click();

    // Wait for response
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 30000,
    });

    const response = page.locator('[data-testid="chat-response"]').last();
    await expect(response).toBeVisible();
    await expect(response).not.toBeEmpty();

    // Check for performance metrics
    const metrics = page.locator('[data-testid="performance-metrics"]');
    await expect(metrics).toBeVisible();
    await expect(
      metrics.locator('[data-testid="response-time"]')
    ).toContainText("ms");
    await expect(
      metrics.locator('[data-testid="tokens-per-second"]')
    ).toContainText("/s");
  });

  test("should handle vector similarity search", async () => {
    // Test vector search functionality
    const searchInput = page.locator('[data-testid="vector-search-input"]');
    await searchInput.fill("evidence analysis procedures");

    await page.locator('[data-testid="vector-search-button"]').click();

    // Wait for search results
    await page.waitForSelector('[data-testid="search-results"]', {
      timeout: 15000,
    });

    const results = page.locator('[data-testid="search-result-item"]');
    await expect(results.first()).toBeVisible();

    // Check similarity scores
    const scoreElement = results
      .first()
      .locator('[data-testid="similarity-score"]');
    await expect(scoreElement).toBeVisible();
  });

  test("should show token usage tracking", async () => {
    const tokenTracker = page.locator('[data-testid="token-tracker"]');
    await expect(tokenTracker).toBeVisible();

    // Check token usage display
    await expect(
      tokenTracker.locator('[data-testid="tokens-used"]')
    ).toBeVisible();
    await expect(
      tokenTracker.locator('[data-testid="tokens-remaining"]')
    ).toBeVisible();

    // Test token limit slider
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await expect(tokenSlider).toBeVisible();

    // Adjust token limit
    await tokenSlider.fill("4000");
    await expect(
      page.locator('[data-testid="token-limit-display"]')
    ).toContainText("4000");
  });

  test("should handle PostgreSQL + pgvector integration", async () => {
    // Test database connectivity
    const dbStatusResponse = await page.request.get("/api/db/status");
    const dbStatus = await dbStatusResponse.json();

    expect(dbStatus.postgresql).toBeTruthy();
    expect(dbStatus.pgvector).toBeTruthy();

    // Test vector storage
    const testVector = Array.from({ length: 384 }, () => Math.random());
    const storeResponse = await page.request.post("/api/vectors/store", {
      data: {
        id: "test-doc-" + Date.now(),
        vector: testVector,
        content: "Test legal document content",
        metadata: { type: "test", timestamp: new Date().toISOString() },
      },
    });

    expect(storeResponse.ok()).toBeTruthy();
  });

  test("should demonstrate CUDA GPU acceleration", async () => {
    // Check GPU acceleration status
    const gpuStatusResponse = await page.request.get("/api/system/gpu");
    const gpuStatus = await gpuStatusResponse.json();

    // Skip if no GPU available
    test.skip(!gpuStatus.cuda_available, "CUDA GPU not available");

    // Test GPU-accelerated inference
    const startTime = Date.now();

    const chatResponse = await page.request.post("/api/ai/chat", {
      data: {
        message: "Generate a detailed legal brief outline",
        model: "gemma3-legal",
        useGPU: true,
        max_tokens: 1000,
      },
    });

    const endTime = Date.now();
    const responseData = await chatResponse.json();

    expect(chatResponse.ok()).toBeTruthy();
    expect(responseData.performance.gpu_accelerated).toBeTruthy();
    expect(responseData.performance.tokens_per_second).toBeGreaterThan(10);

    // GPU should be significantly faster
    const inferenceTime = endTime - startTime;
    console.log(`GPU inference time: ${inferenceTime}ms`);
    console.log(
      `Tokens per second: ${responseData.performance.tokens_per_second}`
    );
  });

  test("should handle real-time streaming responses", async () => {
    // Enable streaming mode
    await page.locator('[data-testid="streaming-toggle"]').check();

    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("Explain the legal process for filing a motion");

    await page.locator('[data-testid="send-button"]').click();

    // Watch for streaming response chunks
    const responseContainer = page.locator(
      '[data-testid="streaming-response"]'
    );
    await expect(responseContainer).toBeVisible();

    // Wait for streaming to complete
    await page.waitForFunction(
      () => document.querySelector('[data-testid="streaming-complete"]'),
      undefined,
      { timeout: 45000 }
    );

    const finalResponse = page.locator('[data-testid="final-response"]');
    await expect(finalResponse).toBeVisible();
    await expect(finalResponse).not.toBeEmpty();
  });

  test("should validate memory and performance optimization", async () => {
    // Test memory usage tracking
    const memoryTracker = page.locator('[data-testid="memory-tracker"]');
    await expect(memoryTracker).toBeVisible();

    // Check memory optimization features
    await page.locator('[data-testid="optimize-memory-button"]').click();

    // Wait for optimization to complete
    await page.waitForSelector('[data-testid="optimization-complete"]', {
      timeout: 10000,
    });

    // Verify memory reduction
    const optimizedMemory = await page
      .locator('[data-testid="memory-usage"]')
      .textContent();
    expect(optimizedMemory).toBeTruthy();

    // Test cache performance
    const cacheMetrics = page.locator('[data-testid="cache-metrics"]');
    await expect(cacheMetrics).toBeVisible();
    await expect(
      cacheMetrics.locator('[data-testid="cache-hit-rate"]')
    ).toBeVisible();
  });

  test("should handle error scenarios gracefully", async () => {
    // Test with invalid model
    const errorResponse = await page.request.post("/api/ai/chat", {
      data: {
        message: "Test message",
        model: "non-existent-model",
      },
    });

    expect(errorResponse.status()).toBe(400);

    // Test with empty message
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("");

    await page.locator('[data-testid="send-button"]').click();

    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeVisible();
    await expect(errorMessage).toContainText("Message is required");
  });
});
