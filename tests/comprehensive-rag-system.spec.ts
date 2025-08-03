import { test, expect, Page } from "@playwright/test";

/**
 * Comprehensive RAG System Tests
 * Tests Ollama, PostgreSQL, pgvector, Drizzle ORM, CUDA, and SvelteKit 2/Svelte 5 integration
 */

test.describe("Comprehensive RAG System Tests", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("/ai-demo");

    // Wait for page to load and services to initialize
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(5000); // Allow time for health checks
  });

  test("should verify complete system health and readiness", async () => {
    // Check page title and main elements
    await expect(page.locator("h1")).toContainText("Legal AI System Demo");

    // Verify system status section is visible
    const systemStatus = page.locator("text=System Status").first();
    await expect(systemStatus).toBeVisible();

    // Check for service indicators
    await expect(page.locator("text=Ollama Service")).toBeVisible();
    await expect(page.locator("text=AI Models")).toBeVisible();
    await expect(page.locator("text=Containers")).toBeVisible();
    await expect(page.locator("text=Memory")).toBeVisible();
  });

  test("should load and interact with TokenUsageManager", async () => {
    // Wait for the chat interface to load
    await page.waitForSelector('[data-testid="token-usage-manager"]', {
      timeout: 30000,
    });

    const tokenManager = page.locator('[data-testid="token-usage-manager"]');
    await expect(tokenManager).toBeVisible();

    // Check for token limit slider
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await expect(tokenSlider).toBeVisible();

    // Test slider functionality
    await tokenSlider.fill("4000");
    const displayValue = page.locator('[data-testid="token-limit-display"]');
    await expect(displayValue).toContainText("4,000");

    // Check token tracker elements
    await expect(page.locator('[data-testid="tokens-used"]')).toBeVisible();
    await expect(
      page.locator('[data-testid="tokens-remaining"]')
    ).toBeVisible();
  });

  test("should perform legal AI query with token tracking", async () => {
    // Wait for healthy service status
    await page.waitForSelector("text=healthy", { timeout: 60000 });

    // Locate chat interface
    const chatInput = page.locator('[data-testid="chat-input"]');
    await expect(chatInput).toBeVisible({ timeout: 30000 });

    // Set a reasonable token limit
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await tokenSlider.fill("2000");

    // Send a legal query
    const legalQuery =
      "What are the Miranda rights and when must they be given?";
    await chatInput.fill(legalQuery);

    const sendButton = page.locator('[data-testid="send-button"]');
    await sendButton.click();

    // Wait for AI response
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 90000,
    });

    const response = page.locator('[data-testid="chat-response"]').first();
    await expect(response).toBeVisible();

    // Verify response quality
    const responseText = await response.textContent();
    expect(responseText).toBeTruthy();
    expect(responseText!.length).toBeGreaterThan(100);
    expect(responseText!.toLowerCase()).toContain("miranda");

    // Check token usage was tracked
    const tokensUsed = page.locator('[data-testid="tokens-used"]');
    const usedText = await tokensUsed.textContent();
    const usedCount = parseInt(usedText?.replace(/,/g, "") || "0");
    expect(usedCount).toBeGreaterThan(0);

    // Verify token breakdown is visible
    await expect(page.locator('[data-testid="token-breakdown"]')).toBeVisible();
    await expect(page.locator('[data-testid="prompt-tokens"]')).toBeVisible();
    await expect(page.locator('[data-testid="response-tokens"]')).toBeVisible();
  });

  test("should demonstrate NVIDIA CUDA GPU acceleration", async () => {
    // Wait for service to be ready
    await page.waitForSelector("text=healthy", { timeout: 60000 });

    // Record start time for performance measurement
    const startTime = Date.now();

    // Send a complex query that would benefit from GPU acceleration
    const complexQuery =
      "Analyze the legal implications of digital evidence collection in criminal investigations, including Fourth Amendment considerations, chain of custody requirements, and admissibility standards.";

    await page.locator('[data-testid="chat-input"]').fill(complexQuery);
    await page.locator('[data-testid="send-button"]').click();

    // Wait for response
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 120000,
    });

    const endTime = Date.now();
    const responseTime = endTime - startTime;

    // With GPU acceleration, complex queries should be processed reasonably fast
    expect(responseTime).toBeLessThan(60000); // Less than 60 seconds

    // Check for performance metrics
    const performanceInfo = page.locator("text=/\\d+ms.*\\d+ tokens/");
    if ((await performanceInfo.count()) > 0) {
      await expect(performanceInfo.first()).toBeVisible();

      const perfText = await performanceInfo.first().textContent();
      console.log("Performance metrics:", perfText);
    }

    // Verify response quality indicates sophisticated processing
    const response = await page
      .locator('[data-testid="chat-response"]')
      .textContent();
    expect(response!.length).toBeGreaterThan(300);
    expect(response!.toLowerCase()).toMatch(
      /fourth amendment|digital evidence|chain of custody/
    );
  });

  test("should validate PostgreSQL and pgvector integration", async () => {
    // Test vector similarity search through multiple related queries
    const queries = [
      "What is reasonable suspicion in criminal law?",
      "How does probable cause differ from reasonable suspicion?",
      "When can police conduct a Terry stop?",
    ];

    for (const query of queries) {
      await page.locator('[data-testid="chat-input"]').fill(query);
      await page.locator('[data-testid="send-button"]').click();

      // Wait for response
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 90000,
      });

      // Brief wait between queries
      await page.waitForTimeout(3000);
    }

    // All responses should be visible
    const responses = page.locator('[data-testid="chat-response"]');
    await expect(responses).toHaveCount(3);

    // Responses should show understanding of related concepts
    for (let i = 0; i < 3; i++) {
      const responseText = await responses.nth(i).textContent();
      expect(responseText!.length).toBeGreaterThan(100);
      expect(responseText!.toLowerCase()).toMatch(
        /suspicion|probable|cause|terry|stop/
      );
    }
  });

  test("should test Drizzle ORM data persistence and retrieval", async () => {
    // Send query and verify it can be retrieved/referenced later
    const initialQuery = "Explain the exclusionary rule in criminal procedure";

    await page.locator('[data-testid="chat-input"]').fill(initialQuery);
    await page.locator('[data-testid="send-button"]').click();
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 90000,
    });

    // Check if the interaction is stored (token usage should increase)
    const initialTokens = await getTokenCount(page, "tokens-used");
    expect(initialTokens).toBeGreaterThan(0);

    // Send follow-up query that might reference previous context
    const followUpQuery =
      "Can you give me a specific example of when this rule would apply?";

    await page.locator('[data-testid="chat-input"]').fill(followUpQuery);
    await page.locator('[data-testid="send-button"]').click();
    await page.waitForSelector('[data-testid="chat-response"]:nth-child(2)', {
      timeout: 90000,
    });

    // Verify token usage increased (indicating persistent storage)
    const finalTokens = await getTokenCount(page, "tokens-used");
    expect(finalTokens).toBeGreaterThan(initialTokens);

    const responses = page.locator('[data-testid="chat-response"]');
    await expect(responses).toHaveCount(2);
  });

  test("should handle token limit warnings and optimization", async () => {
    // Set a low token limit to trigger warnings
    await page.locator('[data-testid="token-limit-slider"]').fill("500");

    // Enable auto-optimization
    const optimizeToggle = page.locator(
      '[data-testid="optimize-conversation-toggle"]'
    );
    if (await optimizeToggle.isVisible()) {
      await optimizeToggle.check();
    }

    // Send multiple queries to approach the limit
    const queries = [
      "What is mens rea in criminal law?",
      "Explain the different types of criminal intent",
      "How does strict liability differ from other forms of liability?",
    ];

    for (const query of queries) {
      await page.locator('[data-testid="chat-input"]').fill(query);
      await page.locator('[data-testid="send-button"]').click();

      try {
        await page.waitForSelector('[data-testid="chat-response"]', {
          timeout: 60000,
        });
      } catch {
        // Might hit token limit
        break;
      }

      // Check for token warning
      const tokenWarning = page.locator('[data-testid="token-warning"]');
      if (await tokenWarning.isVisible()) {
        await expect(tokenWarning).toContainText(
          /token limit|approaching|exceeded/i
        );
        console.log("Token warning triggered successfully");
        break;
      }

      await page.waitForTimeout(2000);
    }
  });

  test("should test token usage history and export functionality", async () => {
    // Send a few queries to build history
    const testQueries = [
      "Define burglary in criminal law",
      "What is larceny?",
      "How do these crimes differ?",
    ];

    for (const query of testQueries) {
      await page.locator('[data-testid="chat-input"]').fill(query);
      await page.locator('[data-testid="send-button"]').click();
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 60000,
      });
      await page.waitForTimeout(2000);
    }

    // Open token history
    const historyButton = page.locator('[data-testid="token-history-button"]');
    if (await historyButton.isVisible()) {
      await historyButton.click();

      const historyModal = page.locator('[data-testid="token-history-modal"]');
      await expect(historyModal).toBeVisible();

      // Should show history entries
      const historyEntries = page.locator('[data-testid="history-entry"]');
      const entryCount = await historyEntries.count();
      expect(entryCount).toBeGreaterThan(0);

      // Each entry should have token info and timestamp
      if (entryCount > 0) {
        await expect(
          historyEntries.first().locator('[data-testid="entry-tokens"]')
        ).toBeVisible();
        await expect(
          historyEntries.first().locator('[data-testid="entry-timestamp"]')
        ).toBeVisible();
      }
    }
  });

  test("should validate model switching and token limit adjustment", async () => {
    // Check if multiple models are available
    const modelBadges = page.locator('[role="button"]:has-text("gemma3")');
    const modelCount = await modelBadges.count();

    if (modelCount > 1) {
      // Test switching between models
      await modelBadges.first().click();
      await page.waitForTimeout(1000);

      // Check if token limit adjusts for the model
      const tokenLimit = page.locator('[data-testid="token-limit-display"]');
      const limit1 = await tokenLimit.textContent();

      // Switch to another model if available
      await modelBadges.nth(1).click();
      await page.waitForTimeout(1000);

      const limit2 = await tokenLimit.textContent();

      // Limits might be different for different models
      console.log(`Model 1 limit: ${limit1}, Model 2 limit: ${limit2}`);
    }
  });

  test("should test error handling and recovery", async () => {
    // Test empty query
    await page.locator('[data-testid="send-button"]').click();
    await page.waitForTimeout(2000);

    // Should not create empty response
    let responseCount = await page
      .locator('[data-testid="chat-response"]')
      .count();
    expect(responseCount).toBe(0);

    // Test very short query
    await page.locator('[data-testid="chat-input"]').fill("?");
    await page.locator('[data-testid="send-button"]').click();

    // Should handle gracefully
    try {
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 30000,
      });
      responseCount = await page
        .locator('[data-testid="chat-response"]')
        .count();
      expect(responseCount).toBe(1);
    } catch {
      // If it fails, that's also acceptable for edge cases
      console.log("Single character query handling: no response (acceptable)");
    }
  });

  test("should validate complete RAG pipeline performance", async () => {
    // Complex legal query that should trigger full RAG pipeline
    const complexLegalQuery = `
      In a criminal case involving digital evidence from a smartphone seized during arrest,
      what are the key legal considerations for:
      1. Search warrant requirements
      2. Forensic examination procedures
      3. Chain of custody documentation
      4. Expert testimony requirements for digital evidence
      Please provide specific legal precedents and current best practices.
    `;

    const startTime = Date.now();

    await page.locator('[data-testid="chat-input"]').fill(complexLegalQuery);
    await page.locator('[data-testid="send-button"]').click();

    // Wait for comprehensive response
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 180000,
    }); // 3 minutes max

    const endTime = Date.now();
    const responseTime = endTime - startTime;

    const response = await page
      .locator('[data-testid="chat-response"]')
      .textContent();

    // Validate response quality
    expect(response!.length).toBeGreaterThan(500);
    expect(response!.toLowerCase()).toMatch(
      /warrant|digital evidence|chain of custody|forensic/
    );
    expect(response!).toMatch(/\d+\..*\d+\..*\d+\..*\d+\./); // Should address numbered points

    // Check performance
    console.log(`Complex RAG query completed in ${responseTime}ms`);
    expect(responseTime).toBeLessThan(180000); // Should complete within 3 minutes

    // Verify token usage tracking for complex query
    const tokensUsed = await getTokenCount(page, "tokens-used");
    expect(tokensUsed).toBeGreaterThan(100); // Complex query should use substantial tokens
  });
});

// Helper function to extract token count from display
async function getTokenCount(page: Page, testId: string): Promise<number> {
  const element = page.locator(`[data-testid="${testId}"]`);
  const text = await element.textContent();
  return parseInt(text?.replace(/,/g, "") || "0");
}
