import { test, expect, type Page } from "@playwright/test";

/**
 * Token Usage Management Tests
 * Tests token tracking, limits, and usage optimization
 */

test.describe("Token Usage Management Tests", () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    await page.goto("/ai-demo");

    // Wait for page to load
    await page.waitForLoadState("networkidle");
  });

  test("should display token usage tracker", async () => {
    const tokenTracker = page.locator('[data-testid="token-tracker"]');
    await expect(tokenTracker).toBeVisible();

    // Check for token usage display elements
    await expect(
      tokenTracker.locator('[data-testid="tokens-used"]')
    ).toBeVisible();
    await expect(
      tokenTracker.locator('[data-testid="tokens-remaining"]')
    ).toBeVisible();
    await expect(
      tokenTracker.locator('[data-testid="token-limit"]')
    ).toBeVisible();
  });

  test("should have adjustable token limit slider", async () => {
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await expect(tokenSlider).toBeVisible();

    // Get initial value
    const initialValue = await tokenSlider.inputValue();
    expect(parseInt(initialValue)).toBeGreaterThan(0);

    // Adjust slider to different values
    const testValues = [1000, 2000, 4000, 8000];

    for (const value of testValues) {
      await tokenSlider.fill(value.toString());

      // Verify the display updates
      const displayValue = page.locator('[data-testid="token-limit-display"]');
      await expect(displayValue).toContainText(value.toString());

      // Check remaining tokens calculation
      const remainingTokens = page.locator('[data-testid="tokens-remaining"]');
      const remainingText = await remainingTokens.textContent();
      const remaining = parseInt(remainingText?.match(/\d+/)?.[0] || "0");
      expect(remaining).toBeLessThanOrEqual(value);
    }
  });

  test("should track token usage during chat", async () => {
    // Set a specific token limit
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await tokenSlider.fill("2000");

    // Get initial token count
    const initialUsed = await getTokensUsed(page);

    // Send a chat message
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill(
      "What are the legal requirements for contract formation?"
    );

    await page.locator('[data-testid="send-button"]').click();

    // Wait for response
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 30000,
    });

    // Check that token usage increased
    const finalUsed = await getTokensUsed(page);
    expect(finalUsed).toBeGreaterThan(initialUsed);

    // Verify token breakdown
    const tokenBreakdown = page.locator('[data-testid="token-breakdown"]');
    await expect(tokenBreakdown).toBeVisible();

    await expect(
      tokenBreakdown.locator('[data-testid="prompt-tokens"]')
    ).toBeVisible();
    await expect(
      tokenBreakdown.locator('[data-testid="response-tokens"]')
    ).toBeVisible();
    await expect(
      tokenBreakdown.locator('[data-testid="total-tokens"]')
    ).toBeVisible();
  });

  test("should warn when approaching token limit", async () => {
    // Set a low token limit
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await tokenSlider.fill("500");

    // Send multiple messages to approach limit
    const chatInput = page.locator('[data-testid="chat-input"]');

    for (let i = 0; i < 3; i++) {
      await chatInput.fill(
        `Legal question ${i + 1}: Explain the process of discovery in litigation.`
      );
      await page.locator('[data-testid="send-button"]').click();
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 30000,
      });

      // Check for warning when tokens are running low
      const warningAlert = page.locator('[data-testid="token-warning"]');
      if (await warningAlert.isVisible()) {
        await expect(warningAlert).toContainText("token limit");
        break;
      }
    }
  });

  test("should prevent sending when token limit exceeded", async () => {
    // Set a very low token limit
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await tokenSlider.fill("100");

    // Try to send a long message
    const chatInput = page.locator('[data-testid="chat-input"]');
    const longMessage =
      "This is a very long legal question about contract law, tort law, criminal law, constitutional law, and administrative law that should exceed the token limit that we have set for this particular test case to validate that the system properly prevents sending messages when the token limit would be exceeded.";

    await chatInput.fill(longMessage);

    // Send button should be disabled or show error
    const sendButton = page.locator('[data-testid="send-button"]');
    const isDisabled = await sendButton.isDisabled();

    if (!isDisabled) {
      await sendButton.click();
      // Should show error message
      const errorMessage = page.locator('[data-testid="token-limit-error"]');
      await expect(errorMessage).toBeVisible();
      await expect(errorMessage).toContainText("token limit");
    }
  });

  test("should show token usage history", async () => {
    // Send a few test messages
    const testMessages = [
      "What is negligence in tort law?",
      "Explain contract breach remedies.",
      "Define criminal intent.",
    ];

    for (const message of testMessages) {
      const chatInput = page.locator('[data-testid="chat-input"]');
      await chatInput.fill(message);
      await page.locator('[data-testid="send-button"]').click();
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 30000,
      });
    }

    // Check token usage history
    const historyButton = page.locator('[data-testid="token-history-button"]');
    await historyButton.click();

    const historyModal = page.locator('[data-testid="token-history-modal"]');
    await expect(historyModal).toBeVisible();

    // Should show entries for each message
    const historyEntries = page.locator('[data-testid="history-entry"]');
    await expect(historyEntries.first()).toBeVisible();

    // Each entry should show tokens used
    const firstEntry = historyEntries.first();
    await expect(
      firstEntry.locator('[data-testid="entry-tokens"]')
    ).toBeVisible();
    await expect(
      firstEntry.locator('[data-testid="entry-timestamp"]')
    ).toBeVisible();
  });

  test("should optimize token usage for long conversations", async () => {
    // Enable conversation optimization
    const optimizeToggle = page.locator(
      '[data-testid="optimize-conversation-toggle"]'
    );
    await optimizeToggle.check();

    // Set moderate token limit
    const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
    await tokenSlider.fill("1500");

    // Start a long conversation
    const messages = [
      "What is the difference between civil and criminal law?",
      "Can you elaborate on the burden of proof in each?",
      "How does evidence admissibility differ between them?",
      "What about statute of limitations?",
      "Tell me more about appeals processes.",
    ];

    for (const message of messages) {
      const chatInput = page.locator('[data-testid="chat-input"]');
      await chatInput.fill(message);
      await page.locator('[data-testid="send-button"]').click();
      await page.waitForSelector('[data-testid="chat-response"]', {
        timeout: 30000,
      });
    }

    // Check optimization metrics
    const optimizationMetrics = page.locator(
      '[data-testid="optimization-metrics"]'
    );
    await expect(optimizationMetrics).toBeVisible();

    await expect(
      optimizationMetrics.locator('[data-testid="context-compressed"]')
    ).toBeVisible();
    await expect(
      optimizationMetrics.locator('[data-testid="tokens-saved"]')
    ).toBeVisible();
  });

  test("should handle different model token limits", async () => {
    const models = [
      { name: "gemma3:2b", expectedLimit: 2048 },
      { name: "gemma3:7b", expectedLimit: 4096 },
      { name: "gemma3:13b", expectedLimit: 8192 },
    ];

    for (const model of models) {
      // Select model
      const modelSelector = page.locator('[data-testid="model-selector"]');
      await modelSelector.selectOption(model.name);

      // Check that token limit auto-adjusts
      const tokenLimitDisplay = page.locator(
        '[data-testid="token-limit-display"]'
      );
      await expect(tokenLimitDisplay).toContainText(
        model.expectedLimit.toString()
      );

      // Verify slider max value updates
      const tokenSlider = page.locator('[data-testid="token-limit-slider"]');
      const maxValue = await tokenSlider.getAttribute("max");
      expect(parseInt(maxValue || "0")).toBe(model.expectedLimit);
    }
  });

  test("should export token usage analytics", async () => {
    // Generate some token usage data
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("Generate usage data for analytics test");
    await page.locator('[data-testid="send-button"]').click();
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 30000,
    });

    // Open analytics
    const analyticsButton = page.locator(
      '[data-testid="token-analytics-button"]'
    );
    await analyticsButton.click();

    const analyticsModal = page.locator('[data-testid="analytics-modal"]');
    await expect(analyticsModal).toBeVisible();

    // Check analytics data
    await expect(
      analyticsModal.locator('[data-testid="total-tokens-used"]')
    ).toBeVisible();
    await expect(
      analyticsModal.locator('[data-testid="average-tokens-per-request"]')
    ).toBeVisible();
    await expect(
      analyticsModal.locator('[data-testid="peak-usage-time"]')
    ).toBeVisible();

    // Test export functionality
    const exportButton = page.locator(
      '[data-testid="export-analytics-button"]'
    );
    await exportButton.click();

    // Wait for download (this might need adjustment based on browser)
    await page.waitForTimeout(2000);
  });

  test("should reset token counter", async () => {
    // Use some tokens
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill("Test message for token reset");
    await page.locator('[data-testid="send-button"]').click();
    await page.waitForSelector('[data-testid="chat-response"]', {
      timeout: 30000,
    });

    // Get current token usage
    const tokensUsedBefore = await getTokensUsed(page);
    expect(tokensUsedBefore).toBeGreaterThan(0);

    // Reset tokens
    const resetButton = page.locator('[data-testid="reset-tokens-button"]');
    await resetButton.click();

    // Confirm reset
    const confirmButton = page.locator('[data-testid="confirm-reset-button"]');
    await confirmButton.click();

    // Check that tokens are reset
    const tokensUsedAfter = await getTokensUsed(page);
    expect(tokensUsedAfter).toBe(0);
  });
});

// Helper function to get current token usage
async function getTokensUsed(page: Page): Promise<number> {
  const tokensUsedElement = page.locator('[data-testid="tokens-used"]');
  const text = await tokensUsedElement.textContent();
  return parseInt(text?.match(/\d+/)?.[0] || "0");
}
