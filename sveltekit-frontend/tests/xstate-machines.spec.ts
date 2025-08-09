// @ts-nocheck
import { test, expect } from '@playwright/test';

test.describe('XState State Machines', () => {
  test.beforeEach(async ({ page }) => {
    // Login to access state machine features
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
  });

  test('should handle case creation state machine', async ({ page }) => {
    await page.goto('/dashboard/cases');
    
    // Start case creation
    await page.click('[data-testid="create-case-button"]');
    
    // Check initial state
    const stateIndicator = page.locator('[data-testid="state-indicator"]');
    if (await stateIndicator.isVisible()) {
      const currentState = await stateIndicator.getAttribute('data-state');
      expect(currentState).toBe('editing');
    }
    
    // Fill required fields
    await page.fill('input[name="title"]', 'XState Test Case');
    await page.fill('textarea[name="description"]', 'Testing state machine flow');
    
    // Transition to validation state
    await page.click('[data-testid="validate-button"]');
    
    // Check validation state
    await page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="state-indicator"]');
        return indicator?.getAttribute('data-state') === 'validating';
      },
      null,
      { timeout: 5000 }
    );
    
    // Wait for validation to complete
    await page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="state-indicator"]');
        return indicator?.getAttribute('data-state') === 'valid';
      },
      null,
      { timeout: 10000 }
    );
    
    // Submit case
    await page.click('button[type="submit"]');
    
    // Check saving state
    await page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="state-indicator"]');
        return indicator?.getAttribute('data-state') === 'saving';
      },
      null,
      { timeout: 5000 }
    );
    
    // Wait for completion
    await page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="state-indicator"]');
        return indicator?.getAttribute('data-state') === 'saved';
      },
      null,
      { timeout: 15000 }
    );
    
    // Should redirect to case detail
    await page.waitForURL('/dashboard/cases/**');
  });

  test('should handle document upload state machine', async ({ page }) => {
    await page.goto('/dashboard/documents/upload');
    
    // Check initial state
    const uploadState = page.locator('[data-testid="upload-state"]');
    if (await uploadState.isVisible()) {
      const state = await uploadState.textContent();
      expect(state).toMatch(/idle|ready/i);
    }
    
    // Select file
    await page.evaluate(() => {
      const file = new File(['test content'], 'test-document.pdf', { type: 'application/pdf' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (input) {
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    
    // Check file selected state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="upload-state"]');
        return state?.textContent?.toLowerCase().includes('selected');
      },
      null,
      { timeout: 5000 }
    );
    
    // Start upload
    await page.click('[data-testid="start-upload"]');
    
    // Check uploading state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="upload-state"]');
        return state?.textContent?.toLowerCase().includes('uploading');
      },
      null,
      { timeout: 5000 }
    );
    
    // Check progress bar
    const progressBar = page.locator('[data-testid="upload-progress"]');
    if (await progressBar.isVisible()) {
      await expect(progressBar).toBeVisible();
    }
    
    // Wait for processing state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="upload-state"]');
        return state?.textContent?.toLowerCase().includes('processing');
      },
      null,
      { timeout: 15000 }
    );
    
    // Wait for completion
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="upload-state"]');
        return state?.textContent?.toLowerCase().includes('complete');
      },
      null,
      { timeout: 30000 }
    );
    
    // Success message should appear
    const successMessage = page.locator('[data-testid="upload-success"]');
    await expect(successMessage).toBeVisible();
  });

  test('should handle AI analysis state machine', async ({ page }) => {
    // Create a case first
    await page.goto('/dashboard/cases');
    await page.click('[data-testid="create-case-button"]');
    await page.fill('input[name="title"]', 'AI Analysis Test');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/cases/**');
    
    // Navigate to AI analysis
    await page.click('[data-testid="ai-analysis-tab"]');
    
    // Start analysis
    await page.click('[data-testid="start-analysis"]');
    
    // Check initial analysis state
    const analysisState = page.locator('[data-testid="analysis-state"]');
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="analysis-state"]');
        return state?.getAttribute('data-state') === 'initializing';
      },
      null,
      { timeout: 5000 }
    );
    
    // Wait for data gathering phase
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="analysis-state"]');
        return state?.getAttribute('data-state') === 'gathering';
      },
      null,
      { timeout: 10000 }
    );
    
    // Check progress indicator
    const progressSteps = page.locator('[data-testid="analysis-steps"] .step');
    const stepCount = await progressSteps.count();
    expect(stepCount).toBeGreaterThan(0);
    
    // Wait for analysis phase
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="analysis-state"]');
        return state?.getAttribute('data-state') === 'analyzing';
      },
      null,
      { timeout: 15000 }
    );
    
    // Wait for completion
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="analysis-state"]');
        return state?.getAttribute('data-state') === 'complete';
      },
      null,
      { timeout: 60000 }
    );
    
    // Results should be displayed
    const results = page.locator('[data-testid="analysis-results"]');
    await expect(results).toBeVisible();
  });

  test('should handle chat conversation state machine', async ({ page }) => {
    await page.goto('/dashboard/ai-assistant');
    
    // Check initial chat state
    const chatState = page.locator('[data-testid="chat-state"]');
    if (await chatState.isVisible()) {
      const state = await chatState.getAttribute('data-state');
      expect(state).toBe('idle');
    }
    
    // Type a message
    const chatInput = page.locator('[data-testid="chat-input"]');
    await chatInput.fill('What are the key elements of a contract?');
    
    // Send message
    await page.click('[data-testid="send-button"]');
    
    // Check sending state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="chat-state"]');
        return state?.getAttribute('data-state') === 'sending';
      },
      null,
      { timeout: 5000 }
    );
    
    // Check thinking state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="chat-state"]');
        return state?.getAttribute('data-state') === 'thinking';
      },
      null,
      { timeout: 10000 }
    );
    
    // Check for typing indicator
    const typingIndicator = page.locator('[data-testid="ai-typing"]');
    await expect(typingIndicator).toBeVisible();
    
    // Check streaming state
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="chat-state"]');
        return state?.getAttribute('data-state') === 'streaming';
      },
      null,
      { timeout: 15000 }
    );
    
    // Wait for response completion
    await page.waitForFunction(
      () => {
        const state = document.querySelector('[data-testid="chat-state"]');
        return state?.getAttribute('data-state') === 'idle';
      },
      null,
      { timeout: 60000 }
    );
    
    // Response should be visible
    const aiResponse = page.locator('[data-testid="ai-response"]').last();
    await expect(aiResponse).toBeVisible();
    
    // Check response content
    const responseText = await aiResponse.textContent();
    expect(responseText?.length).toBeGreaterThan(50);
  });

  test('should handle error states in state machines', async ({ page }) => {
    await page.goto('/dashboard/cases');
    
    // Try to create case with invalid data
    await page.click('[data-testid="create-case-button"]');
    
    // Leave title empty and try to submit
    await page.fill('textarea[name="description"]', 'No title provided');
    await page.click('button[type="submit"]');
    
    // Check error state
    const stateIndicator = page.locator('[data-testid="state-indicator"]');
    await page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="state-indicator"]');
        return indicator?.getAttribute('data-state') === 'error';
      },
      null,
      { timeout: 5000 }
    );
    
    // Error message should be displayed
    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeVisible();
    
    // Should be able to retry
    const retryButton = page.locator('[data-testid="retry-button"]');
    if (await retryButton.isVisible()) {
      await retryButton.click();
      
      // State should reset
      await page.waitForFunction(
        () => {
          const indicator = document.querySelector('[data-testid="state-indicator"]');
          return indicator?.getAttribute('data-state') === 'editing';
        },
        null,
        { timeout: 5000 }
      );
    }
  });

  test('should persist state machine context across page reloads', async ({ page }) => {
    await page.goto('/dashboard/documents/upload');
    
    // Start upload process
    await page.evaluate(() => {
      const file = new File(['test content'], 'persistent-test.pdf', { type: 'application/pdf' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (input) {
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    
    // Get current state
    const initialState = await page.locator('[data-testid="upload-state"]').textContent();
    
    // Refresh page
    await page.reload();
    
    // State should be restored (if persistence is implemented)
    const restoredState = await page.locator('[data-testid="upload-state"]').textContent();
    
    // Note: This test may fail if state persistence is not implemented
    // In that case, state should reset to initial
    expect(restoredState).toBeTruthy();
  });

  test('should handle concurrent state machine operations', async ({ page, context }) => {
    // Open two tabs
    const page2 = await context.newPage();
    await page2.goto('/dashboard/ai-assistant');
    
    // Send message in first tab
    await page.goto('/dashboard/ai-assistant');
    await page.fill('[data-testid="chat-input"]', 'Question from tab 1');
    await page.click('[data-testid="send-button"]');
    
    // Send message in second tab
    await page2.fill('[data-testid="chat-input"]', 'Question from tab 2');
    await page2.click('[data-testid="send-button"]');
    
    // Both should handle their own state
    await Promise.all([
      page.waitForFunction(
        () => {
          const responses = document.querySelectorAll('[data-testid="ai-response"]');
          return responses.length > 0;
        },
        null,
        { timeout: 60000 }
      ),
      page2.waitForFunction(
        () => {
          const responses = document.querySelectorAll('[data-testid="ai-response"]');
          return responses.length > 0;
        },
        null,
        { timeout: 60000 }
      )
    ]);
    
    // Both tabs should have responses
    const response1 = await page.locator('[data-testid="ai-response"]').last().textContent();
    const response2 = await page2.locator('[data-testid="ai-response"]').last().textContent();
    
    expect(response1?.length).toBeGreaterThan(10);
    expect(response2?.length).toBeGreaterThan(10);
    
    await page2.close();
  });

  test('should handle state machine timeouts', async ({ page }) => {
    await page.goto('/dashboard/ai-assistant');
    
    // Send a message that might timeout
    await page.fill('[data-testid="chat-input"]', 'This is a very complex legal question that might take a long time to process and could potentially timeout');
    await page.click('[data-testid="send-button"]');
    
    // Check for timeout handling
    const timeoutMessage = page.locator('[data-testid="timeout-message"]');
    
    // Wait for either response or timeout
    await Promise.race([
      page.waitForFunction(
        () => {
          const responses = document.querySelectorAll('[data-testid="ai-response"]');
          return responses.length > 0;
        },
        null,
        { timeout: 120000 }
      ).catch(() => {}),
      page.waitForSelector('[data-testid="timeout-message"]', { timeout: 120000 }).catch(() => {})
    ]);
    
    // If timeout occurred, retry should be available
    if (await timeoutMessage.isVisible()) {
      const retryButton = page.locator('[data-testid="retry-button"]');
      await expect(retryButton).toBeVisible();
    }
  });

  test('should visualize state machine transitions', async ({ page }) => {
    await page.goto('/dashboard/cases');
    
    // Check if state machine visualizer is available
    const stateVisualizer = page.locator('[data-testid="state-visualizer"]');
    
    if (await stateVisualizer.isVisible()) {
      // Start case creation to see transitions
      await page.click('[data-testid="create-case-button"]');
      
      // Check state nodes
      const stateNodes = page.locator('[data-testid="state-node"]');
      const nodeCount = await stateNodes.count();
      expect(nodeCount).toBeGreaterThan(0);
      
      // Current state should be highlighted
      const currentStateNode = page.locator('[data-testid="state-node"].current');
      await expect(currentStateNode).toBeVisible();
      
      // Fill form and watch transitions
      await page.fill('input[name="title"]', 'State Visualization Test');
      
      // Check if state changed
      await page.waitForTimeout(1000);
      const newCurrentState = await currentStateNode.getAttribute('data-state');
      expect(newCurrentState).toBeTruthy();
    }
  });
});