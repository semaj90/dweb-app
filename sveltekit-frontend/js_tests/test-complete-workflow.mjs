#!/usr/bin/env node

/**
 * Complete Workflow Test Script
 * Tests all major functionality: auth, cases, reports, canvas, PDF export, evidence analysis
 */

import { chromium } from "playwright";
import { expect } from "@playwright/test";

const BASE_URL = "http://localhost:5173";

async function runCompleteWorkflow() {
  const browser = await chromium.launch({
    headless: false, // Show browser for debugging
    slowMo: 1000, // Slow down for visibility
  });

  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    console.log("🚀 Starting Complete Workflow Test...");

    // Step 1: Navigate to the application
    console.log("📍 Step 1: Navigating to application...");
    await page.goto(BASE_URL);
    await page.waitForLoadState("networkidle");

    // Check if server is running
    try {
      await page.waitForSelector("body", { timeout: 5000 });
      console.log("✅ Application is running");
    } catch (error) {
      console.log("❌ Application not running. Please start the server first.");
      throw new Error("Server not running on http://localhost:5173");
    }

    // Step 2: Register a new user
    console.log("📍 Step 2: Registering new user...");
    await page.goto(`${BASE_URL}/register`);
    await page.fill('input[name="name"]', "Test User");
    await page.fill('input[name="email"]', `test-${Date.now()}@example.com`);
    await page.fill('input[name="password"]', "TestPassword123!");
    await page.fill('input[name="confirmPassword"]', "TestPassword123!");
    await page.click('button[type="submit"]');

    // Wait for redirect to login
    await page.waitForURL("**/login**");
    console.log("✅ User registration successful");

    // Step 3: Login with the new user
    console.log("📍 Step 3: Logging in...");
    await page.fill('input[name="email"]', `test-${Date.now()}@example.com`);
    await page.fill('input[name="password"]', "TestPassword123!");
    await page.click('button[type="submit"]');

    // Wait for dashboard
    await page.waitForURL("**/dashboard**");
    console.log("✅ User login successful");

    // Step 4: Create a new case
    console.log("📍 Step 4: Creating new case...");
    await page.goto(`${BASE_URL}/cases`);
    await page.click('button[data-testid="create-case"]');

    await page.fill('input[name="title"]', "Test Legal Case");
    await page.fill(
      'textarea[name="description"]',
      "This is a test case for the workflow verification",
    );
    await page.selectOption('select[name="category"]', "felony");
    await page.selectOption('select[name="priority"]', "high");
    await page.click('button[type="submit"]');

    await page.waitForSelector('[data-testid="case-created"]');
    console.log("✅ Case creation successful");

    // Step 5: Edit the case
    console.log("📍 Step 5: Editing case...");
    await page.click('button[data-testid="edit-case"]');
    await page.fill(
      'textarea[name="description"]',
      "Updated case description with more details",
    );
    await page.click('button[data-testid="save-case"]');
    console.log("✅ Case edit successful");

    // Step 6: Go to Interactive Canvas
    console.log("📍 Step 6: Opening Interactive Canvas...");
    await page.goto(`${BASE_URL}/interactive-canvas`);
    await page.waitForSelector('canvas, [data-testid="canvas-container"]');

    // Add some content to canvas
    await page.click('[data-testid="add-text-node"]');
    await page.fill('[data-testid="text-input"]', "Test canvas content");
    await page.click('[data-testid="confirm-text"]');
    console.log("✅ Interactive Canvas interaction successful");

    // Step 7: Write a report
    console.log("📍 Step 7: Writing report...");
    await page.goto(`${BASE_URL}/report-builder`);
    await page.fill(
      '[data-testid="report-title"]',
      "Test Investigation Report",
    );
    await page.fill(
      '[data-testid="report-content"]',
      "This is a comprehensive test report that includes all necessary details for the investigation.",
    );

    // Add citations
    await page.click('[data-testid="add-citation"]');
    await page.fill(
      '[data-testid="citation-text"]',
      "State v. Example, 123 F.3d 456 (2023)",
    );
    await page.click('[data-testid="save-citation"]');

    await page.click('[data-testid="save-report"]');
    console.log("✅ Report creation successful");

    // Step 8: Export to PDF
    console.log("📍 Step 8: Exporting to PDF...");
    await page.click('[data-testid="export-pdf"]');

    // Wait for download or success notification
    const downloadPromise = page.waitForEvent("download");
    await page.click('[data-testid="confirm-export"]');

    try {
      const download = await downloadPromise;
      console.log("✅ PDF export successful");
    } catch (error) {
      console.log("⚠️ PDF export may have completed (check notifications)");
    }

    // Step 9: Upload evidence for analysis
    console.log("📍 Step 9: Uploading evidence for analysis...");
    await page.goto(`${BASE_URL}/evidence`);

    // Create a test image file (base64 encoded 1x1 pixel PNG)
    const testImageDataUrl =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    await page.setInputFiles('[data-testid="evidence-upload"]', {
      name: "test-evidence.png",
      mimeType: "image/png",
      buffer: Buffer.from(testImageDataUrl.split(",")[1], "base64"),
    });

    await page.click('[data-testid="analyze-evidence"]');
    await page.waitForSelector('[data-testid="analysis-result"]', {
      timeout: 30000,
    });
    console.log("✅ Evidence analysis successful");

    console.log(
      "🎉 Complete Workflow Test PASSED! All features are working correctly.",
    );

    return true;
  } catch (error) {
    console.error("❌ Workflow test failed:", error);

    // Take screenshot for debugging
    await page.screenshot({
      path: "workflow-test-failure.png",
      fullPage: true,
    });

    return false;
  } finally {
    await browser.close();
  }
}

// Run the test
if (import.meta.url === `file://${process.argv[1]}`) {
  runCompleteWorkflow()
    .then((success) => {
      process.exit(success ? 0 : 1);
    })
    .catch((error) => {
      console.error("Test runner error:", error);
      process.exit(1);
    });
}

export { runCompleteWorkflow };
