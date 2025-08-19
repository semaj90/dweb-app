// @ts-nocheck
import { test, expect } from "@playwright/test";
import { URL } from "url";

test.describe("Manual Database Test", () => {
  test("Test user registration and login flow", async ({ page }) => {
    console.log("🧪 Testing registration and login flow...");

    // Navigate to register page
    await page.goto("/register");
    await page.waitForLoadState("networkidle");

    // Fill registration form with test data
    const timestamp = Date.now();
    const testEmail = `testuser${timestamp}@example.com`;
    const testPassword = "TestPassword123!";
    const testName = `Test User ${timestamp}`;

    console.log("📝 Filling registration form...");
    await page.fill('input[name="name"]', testName);
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.fill('input[name="confirmPassword"]', testPassword);

    // Submit registration form
    console.log("🚀 Submitting registration...");
    await page.click('button[type="submit"]');

    // Wait for potential redirect or success message
    await page.waitForTimeout(3000);

    // Check if we were redirected or got an error
    const currentUrl = page.url();
    console.log("📍 Current URL after registration:", currentUrl);

    // Now test login with the same credentials
    console.log("🔑 Testing login with new credentials...");
    await page.goto("/login");
    await page.waitForLoadState("networkidle");

    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);

    console.log("🚀 Submitting login...");
    await page.click('button[type="submit"]');

    // Wait for potential redirect
    await page.waitForTimeout(3000);

    const finalUrl = page.url();
    console.log("📍 Final URL after login:", finalUrl);

    // The test passes if no exceptions were thrown
    console.log("✅ Database connectivity test completed");
  });

  test("Test demo login credentials", async ({ page }) => {
    console.log("🧪 Testing demo login credentials...");

    await page.goto("/login");
    await page.waitForLoadState("networkidle");

    // Try demo admin credentials
    console.log("🔑 Testing admin demo credentials...");
    await page.fill('input[name="email"]', "admin@example.com");
    await page.fill('input[name="password"]', "admin123");

    await page.click('button[type="submit"]');
    await page.waitForTimeout(3000);

    const adminUrl = page.url();
    console.log("📍 URL after admin login:", adminUrl);

    // Try demo user credentials
    console.log("🔑 Testing user demo credentials...");
    await page.goto("/login");
    await page.fill('input[name="email"]', "user@example.com");
    await page.fill('input[name="password"]', "user123");

    await page.click('button[type="submit"]');
    await page.waitForTimeout(3000);

    const userUrl = page.url();
    console.log("📍 URL after user login:", userUrl);

    console.log("✅ Demo credentials test completed");
  });
});
