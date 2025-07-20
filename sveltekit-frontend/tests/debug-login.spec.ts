import { test, expect } from "@playwright/test";

test("Debug Login Form", async ({ page }) => {
  // Enable console logging
  page.on("console", (msg) =>
    console.log("Browser console:", msg.type(), msg.text()),
  );

  // Navigate to login page
  await page.goto("/login");

  // Fill the form manually
  await page.fill('input[name="email"]', "admin@example.com");
  await page.fill('input[name="password"]', "admin123");

  // Check what happens when we click submit
  console.log("About to click submit button...");

  // Listen for network requests
  page.on("request", (request) => {
    if (request.url().includes("/api/auth/login")) {
      console.log("API request detected:", request.method(), request.url());
      console.log("Request body:", request.postData());
    }
  });

  // Click submit and wait a bit to see what happens
  await page.click('button:has-text("Sign In")');
  await page.waitForTimeout(2000);

  // Log current URL
  console.log("Current URL after submit:", page.url());
});
