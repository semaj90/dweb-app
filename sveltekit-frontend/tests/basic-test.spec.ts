import { test, expect } from "@playwright/test";

test.describe("Basic App Test", () => {
  test("Check if homepage loads", async ({ page }) => {
    // Add more timeout for potentially slow dev server
    await page.goto("/", { timeout: 30000 });

    // Check if page loads without errors
    await expect(page).toHaveURL(/\//);

    // Log page title and content for debugging
    const title = await page.title();
    console.log("Page title:", title);

    // Take a screenshot for debugging
    await page.screenshot({ path: "homepage-debug.png" });

    // Check if there's any content
    const body = await page.locator("body").textContent();
    console.log("Page has content:", body && body.length > 0);

    // Basic assertion - page should not be completely empty
    expect(body && body.length > 10).toBeTruthy();
  });

  test("Check if login page loads", async ({ page }) => {
    await page.goto("/login", { timeout: 30000 });

    // Check if we're on login page
    await expect(page).toHaveURL(/.*login/);

    // Take screenshot for debugging
    await page.screenshot({ path: "login-debug.png" });

    // Log page content
    const content = await page.locator("body").textContent();
    console.log("Login page content length:", content?.length || 0);
  });
});
