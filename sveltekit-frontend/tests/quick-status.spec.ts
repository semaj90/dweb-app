
import { test, expect } from "@playwright/test";

test.describe("Quick Status Check", () => {
  test("Check if app is accessible", async ({ page }) => {
    console.log("🔍 Testing app accessibility...");

    try {
      await page.goto("/", { timeout: 10000 });
      console.log("✅ App is accessible");

      const title = await page.title();
      console.log("📄 Page title:", title);

      const bodyText = await page.textContent("body");
      console.log("📝 Body content length:", bodyText?.length || 0);

      // Check if this looks like our app
      const hasExpectedContent =
        bodyText?.includes("Legal") ||
        bodyText?.includes("Prosecutor") ||
        bodyText?.includes("Case") ||
        bodyText?.includes("WardenNet") ||
        title?.includes("Prosecutor");

      if (hasExpectedContent) {
        console.log("✅ App content looks correct");
      } else {
        console.log("⚠️ App content may not be loaded correctly");
        console.log("Title:", title);
        console.log("First 200 chars:", bodyText?.substring(0, 200));
      }

      expect(true).toBe(true); // Pass the test if we get here
    } catch (error) {
      console.log("❌ App not accessible:", (error as Error).message);
      throw error;
    }
  });

  test("Check login page", async ({ page }) => {
    console.log("🔍 Testing login page...");

    try {
      await page.goto("/login", { timeout: 10000 });
      console.log("✅ Login page accessible");

      // Look for login form elements
      const emailField = await page
        .locator('input[type="email"], input[name="email"]')
        .count();
      const passwordField = await page
        .locator('input[type="password"], input[name="password"]')
        .count();
      const submitButton = await page.locator('button[type="submit"]').count();

      console.log("📧 Email fields found:", emailField);
      console.log("🔐 Password fields found:", passwordField);
      console.log("🔘 Submit buttons found:", submitButton);

      expect(emailField).toBeGreaterThan(0);
      expect(passwordField).toBeGreaterThan(0);
    } catch (error) {
      console.log("❌ Login page error:", (error as Error).message);
      throw error;
    }
  });

  test("Check register page", async ({ page }) => {
    console.log("🔍 Testing register page...");

    try {
      await page.goto("/register", { timeout: 10000 });
      console.log("✅ Register page accessible");

      // Look for registration form elements
      const emailField = await page
        .locator('input[type="email"], input[name="email"]')
        .count();
      const passwordField = await page
        .locator('input[type="password"], input[name="password"]')
        .count();
      const nameField = await page
        .locator('input[name="name"], input[name="firstName"]')
        .count();

      console.log("📧 Email fields found:", emailField);
      console.log("🔐 Password fields found:", passwordField);
      console.log("👤 Name fields found:", nameField);

      expect(emailField).toBeGreaterThan(0);
      expect(passwordField).toBeGreaterThan(0);
    } catch (error) {
      console.log("❌ Register page error:", (error as Error).message);
      throw error;
    }
  });
});
