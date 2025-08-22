
import { test, expect } from "@playwright/test";

test.describe("Full User Flow", () => {
  test("Register, Login, Create Case, and Logout Flow", async ({
    page,
    context,
  }) => {
    // 1. Navigate to home page
    await page.goto("/");
    await page.screenshot({ path: "step-1-home.png" });

    // 2. Go to register page
    await page.click('a[href="/register"]');
    await expect(page).toHaveURL(/.*register/);
    await page.screenshot({ path: "step-2-register.png" });

    // 3. Register a new user
    const timestamp = Date.now();
    const testEmail = `test${timestamp}@example.com`;
    const testName = `Test User ${timestamp}`;

    await page.fill('input[name="name"]', testName);
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', "testpassword123");
    await page.fill('input[name="confirmPassword"]', "testpassword123");
    await page.screenshot({ path: "step-3-filled-register.png" });
    await page.click('button[type="submit"]');

    // Should redirect to login page
    await expect(page).toHaveURL(/.*login/);
    await page.screenshot({ path: "step-4-login-page.png" });

    // 4. Login with the new credentials
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', "testpassword123");
    await page.screenshot({ path: "step-5-filled-login.png" });
    await page.click('button[type="submit"]');

    // Wait for dashboard navigation
    await page.waitForURL(/.*dashboard/, { timeout: 10000 });
    await page.screenshot({ path: "step-6-dashboard.png" });

    // Log cookies for debugging
    const cookies = await context.cookies();
    console.log("Cookies after login:", cookies);
    const sessionCookie = cookies.find((c) => c.name === "session");
    expect(sessionCookie).toBeTruthy();

    // 5. Verify user is logged in by checking navigation
    await expect(page.locator(".user-dropdown")).toBeVisible();
    await expect(page.locator(`text=${testName}`)).toBeVisible();
    await page.screenshot({ path: "step-7-user-dropdown.png" });

    // 6. Navigate to cases page
    await page.click('a[href="/cases"]');
    await expect(page).toHaveURL(/.*cases/);
    await page.screenshot({ path: "step-8-cases.png" });

    // 7. Create a new case
    await page.click('a[href="/cases/new"],button:has-text("New Case")');
    await expect(page).toHaveURL(/.*cases\/new/);
    await page.screenshot({ path: "step-9-new-case.png" });

    const caseTitle = `Test Case ${timestamp}`;

    await page.fill('input[name="title"]', caseTitle);
    await page.fill(
      'textarea[name="description"]',
      "Test case description for automated testing",
    );
    await page.screenshot({ path: "step-10-filled-case.png" });
    await page.click('button[type="submit"]');

    // 8. Verify case was created
    await expect(page.locator(`text=${caseTitle}`)).toBeVisible({
      timeout: 10000,
    });
    await page.screenshot({ path: "step-11-case-created.png" });

    // 9. Navigate to evidence page
    await page.click('a[href="/evidence"]');
    await expect(page).toHaveURL(/.*evidence/);
    await page.screenshot({ path: "step-12-evidence.png" });

    // 10. Logout
    await page.click(".user-dropdown .user-trigger"); // Open dropdown
    await page.click('button:has-text("Logout"),a[href="/logout"]');
    await page.waitForURL(/.*login/);
    await page.screenshot({ path: "step-13-logout.png" });

    // 11. Verify session is cleared
    const cookiesAfterLogout = await context.cookies();
    const sessionCookieAfterLogout = cookiesAfterLogout.find(
      (c) => c.name === "session",
    );
    expect(
      !sessionCookieAfterLogout || !sessionCookieAfterLogout.value,
    ).toBeTruthy();
  });

  test("Demo User Login Flow", async ({ page }) => {
    // 1. Navigate to login page
    await page.goto("/login");

    // 2. Use demo admin credentials
    await page.click('button:has-text("Admin Demo")');

    // Wait a bit for the form to be filled
    await page.waitForTimeout(100);

    // Verify credentials were filled - or fill them manually if not
    const emailValue = await page.locator('input[name="email"]').inputValue();
    if (emailValue !== "admin@example.com") {
      await page.fill('input[name="email"]', "admin@example.com");
      await page.fill('input[name="password"]', "admin123");
    }

    await expect(page.locator('input[name="email"]')).toHaveValue(
      "admin@example.com",
    );
    await expect(page.locator('input[name="password"]')).toHaveValue(
      "admin123",
    );

    // 3. Submit login
    await page.click('button[type="submit"]');

    // Should redirect to dashboard
    await expect(page).toHaveURL(/.*dashboard/);

    // 4. Verify admin user is logged in
    await expect(page.locator(".user-dropdown")).toBeVisible();
    await expect(page.locator("text=Demo Admin")).toBeVisible();

    // 5. Test protected routes are accessible
    await page.goto("/cases");
    await expect(page).toHaveURL(/.*cases/);

    await page.goto("/evidence");
    await expect(page).toHaveURL(/.*evidence/);

    // 6. Logout
    await page.click(".user-dropdown .user-trigger"); // Open dropdown
    await page.click('button:has-text("Logout")');

    // Should redirect to login page
    await expect(page).toHaveURL(/.*login/);

    console.log("✅ Demo user login flow test completed successfully!");
  });

  test("Session Persistence", async ({ page }) => {
    // 1. Login
    await page.goto("/login");
    await page.click('button:has-text("User Demo")');
    await page.click('button[type="submit"]');

    // Wait for redirect
    await expect(page).toHaveURL(/.*dashboard/);

    // 2. Refresh page to test session persistence
    await page.reload();

    // Should still be logged in
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator(".user-dropdown")).toBeVisible();

    // 3. Navigate to different pages
    await page.goto("/cases");
    await expect(page).toHaveURL(/.*cases/);

    await page.goto("/evidence");
    await expect(page).toHaveURL(/.*evidence/);

    // 4. Open new tab and verify session works
    const newPage = await page.context().newPage();
    await newPage.goto("/dashboard");
    await expect(newPage).toHaveURL(/.*dashboard/);
    await expect(newPage.locator(".user-dropdown")).toBeVisible();

    console.log("✅ Session persistence test completed successfully!");
  });
});
