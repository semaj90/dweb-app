
import { test, expect } from '@playwright/test';
import { URL } from "url";

test.describe('User Authentication and Session Management', () => {
  const testEmail = `test-user-${Date.now()}@example.com`;
  const testPassword = 'SecureTestPass123!';
  const testName = 'Test User';

  test('should create a new user account', async ({ page }) => {
    // Navigate to registration page
    await page.goto('/register');
    
    // Fill registration form
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.fill('input[name="confirmPassword"]', testPassword);
    await page.fill('input[name="name"]', testName);
    
    // Accept terms if present
    const termsCheckbox = page.locator('input[name="terms"]');
    if (await termsCheckbox.isVisible()) {
      await termsCheckbox.check();
    }
    
    // Submit registration
    await page.click('button[type="submit"]');
    
    // Wait for redirect to dashboard or login
    await page.waitForURL(/(dashboard|login|verify-email)/);
    
    // Verify registration success
    const currentUrl = page.url();
    expect(currentUrl).toMatch(/(dashboard|login|verify-email)/);
    
    // Check for success message or email verification notice
    const successMessage = page.locator('[data-testid="success-message"], .success-message, .alert-success');
    if (await successMessage.isVisible()) {
      const message = await successMessage.textContent();
      expect(message).toMatch(/success|created|verify/i);
    }
  });

  test('should login with created account', async ({ page }) => {
    // Navigate to login page
    await page.goto('/login');
    
    // Fill login form
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    
    // Submit login
    await page.click('button[type="submit"]');
    
    // Wait for redirect to dashboard
    await page.waitForURL('/dashboard/**', { timeout: 10000 });
    
    // Verify we're logged in
    expect(page.url()).toContain('/dashboard');
    
    // Check for user info in header/nav
    const userMenu = page.locator('[data-testid="user-menu"], .user-menu, .user-avatar');
    await expect(userMenu).toBeVisible();
    
    // Verify user name is displayed
    const userName = page.locator('[data-testid="user-name"], .user-name');
    if (await userName.isVisible()) {
      const displayedName = await userName.textContent();
      expect(displayedName).toContain(testName);
    }
  });

  test('should persist session across page refreshes', async ({ page, context }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Get cookies before refresh
    const cookies = await context.cookies();
    const sessionCookie = cookies.find(c: unknown => c.name.includes('session') || c.name.includes('auth'));
    expect(sessionCookie).toBeDefined();
    
    // Refresh the page
    await page.reload();
    
    // Verify still on dashboard (not redirected to login)
    expect(page.url()).toContain('/dashboard');
    
    // Verify user is still logged in
    const userMenu = page.locator('[data-testid="user-menu"], .user-menu, .user-avatar');
    await expect(userMenu).toBeVisible();
  });

  test('should maintain session across different pages', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Navigate to different protected pages
    const protectedPages = [
      '/dashboard/cases',
      '/dashboard/documents',
      '/dashboard/profile',
      '/dashboard/settings'
    ];
    
    for (const pagePath of protectedPages) {
      await page.goto(pagePath);
      
      // Should not be redirected to login
      expect(page.url()).toContain(pagePath);
      
      // User menu should still be visible
      const userMenu = page.locator('[data-testid="user-menu"], .user-menu, .user-avatar');
      await expect(userMenu).toBeVisible();
    }
  });

  test('should handle session timeout gracefully', async ({ page, context }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Simulate session expiry by clearing cookies
    await context.clearCookies();
    
    // Try to navigate to protected page
    await page.goto('/dashboard/cases');
    
    // Should be redirected to login
    await page.waitForURL('/login**');
    
    // Check for session expired message
    const expiredMessage = page.locator('[data-testid="session-expired"], .alert-warning');
    if (await expiredMessage.isVisible()) {
      const message = await expiredMessage.textContent();
      expect(message).toMatch(/session|expired|login again/i);
    }
  });

  test('should logout successfully', async ({ page, context }) => {
    // Login first
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Find and click logout button
    const userMenu = page.locator('[data-testid="user-menu"], .user-menu, .user-avatar');
    await userMenu.click();
    
    const logoutButton = page.locator('[data-testid="logout-button"], button:has-text("Logout"), a:has-text("Logout")');
    await logoutButton.click();
    
    // Wait for redirect to home or login
    await page.waitForURL(/^\/$|\/login/);
    
    // Verify cookies are cleared
    const cookies = await context.cookies();
    const sessionCookie = cookies.find(c: unknown => c.name.includes('session') || c.name.includes('auth'));
    expect(sessionCookie).toBeUndefined();
    
    // Try to access protected page
    await page.goto('/dashboard');
    
    // Should be redirected to login
    await page.waitForURL('/login**');
  });

  test('should handle concurrent sessions', async ({ browser }) => {
    // Create two browser contexts (simulating different devices)
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();
    
    try {
      // Login on first device
      await page1.goto('/login');
      await page1.fill('input[name="email"]', testEmail);
      await page1.fill('input[name="password"]', testPassword);
      await page1.click('button[type="submit"]');
      await page1.waitForURL('/dashboard/**');
      
      // Login on second device
      await page2.goto('/login');
      await page2.fill('input[name="email"]', testEmail);
      await page2.fill('input[name="password"]', testPassword);
      await page2.click('button[type="submit"]');
      await page2.waitForURL('/dashboard/**');
      
      // Both sessions should be active
      await page1.reload();
      expect(page1.url()).toContain('/dashboard');
      
      await page2.reload();
      expect(page2.url()).toContain('/dashboard');
      
      // Check if system tracks active sessions
      await page1.goto('/dashboard/settings/sessions');
      const sessionList = page1.locator('[data-testid="active-sessions"]');
      if (await sessionList.isVisible()) {
        const sessionCount = await sessionList.locator('.session-item').count();
        expect(sessionCount).toBeGreaterThanOrEqual(2);
      }
    } finally {
      await context1.close();
      await context2.close();
    }
  });

  test('should remember user preferences', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Navigate to settings
    await page.goto('/dashboard/settings');
    
    // Change theme preference
    const themeToggle = page.locator('[data-testid="theme-toggle"], .theme-toggle');
    if (await themeToggle.isVisible()) {
      await themeToggle.click();
      
      // Wait for theme to apply
      await page.waitForTimeout(500);
      
      // Check if theme changed
      const isDarkMode = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') || 
               document.body.classList.contains('dark-theme');
      });
      
      // Refresh page
      await page.reload();
      
      // Theme should persist
      const isDarkModeAfterRefresh = await page.evaluate(() => {
        return document.documentElement.classList.contains('dark') || 
               document.body.classList.contains('dark-theme');
      });
      
      expect(isDarkModeAfterRefresh).toBe(isDarkMode);
    }
  });

  test('should handle password reset flow', async ({ page }) => {
    // Navigate to forgot password
    await page.goto('/forgot-password');
    
    // Enter email
    await page.fill('input[name="email"]', testEmail);
    await page.click('button[type="submit"]');
    
    // Check for success message
    await page.waitForSelector('[data-testid="reset-email-sent"], .success-message');
    
    // In a real test, you would:
    // 1. Check email inbox for reset link
    // 2. Extract reset token
    // 3. Navigate to reset page with token
    // 4. Set new password
    
    // For now, just verify the UI flow works
    const successMessage = await page.locator('[data-testid="reset-email-sent"], .success-message').textContent();
    expect(successMessage).toMatch(/email.*sent|check.*inbox/i);
  });

  test('should enforce authentication on protected routes', async ({ page }) => {
    // Try to access protected routes without login
    const protectedRoutes = [
      '/dashboard',
      '/dashboard/cases',
      '/dashboard/documents',
      '/dashboard/ai-assistant',
      '/api/cases',
      '/api/documents'
    ];
    
    for (const route of protectedRoutes) {
      if (route.startsWith('/api')) {
        // Test API routes
        const response = await page.request.get(route);
        expect([401, 403]).toContain(response.status());
      } else {
        // Test page routes
        await page.goto(route);
        await page.waitForURL('/login**');
        expect(page.url()).toContain('/login');
      }
    }
  });
});