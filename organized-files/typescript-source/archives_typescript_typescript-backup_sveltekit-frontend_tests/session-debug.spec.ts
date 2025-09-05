import { test, expect } from '@playwright/test';

test.describe('Session Debug Panel Testing', () => {
  const testEmail = 'demo@legalai.gov';
  const testPassword = 'demo123';

  test('should show "Not logged in" status on homepage when not authenticated', async ({ page }) => {
    // Navigate to homepage
    await page.goto('/');
    
    // Check for session debug panel
    await expect(page.locator('.session-debug-panel')).toBeVisible();
    
    // Verify debug title
    await expect(page.locator('.debug-title')).toHaveText('🔐 SESSION DEBUG PANEL');
    
    // Check for not authenticated status
    await expect(page.locator('.status-message')).toHaveText('❌ Not logged in');
    
    // Verify login and register links are present
    await expect(page.locator('.action-btn.login')).toBeVisible();
    await expect(page.locator('.action-btn.login')).toHaveText('🔑 Login');
    
    // Check that session details are not shown
    await expect(page.locator('.session-details')).not.toBeVisible();
  });

  test('should display full session information after login', async ({ page }) => {
    // Navigate to homepage first to verify not logged in
    await page.goto('/');
    await expect(page.locator('.status-message')).toHaveText('❌ Not logged in');
    
    // Navigate to login page
    await page.goto('/auth/login');
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Fill login form with demo credentials
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    
    // Submit login form
    await page.click('button[type="submit"]');
    
    // Wait for redirect to dashboard
    await page.waitForURL('/dashboard', { timeout: 10000 });
    
    // Go back to homepage to check session debug
    await page.goto('/');
    
    // Check for authenticated status
    await expect(page.locator('.status-message')).toContainText('✅ Logged in as: demo@legalai.gov');
    
    // Verify session details are visible
    await expect(page.locator('.session-details')).toBeVisible();
    
    // Check User ID display
    await expect(page.locator('.detail-item').first()).toContainText('User ID:');
    await expect(page.locator('.value').first()).toContainText('demo-user-001');
    
    // Check Session ID display
    await expect(page.locator('.detail-item').nth(1)).toContainText('Session ID:');
    await expect(page.locator('.session-id')).toBeVisible();
    
    // Verify logout button is present
    await expect(page.locator('.action-btn.logout')).toBeVisible();
    await expect(page.locator('.action-btn.logout')).toHaveText('🔓 Logout');
    
    // Verify dashboard link is present
    await expect(page.locator('.action-btn.dashboard')).toBeVisible();
    await expect(page.locator('.action-btn.dashboard')).toHaveText('📊 Go to Dashboard');
  });

  test('should show consistent session info on dashboard page', async ({ page }) => {
    // Login first
    await page.goto('/auth/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // Check dashboard session debug panel
    await expect(page.locator('.session-debug-panel')).toBeVisible();
    await expect(page.locator('.debug-title')).toHaveText('🔐 DASHBOARD SESSION DEBUG');
    
    // Verify authenticated status
    await expect(page.locator('.status-message')).toContainText('✅ Logged in as: demo@legalai.gov');
    
    // Check session details consistency
    await expect(page.locator('.value').first()).toContainText('demo-user-001');
    await expect(page.locator('.session-id')).toBeVisible();
    
    // Verify navigation links
    await expect(page.locator('.action-btn.dashboard')).toHaveText('🏠 Back to Homepage');
    await expect(page.locator('.action-btn.logout')).toHaveText('🔓 Logout');
  });

  test('should handle logout functionality correctly', async ({ page, context }) => {
    // Login first
    await page.goto('/auth/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // Go to homepage to test logout
    await page.goto('/');
    
    // Verify logged in state
    await expect(page.locator('.status-message')).toContainText('✅ Logged in as: demo@legalai.gov');
    
    // Get cookies before logout
    const cookiesBefore = await context.cookies();
    const sessionCookieBefore = cookiesBefore.find(c => c.name === 'auth-session');
    expect(sessionCookieBefore).toBeDefined();
    
    // Click logout button
    await page.click('.action-btn.logout');
    
    // Wait for redirect back to homepage
    await page.waitForURL('/', { timeout: 5000 });
    
    // Verify logged out state
    await expect(page.locator('.status-message')).toHaveText('❌ Not logged in');
    
    // Verify session details are hidden
    await expect(page.locator('.session-details')).not.toBeVisible();
    
    // Verify login links are shown
    await expect(page.locator('.action-btn.login')).toBeVisible();
    
    // Check that auth cookie is cleared
    const cookiesAfter = await context.cookies();
    const sessionCookieAfter = cookiesAfter.find(c => c.name === 'auth-session');
    expect(sessionCookieAfter).toBeUndefined();
  });

  test('should maintain session across page navigation', async ({ page }) => {
    // Login
    await page.goto('/auth/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // Navigate between pages and verify session persistence
    const pages = ['/', '/dashboard'];
    
    for (const pagePath of pages) {
      await page.goto(pagePath);
      
      // Verify session debug panel shows authenticated state
      await expect(page.locator('.session-debug-panel')).toBeVisible();
      await expect(page.locator('.status-message')).toContainText('✅ Logged in as: demo@legalai.gov');
      await expect(page.locator('.session-details')).toBeVisible();
      await expect(page.locator('.value').first()).toContainText('demo-user-001');
    }
  });

  test('should show proper styling for session debug panel', async ({ page }) => {
    await page.goto('/');
    
    // Check panel styling
    const panel = page.locator('.session-debug-panel');
    await expect(panel).toBeVisible();
    
    // Verify it has the expected styling attributes
    await expect(panel).toHaveCSS('border-color', 'rgb(255, 215, 0)'); // #ffd700
    await expect(panel).toHaveCSS('background-color', 'rgb(26, 26, 26)'); // #1a1a1a
    
    // Check debug title styling
    const title = page.locator('.debug-title');
    await expect(title).toHaveCSS('color', 'rgb(255, 215, 0)'); // #ffd700
    await expect(title).toHaveCSS('text-align', 'center');
  });

  test('should handle session from dashboard logout', async ({ page }) => {
    // Login and go to dashboard
    await page.goto('/auth/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // Verify dashboard session debug
    await expect(page.locator('.status-message')).toContainText('✅ Logged in as: demo@legalai.gov');
    
    // Logout from dashboard
    await page.click('.action-btn.logout');
    
    // Should redirect to homepage
    await page.waitForURL('/');
    
    // Verify logged out state on homepage
    await expect(page.locator('.status-message')).toHaveText('❌ Not logged in');
  });

  test('should show session ID as valid UUID format', async ({ page }) => {
    // Login first
    await page.goto('/auth/login');
    await page.fill('input[name="email"]', testEmail);
    await page.fill('input[name="password"]', testPassword);
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');
    
    // Go to homepage to check session ID
    await page.goto('/');
    
    // Get session ID value
    const sessionId = await page.locator('.session-id').textContent();
    
    // Verify it looks like a UUID (basic format check)
    expect(sessionId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
    
    // Verify it's displayed in both places consistently
    await page.goto('/dashboard');
    const dashboardSessionId = await page.locator('.session-id').textContent();
    expect(sessionId).toBe(dashboardSessionId);
  });
});