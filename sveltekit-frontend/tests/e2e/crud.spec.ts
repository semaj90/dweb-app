import { test, expect } from '@playwright/test';
import path from 'path';
import { queryDb } from './helpers/db';

test.describe('Full CRUD flow', () => {
  const testUser = { email: `e2e+${Date.now()}@example.test`, password: 'Password123!' };
  let caseId: string | null = null;

  test('register -> login -> create case -> upload evidence -> verify DB', async ({ page }) => {
    await page.goto('/');

    // Navigate to register (adjust selectors to your app)
    await page.click('text=Register');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button:has-text("Create account")');

    // If app redirects to login, handle it
    try {
      await page.waitForURL('**/login', { timeout: 5000 });
      await page.fill('input[name="email"]', testUser.email);
      await page.fill('input[name="password"]', testUser.password);
      await page.click('button:has-text("Sign in")');
    } catch (e) {
      // not redirected to login
    }

    // Wait for dashboard
    await page.waitForURL('**/user/dashboard', { timeout: 20000 });
    expect(page.url()).toContain('/user/dashboard');

    // Create a new case
    await page.click('text=New Case');
    await page.fill('input[name="title"]', 'E2E Test Case');
    await page.fill('textarea[name="description"]', 'This case was created by an automated test.');
    await page.click('button:has-text("Create")');

    // Wait for case list or case page
    await page.waitForSelector('text=E2E Test Case', { timeout: 10000 });
    const url = page.url();
    if (url.includes('/cases/')) caseId = url.split('/cases/').pop() || null;
    if (!caseId) {
      const el = await page.$('[data-case-id]');
      if (el) caseId = await el.getAttribute('data-case-id');
    }
    expect(caseId).not.toBeNull();

    // Upload evidence
    const filePath = path.resolve(__dirname, 'fixtures', 'evidence.txt');
    await page.setInputFiles('input[type="file"]', filePath);
    await page.click('button:has-text("Upload Evidence")');
    await page.waitForSelector('text=evidence.txt', { timeout: 10000 });

    // Verify DB
    const rows = await queryDb('SELECT id, title FROM cases WHERE id = $1', [caseId]);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0].title).toBe('E2E Test Case');
  });
});
