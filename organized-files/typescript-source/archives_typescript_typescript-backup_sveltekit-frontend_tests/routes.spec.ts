import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const ROUTES_FILE = path.resolve(process.cwd(), 'routes.txt');
const routes = fs.existsSync(ROUTES_FILE)
  ? fs.readFileSync(ROUTES_FILE, 'utf8').split('\n').map(r=>r.trim()).filter(Boolean)
  : ['/'];

for (const route of routes) {
  test(`Route ${route} should load without SSR/hydration errors`, async ({ page }) => {
    page.on('pageerror', (err) => {
      throw new Error(`Console error on ${route}: ${err?.message ?? String(err)}`);
    });

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        throw new Error(`Console error on ${route}: ${msg.text()}`);
      }
    });

    const res = await page.goto(`http://localhost:5173${route}`, { waitUntil: 'networkidle' });
    expect(res?.ok()).toBeTruthy();
    await expect(page.locator('body')).toBeVisible();
  });
}
