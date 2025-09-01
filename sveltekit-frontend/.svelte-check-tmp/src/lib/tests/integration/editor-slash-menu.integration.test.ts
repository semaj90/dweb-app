import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { chromium, Browser, Page } from '@playwright/test';

// Integration test: spin up a lightweight dev server is assumed externally; this test hits running app.
// If no server running, it will fail fast. User can start via `npm run dev` before running.

const BASE = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

let browser: Browser; let page: Page;

async function ensureEditorReady(p: Page){
  await p.waitForSelector('.tiptap-editor-wrapper', { timeout: 10000 });
}

async function openSlashMenu(p: Page){
  await p.keyboard.type('/');
  await p.waitForSelector('.slash-menu', { timeout: 3000 });
}

describe('Editor Slash Menu Integration', () => {
  beforeAll(async () => {
    browser = await chromium.launch();
    page = await browser.newPage();
    // Navigate to a page expected to contain the editor. Adjust path if different.
    await page.goto(BASE + '/dev/editor' , { waitUntil: 'domcontentloaded' }).catch(async () => {
      // fallback: if dedicated dev route not present, try root and hope editor is rendered dynamically.
      await page.goto(BASE + '/', { waitUntil: 'domcontentloaded' });
    });
    // Best-effort: wait for any potential lazy mount
    await ensureEditorReady(page).catch(()=>{});
  });

  afterAll(async () => {
    await browser?.close();
  });

  it('opens slash menu and navigates commands', async () => {
    await ensureEditorReady(page);
    await openSlashMenu(page);
    const items = await page.$$('.slash-menu li');
    expect(items.length).toBeGreaterThan(0);
    // Press ArrowDown twice
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('ArrowDown');
    // Ensure a second item is highlighted (class text-blue-700)
    const active = await page.$('.slash-menu li.text-blue-700');
    expect(active).not.toBeNull();
  });

  it('filters commands when typing query', async () => {
    // Re-open fresh
    await page.keyboard.press('Escape');
    await openSlashMenu(page);
    await page.keyboard.type('mer');
    // Expect only mermaid-related or matching entries
    const texts = await page.$$eval('.slash-menu li', els => els.map(e => e.textContent || ''));
    expect(texts.some(t => /Mermaid/i.test(t))).toBe(true);
    // Should not list unrelated if filtering reduces set significantly
    // (Heuristic: at least one heading maybe still there if partial includes)
  });

  it('executes a command (Mermaid Diagram) and closes menu', async () => {
    await page.keyboard.press('Escape');
    await openSlashMenu(page);
    await page.keyboard.type('mer');
    await page.keyboard.press('Enter');
    // Menu should close
    const visible = await page.$('.slash-menu');
    expect(visible).toBeNull();
    // Mermaid fenced block inserted (basic pattern check)
    const html = await page.$eval('.tiptap-editor-wrapper', el => el.innerHTML);
    expect(html.includes('```mermaid')).toBe(true);
  });
});
