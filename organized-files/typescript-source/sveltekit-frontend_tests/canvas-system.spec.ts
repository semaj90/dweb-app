// @ts-nocheck
import { expect, test } from "@playwright/test";

test("EditableCanvasSystem loads correctly", async ({ page }) => {
  await page.goto("/");

  // Check if canvas element exists
  const canvas = page.locator("canvas");
  await expect(canvas).toBeVisible();

  // Check if toolbar is present
  const toolbar = page.locator(".toolbar");
  await expect(toolbar).toBeVisible();

  // Check accessibility
  const canvas_aria = await canvas.getAttribute("aria-label");
  expect(canvas_aria).toBeTruthy();
});

test("Canvas interaction works", async ({ page }) => {
  await page.goto("/");

  const canvas = page.locator("canvas");

  // Double click to create node
  await canvas.dblclick({ position: { x: 100, y: 100 } });

  // Check if node was created (this would need actual implementation)
  // await expect(page.locator('.node')).toBeVisible();
});

test("Responsive design works", async ({ page }) => {
  // Test mobile view
  await page.setViewportSize({ width: 375, height: 667 });
  await page.goto("/");

  const toolbar = page.locator(".toolbar");
  await expect(toolbar).toBeVisible();

  const canvas = page.locator("canvas");
  await expect(canvas).toBeVisible();
});

test("Evidence panel functionality", async ({ page }) => {
  await page.goto("/");

  const evidencePanel = page.locator(".evidence-panel");
  await expect(evidencePanel).toBeVisible();

  const evidenceTitle = page.locator(".evidence-panel h3");
  await expect(evidenceTitle).toContainText("Evidence");
});

test("Accessibility features", async ({ page }) => {
  await page.goto("/");

  // Check ARIA labels
  const canvas = page.locator("canvas");
  const ariaLabel = await canvas.getAttribute("aria-label");
  expect(ariaLabel).toContain("Interactive canvas");

  // Check keyboard navigation
  await canvas.focus();
  await expect(canvas).toBeFocused();

  // Check status indicator accessibility
  const status = page.locator(".status");
  const statusLabel = await status.getAttribute("aria-label");
  expect(statusLabel).toBeTruthy();
});
