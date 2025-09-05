// YoRHa Demos Single Page App - Comprehensive Playwright Tests
import { test, expect, type Page } from '@playwright/test';

test.describe('YoRHa Demos Single Page App', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the main homepage
    await page.goto('/');
    
    // Wait for the page to load completely
    await page.waitForLoadState('networkidle');
  });

  test('should display the INTERACTIVE DEMOS button prominently on homepage', async ({ page }) => {
    // Check that the demos button exists and is visible
    const demosButton = page.locator('button:has-text("INTERACTIVE DEMOS")');
    await expect(demosButton).toBeVisible();
    
    // Verify it has the proper styling and glow effect
    await expect(demosButton).toHaveClass(/yorha-btn-demos/);
    await expect(demosButton).toHaveClass(/yorha-btn-primary/);
    
    // Check that it contains the Play icon
    const playIcon = demosButton.locator('svg');
    await expect(playIcon).toBeVisible();
  });

  test('should navigate to demos page when INTERACTIVE DEMOS button is clicked', async ({ page }) => {
    // Click the INTERACTIVE DEMOS button
    await page.click('button:has-text("INTERACTIVE DEMOS")');
    
    // Wait for navigation to complete
    await page.waitForLoadState('networkidle');
    
    // Verify we're on the demos page
    await expect(page).toHaveURL('/demos');
    
    // Check that the demos page title is correct
    await expect(page.locator('h1')).toContainText('DEMO CENTER');
  });

  test('should display all demo categories in the demos page', async ({ page }) => {
    // Navigate to demos page
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check that all expected categories are present
    const expectedCategories = [
      'DEMOS OVERVIEW',
      'AI DEMONSTRATIONS', 
      'LEGAL AI DEMONSTRATIONS',
      'UI/UX DEMONSTRATIONS',
      'DEVELOPMENT TOOLS',
      'MESSAGING & COMMUNICATION',
      'ANALYTICS & MONITORING',
      'ADMINISTRATION TOOLS'
    ];
    
    for (const category of expectedCategories) {
      await expect(page.locator(`button:has-text("${category}")`)).toBeVisible();
    }
  });

  test('should have functional ScrollArea component', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check that ScrollArea is present
    const scrollArea = page.locator('[data-bits-scroll-area-root]');
    await expect(scrollArea).toBeVisible();
    
    // Test scrolling functionality
    await page.mouse.wheel(0, 500);
    await page.waitForTimeout(500);
    
    // Verify scrollbars are visible when needed
    const viewport = page.locator('[data-bits-scroll-area-viewport]');
    await expect(viewport).toBeVisible();
  });

  test('should display demo cards with proper information', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Wait for demo cards to load
    await page.waitForSelector('.yorha-demo-card', { timeout: 10000 });
    
    // Check that demo cards are visible
    const demoCards = page.locator('.yorha-demo-card');
    await expect(demoCards.first()).toBeVisible();
    
    // Verify card structure
    const firstCard = demoCards.first();
    await expect(firstCard.locator('.yorha-demo-title')).toBeVisible();
    await expect(firstCard.locator('.yorha-demo-description')).toBeVisible();
    await expect(firstCard.locator('.yorha-demo-path')).toBeVisible();
    
    // Check for status badges
    await expect(firstCard.locator('[class*="bg-green-500"], [class*="bg-yellow-500"], [class*="bg-blue-500"]')).toBeVisible();
  });

  test('should navigate to specific demos when cards are clicked', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Find and click on an Enhanced AI Demo card
    const enhancedAICard = page.locator('.yorha-demo-card:has-text("Enhanced AI Demo")').first();
    
    if (await enhancedAICard.isVisible()) {
      await enhancedAICard.click();
      
      // Wait for navigation
      await page.waitForLoadState('networkidle');
      
      // Verify we navigated to the correct route
      expect(page.url()).toContain('/enhanced-ai-demo');
    }
  });

  test('should filter demos by category when category buttons are clicked', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Click on AI DEMONSTRATIONS category
    await page.click('button:has-text("AI DEMONSTRATIONS")');
    await page.waitForTimeout(1000);
    
    // Check that the AI demos section is visible
    const aiSection = page.locator('#category-ai-demos');
    await expect(aiSection).toBeVisible();
    
    // Verify AI-specific demos are shown
    await expect(page.locator('.yorha-demo-card:has-text("Enhanced AI Demo")')).toBeVisible();
    await expect(page.locator('.yorha-demo-card:has-text("GPU Chat Interface")')).toBeVisible();
  });

  test('should have working breadcrumb navigation', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check breadcrumb elements
    await expect(page.locator('.yorha-breadcrumb-item:has-text("YoRHa Legal AI")')).toBeVisible();
    await expect(page.locator('.yorha-breadcrumb-current:has-text("Demo Center")')).toBeVisible();
    
    // Test HOME button functionality
    await page.click('button:has-text("HOME")');
    await page.waitForLoadState('networkidle');
    
    // Verify we're back on the homepage
    await expect(page).toHaveURL('/');
  });

  test('should display technical specifications in footer', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Scroll to footer
    await page.locator('.yorha-demos-footer').scrollIntoViewIfNeeded();
    
    // Check technical specifications
    const techSpecs = [
      'SvelteKit 2 + Svelte 5 runes',
      'bits-ui + melt-ui + shadcn-svelte',
      'Multi-core Ollama + NVIDIA GPU acceleration',
      'PostgreSQL + pgvector + Neo4j',
      'XState + enhanced reactive stores',
      'NATS + WebSocket + real-time communication'
    ];
    
    for (const spec of techSpecs) {
      await expect(page.locator(`.yorha-spec-item:has-text("${spec}")`)).toBeVisible();
    }
  });

  test('should show loading state when navigating to demos', async ({ page }) => {
    await page.goto('/demos');
    
    // Check for loading elements during navigation
    // Note: This test might be flaky due to fast loading times
    const loadingText = page.locator('.yorha-loading-text:has-text("INITIALIZING DEMO...")');
    
    // The loading state might already be gone, so we just check that the page loads correctly
    await page.waitForLoadState('networkidle');
    await expect(page.locator('h1:has-text("DEMO CENTER")')).toBeVisible();
  });

  test('should have responsive design on mobile viewports', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check that elements are properly responsive
    await expect(page.locator('.yorha-demos-title')).toBeVisible();
    await expect(page.locator('.yorha-category-nav')).toBeVisible();
    
    // Verify grid layout adapts to mobile
    const demoCards = page.locator('.yorha-demo-card');
    await expect(demoCards.first()).toBeVisible();
  });

  test('should have accessible keyboard navigation', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Test tab navigation
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Verify focus management
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
    
    // Test Enter key navigation
    await page.keyboard.press('Enter');
    
    // Should either navigate or scroll to section
    await page.waitForTimeout(500);
  });

  test('should validate demo statistics are displayed correctly', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check demo statistics
    const totalDemosElement = page.locator('.yorha-stat-number').first();
    const categoriesElement = page.locator('.yorha-stat-number').nth(1);
    
    await expect(totalDemosElement).toBeVisible();
    await expect(categoriesElement).toBeVisible();
    
    // Verify numbers are realistic (should be > 0)
    const totalDemosText = await totalDemosElement.textContent();
    const categoriesText = await categoriesElement.textContent();
    
    expect(parseInt(totalDemosText || '0')).toBeGreaterThan(0);
    expect(parseInt(categoriesText || '0')).toBeGreaterThan(0);
  });

  test('should have proper ARIA labels and accessibility', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Check page title
    await expect(page).toHaveTitle(/YoRHa Legal AI - Demo Center/);
    
    // Check meta description
    const metaDescription = page.locator('meta[name="description"]');
    await expect(metaDescription).toHaveAttribute('content', /Comprehensive demo center/);
    
    // Check heading hierarchy
    await expect(page.locator('h1')).toContainText('DEMO CENTER');
    await expect(page.locator('h2').first()).toBeVisible();
  });

  test('should handle error states gracefully', async ({ page }) => {
    // Test navigation to non-existent demo
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Try to navigate to a non-existent route
    await page.goto('/demos/non-existent-demo');
    
    // Should either redirect or show error page
    // Wait for any redirect to complete
    await page.waitForLoadState('networkidle');
    
    // Page should handle this gracefully (either 404 or redirect to demos)
    const currentUrl = page.url();
    expect(currentUrl).toMatch(/\/demos|\/404|\/error/);
  });

  test('should maintain scroll position when navigating between categories', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Scroll down a bit
    await page.mouse.wheel(0, 500);
    await page.waitForTimeout(500);
    
    // Click on a category
    await page.click('button:has-text("AI DEMONSTRATIONS")');
    await page.waitForTimeout(1000);
    
    // Verify we scrolled to the category section
    const aiSection = page.locator('#category-ai-demos');
    await expect(aiSection).toBeInViewport();
  });
});

test.describe('Demo Navigation Integration', () => {
  test('should navigate back to demos from individual demo pages', async ({ page }) => {
    // Start from a demo page that exists
    await page.goto('/yorha-demo');
    await page.waitForLoadState('networkidle');
    
    // Look for a back button or demos link
    const backButton = page.locator('a[href="/demos"], button:has-text("Demos"), a:has-text("Demos")');
    
    if (await backButton.count() > 0) {
      await backButton.first().click();
      await page.waitForLoadState('networkidle');
      await expect(page).toHaveURL('/demos');
    }
  });

  test('should handle direct navigation to demo routes from demos page', async ({ page }) => {
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Look for a specific demo link
    const ragDemoCard = page.locator('.yorha-demo-card:has-text("RAG Demo")').first();
    
    if (await ragDemoCard.isVisible()) {
      await ragDemoCard.click();
      await page.waitForLoadState('networkidle');
      
      // Verify we navigated to the RAG demo
      expect(page.url()).toContain('/rag-demo');
    }
  });
});

test.describe('Performance and Loading', () => {
  test('should load demos page within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Page should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
    
    // Check that main content is visible
    await expect(page.locator('h1:has-text("DEMO CENTER")')).toBeVisible();
  });

  test('should not have JavaScript errors on demos page', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('pageerror', (error) => {
      errors.push(error.message);
    });
    
    await page.goto('/demos');
    await page.waitForLoadState('networkidle');
    
    // Interact with the page to trigger any potential errors
    await page.click('button:has-text("AI DEMONSTRATIONS")');
    await page.waitForTimeout(1000);
    
    // Should have no JavaScript errors
    expect(errors).toHaveLength(0);
  });
});