import { test, expect } from '@playwright/test';

test.describe('Visual Regression Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set consistent viewport for visual tests
    await page.setViewportSize({ width: 1440, height: 900 });
    
    // Navigate to the main interface
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Wait for any animations to complete
    await page.waitForTimeout(1000);
  });

  test('YoRHa Command Center - Default State', async ({ page }) => {
    // Wait for the command center to fully load
    await page.waitForSelector('.yorha-command-center', { timeout: 10000 });
    
    // Hide dynamic elements that change frequently
    await page.addStyleTag({
      content: `
        .status-indicator .animate-pulse { animation: none !important; }
        .metric-card .transition-all { transition: none !important; }
        [data-testid="timestamp"] { visibility: hidden; }
        .text-xs.opacity-75 { visibility: hidden; }
      `
    });
    
    // Take screenshot of the full command center
    await expect(page.locator('.yorha-command-center')).toHaveScreenshot('yorha-command-center-default.png', {
      clip: { x: 0, y: 0, width: 1440, height: 900 },
      threshold: 0.2,
      maxDiffPixels: 1000
    });
  });

  test('YoRHa Command Center - Modal Open State', async ({ page }) => {
    await page.waitForSelector('.yorha-command-center');
    
    // Open the case creation modal
    await page.click('text=Create New Case');
    await page.waitForSelector('.modal-backdrop', { timeout: 5000 });
    
    // Wait for modal animations
    await page.waitForTimeout(500);
    
    // Hide dynamic timestamps
    await page.addStyleTag({
      content: `
        [data-testid="timestamp"] { visibility: hidden; }
        .animate-pulse { animation: none !important; }
      `
    });
    
    // Take screenshot with modal open
    await expect(page).toHaveScreenshot('yorha-modal-open.png', {
      fullPage: false,
      threshold: 0.2,
      maxDiffPixels: 1500
    });
  });

  test('System Health Panel - Different States', async ({ page }) => {
    const healthPanel = page.locator('.system-health');
    await expect(healthPanel).toBeVisible();
    
    // Disable animations for consistent screenshots
    await page.addStyleTag({
      content: `
        .progress-bar { animation: none !important; transition: none !important; }
        .animate-pulse { animation: none !important; }
      `
    });
    
    // Take screenshot of health panel
    await expect(healthPanel).toHaveScreenshot('system-health-panel.png', {
      threshold: 0.15
    });
  });

  test('Metric Cards Grid Layout', async ({ page }) => {
    const metricsGrid = page.locator('.metric-card').first().locator('xpath=ancestor::div[1]');
    
    // Disable counter animations
    await page.addStyleTag({
      content: `
        .metric-card .text-2xl.font-bold { animation: none !important; }
        .metric-card { transition: none !important; }
      `
    });
    
    await expect(metricsGrid).toHaveScreenshot('metrics-grid-layout.png', {
      threshold: 0.1
    });
  });

  test('Quick Actions Grid', async ({ page }) => {
    const actionsGrid = page.locator('.actions-grid');
    await expect(actionsGrid).toBeVisible();
    
    // Disable hover effects for consistent screenshots
    await page.addStyleTag({
      content: `
        .action-card { transition: none !important; transform: none !important; }
        .action-card:hover { transform: none !important; }
      `
    });
    
    await expect(actionsGrid).toHaveScreenshot('quick-actions-grid.png', {
      threshold: 0.1
    });
  });

  test('Recent Activity Section', async ({ page }) => {
    const activitySection = page.locator('.recent-activity');
    await expect(activitySection).toBeVisible();
    
    // Hide timestamps for consistency
    await page.addStyleTag({
      content: `
        .activity-item .text-xs.opacity-75 { visibility: hidden; }
      `
    });
    
    await expect(activitySection).toHaveScreenshot('recent-activity-section.png', {
      threshold: 0.15
    });
  });

  test('Mobile Responsive Layout - Phone', async ({ page }) => {
    // Switch to mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    
    // Disable animations
    await page.addStyleTag({
      content: `
        * { animation: none !important; transition: none !important; }
        .animate-pulse { animation: none !important; }
        [data-testid="timestamp"] { visibility: hidden; }
      `
    });
    
    await expect(page).toHaveScreenshot('yorha-mobile-phone.png', {
      fullPage: true,
      threshold: 0.2,
      maxDiffPixels: 2000
    });
  });

  test('Mobile Responsive Layout - Tablet', async ({ page }) => {
    // Switch to tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(1000);
    
    // Disable animations
    await page.addStyleTag({
      content: `
        * { animation: none !important; transition: none !important; }
        .animate-pulse { animation: none !important; }
        [data-testid="timestamp"] { visibility: hidden; }
      `
    });
    
    await expect(page).toHaveScreenshot('yorha-tablet.png', {
      fullPage: true,
      threshold: 0.2,
      maxDiffPixels: 2000
    });
  });

  test('Dark Mode Theme Consistency', async ({ page }) => {
    // Check if dark mode toggle exists, otherwise manually apply dark theme
    const darkModeExists = await page.locator('[data-testid="dark-mode-toggle"]').count() > 0;
    
    if (darkModeExists) {
      await page.click('[data-testid="dark-mode-toggle"]');
    } else {
      // Apply dark theme manually
      await page.addStyleTag({
        content: `
          body { background-color: #1a1a1a; color: #ffffff; }
          .yorha-command-center { background-color: #2d2d2d; }
          .metric-card { background-color: #3d3d3d; color: #ffffff; }
          .system-health { background-color: #3d3d3d; }
          .action-card { background-color: #404040; }
        `
      });
    }
    
    await page.waitForTimeout(500);
    
    // Disable animations
    await page.addStyleTag({
      content: `
        * { animation: none !important; transition: none !important; }
        [data-testid="timestamp"] { visibility: hidden; }
      `
    });
    
    await expect(page).toHaveScreenshot('yorha-dark-theme.png', {
      fullPage: false,
      clip: { x: 0, y: 0, width: 1440, height: 900 },
      threshold: 0.3,
      maxDiffPixels: 2000
    });
  });

  test('File Upload Component - Different States', async ({ page }) => {
    // Navigate to a page with file upload if it exists
    const hasFileUpload = await page.locator('[data-testid="file-upload"]').count() > 0;
    
    if (!hasFileUpload) {
      // Navigate to demo or AI assistant page that might have file upload
      await page.goto('/ai-assistant-demo');
      await page.waitForLoadState('networkidle');
    }
    
    const fileUploadComponent = page.locator('[data-testid="file-upload"]');
    
    if (await fileUploadComponent.count() > 0) {
      // Disable animations
      await page.addStyleTag({
        content: `
          .file-upload { transition: none !important; }
          .drag-drop-zone { transition: none !important; }
          .progress-bar { animation: none !important; }
        `
      });
      
      await expect(fileUploadComponent).toHaveScreenshot('file-upload-default.png', {
        threshold: 0.1
      });
      
      // Test hover state
      await fileUploadComponent.hover();
      await page.waitForTimeout(100);
      
      await expect(fileUploadComponent).toHaveScreenshot('file-upload-hover.png', {
        threshold: 0.15
      });
    }
  });

  test('Cross-Browser Visual Consistency', async ({ page, browserName }) => {
    // Add browser-specific selectors and styling fixes if needed
    await page.addStyleTag({
      content: `
        /* Normalize font rendering across browsers */
        * { 
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        
        /* Disable animations for consistent screenshots */
        *, *::before, *::after {
          animation: none !important;
          transition: none !important;
        }
        
        [data-testid="timestamp"] { visibility: hidden; }
      `
    });
    
    await page.waitForTimeout(1000);
    
    await expect(page.locator('.yorha-command-center')).toHaveScreenshot(`yorha-command-center-${browserName}.png`, {
      threshold: 0.25,
      maxDiffPixels: 2000
    });
  });

  test('Component Loading States', async ({ page }) => {
    // Simulate slow loading by blocking network requests temporarily
    await page.route('**/api/**', route => {
      // Delay API responses to capture loading states
      setTimeout(() => {
        route.fulfill({ status: 200, body: '{}' });
      }, 2000);
    });
    
    await page.reload();
    
    // Take screenshot during loading state (should show loading indicators)
    await page.waitForTimeout(500);
    
    await expect(page).toHaveScreenshot('loading-state.png', {
      fullPage: false,
      clip: { x: 0, y: 0, width: 1440, height: 900 },
      threshold: 0.2
    });
  });

  test('Error State Display', async ({ page }) => {
    // Mock API to return errors
    await page.route('**/api/**', route => {
      route.fulfill({ 
        status: 500, 
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Service temporarily unavailable' })
      });
    });
    
    await page.reload();
    await page.waitForTimeout(2000); // Wait for error states to appear
    
    // Look for error indicators or messages
    const errorElements = await page.locator('.error, .alert-error, .text-red-500, [class*="error"]').count();
    
    if (errorElements > 0) {
      await expect(page).toHaveScreenshot('error-state.png', {
        fullPage: false,
        clip: { x: 0, y: 0, width: 1440, height: 900 },
        threshold: 0.2
      });
    }
  });

  // Cleanup after each test
  test.afterEach(async ({ page }) => {
    // Clear any route mocks
    await page.unrouteAll();
  });
});