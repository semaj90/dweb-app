import { test, expect } from '@playwright/test';

test.describe('YoRHa Command Center', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the YoRHa Command Center page
    await page.goto('/');
    
    // Wait for the command center to load
    await page.waitForSelector('.yorha-command-center', { timeout: 10000 });
  });

  test('displays system overview cards', async ({ page }) => {
    // Check that all metric cards are visible
    await expect(page.locator('.metric-card')).toHaveCount(4);
    
    // Verify each metric card has expected content
    const metricCards = page.locator('.metric-card');
    
    await expect(metricCards.nth(0)).toContainText('Active Cases');
    await expect(metricCards.nth(1)).toContainText('Evidence');
    await expect(metricCards.nth(2)).toContainText('Persons');
    await expect(metricCards.nth(3)).toContainText('AI Queries');
    
    // Check that numbers are displayed
    for (let i = 0; i < 4; i++) {
      await expect(metricCards.nth(i).locator('.text-2xl.font-bold')).toBeVisible();
    }
  });

  test('shows system health panel', async ({ page }) => {
    const healthPanel = page.locator('.system-health');
    
    await expect(healthPanel).toBeVisible();
    await expect(healthPanel.locator('h3')).toContainText('System Health');
    
    // Check that all health metrics are present
    await expect(healthPanel.locator('.metric')).toHaveCount(4);
    
    // Verify metric labels
    await expect(healthPanel).toContainText('CPU Load');
    await expect(healthPanel).toContainText('GPU Usage');
    await expect(healthPanel).toContainText('Memory');
    await expect(healthPanel).toContainText('Network');
    
    // Check progress bars are visible
    await expect(healthPanel.locator('.progress-bar')).toHaveCount(3);
  });

  test('displays quick action buttons', async ({ page }) => {
    const actionsGrid = page.locator('.actions-grid');
    
    await expect(actionsGrid).toBeVisible();
    
    // Should have 7 quick action buttons
    const actionCards = actionsGrid.locator('.action-card');
    await expect(actionCards).toHaveCount(7);
    
    // Check specific actions are present
    await expect(page.locator('text=Create New Case')).toBeVisible();
    await expect(page.locator('text=Upload Evidence')).toBeVisible();
    await expect(page.locator('text=AI Analysis')).toBeVisible();
    await expect(page.locator('text=3D AI Assistant')).toBeVisible();
    await expect(page.locator('text=Global Search')).toBeVisible();
    await expect(page.locator('text=Generate Report')).toBeVisible();
    await expect(page.locator('text=Memory Graph')).toBeVisible();
  });

  test('opens case creation modal', async ({ page }) => {
    // Click the "Create New Case" button
    await page.click('text=Create New Case');
    
    // Wait for modal to appear
    await page.waitForSelector('.modal-backdrop', { timeout: 5000 });
    
    // Check modal is visible
    await expect(page.locator('.modal-backdrop')).toBeVisible();
    await expect(page.locator('.modal-content')).toBeVisible();
    
    // Should contain the case form
    await expect(page.locator('text=CREATE NEW CASE')).toBeVisible();
  });

  test('modal closes when clicking backdrop', async ({ page }) => {
    // Open modal
    await page.click('text=Create New Case');
    await page.waitForSelector('.modal-backdrop');
    
    // Click backdrop (outside modal content)
    await page.locator('.modal-backdrop').click({ position: { x: 50, y: 50 } });
    
    // Modal should close
    await expect(page.locator('.modal-backdrop')).not.toBeVisible();
  });

  test('displays recent activity list', async ({ page }) => {
    const activitySection = page.locator('.recent-activity');
    
    await expect(activitySection).toBeVisible();
    await expect(activitySection.locator('h2')).toContainText('Recent Activity');
    
    // Should have activity items
    const activityItems = activitySection.locator('.activity-item');
    await expect(activityItems).toHaveCountGreaterThan(0);
    
    // Each activity item should have an icon, action, and time
    const firstActivity = activityItems.first();
    await expect(firstActivity.locator('.text-lg')).toBeVisible(); // Icon
    await expect(firstActivity.locator('.font-medium')).toBeVisible(); // Action
    await expect(firstActivity.locator('.text-xs.opacity-75')).toBeVisible(); // Time
  });

  test('search panel is functional', async ({ page }) => {
    const searchPanel = page.locator('.search-panel');
    
    await expect(searchPanel).toBeVisible();
    
    // Should contain search component
    await expect(searchPanel.locator('[placeholder*="Search"]')).toBeVisible();
  });

  test('quick actions are interactive', async ({ page }) => {
    // Test hover effects on action cards
    const firstAction = page.locator('.action-card').first();
    
    await firstAction.hover();
    
    // Should have hover effects (scale transformation)
    await expect(firstAction).toHaveCSS('cursor', 'pointer');
  });

  test('system health indicators work', async ({ page }) => {
    const statusIndicator = page.locator('.status-indicator');
    
    await expect(statusIndicator).toBeVisible();
    
    // Should have status light and text
    await expect(statusIndicator.locator('.rounded-full')).toBeVisible();
    await expect(statusIndicator.locator('.uppercase')).toBeVisible();
  });

  test('responsive design works', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Command center should still be visible and functional
    await expect(page.locator('.yorha-command-center')).toBeVisible();
    
    // Actions grid should adjust to mobile layout
    await expect(page.locator('.actions-grid')).toBeVisible();
    
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    
    await expect(page.locator('.yorha-command-center')).toBeVisible();
  });

  test('animation effects are present', async ({ page }) => {
    // Check for CSS animations/transitions
    const metricCard = page.locator('.metric-card').first();
    
    await expect(metricCard).toHaveCSS('transition', /.*duration.*/);
    
    // Status indicator should have pulse animation
    const statusLight = page.locator('.status-indicator .rounded-full');
    await expect(statusLight).toHaveClass(/.*animate-pulse.*/);
  });

  test('accessibility features', async ({ page }) => {
    // Check for proper heading hierarchy
    await expect(page.locator('h1')).toContainText('YoRHa Command Center');
    await expect(page.locator('h2')).toHaveCount(3); // Quick Actions, Legal AI Search, Recent Activity
    await expect(page.locator('h3')).toContainText('System Health');
    
    // Check for semantic elements
    await expect(page.locator('main, section, nav, article')).toHaveCountGreaterThan(0);
    
    // Buttons should be keyboard accessible
    const firstActionButton = page.locator('.action-card').first();
    await firstActionButton.focus();
    await expect(firstActionButton).toBeFocused();
  });
});