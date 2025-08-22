
import { test, expect } from '@playwright/test';

test.describe('Global Store and State Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login to access protected features
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
  });

  test('should maintain global user state across components', async ({ page }) => {
    // Check initial state in header
    const headerUserName = await page.locator('[data-testid="header-user-name"]').textContent();
    
    // Navigate to profile
    await page.goto('/dashboard/profile');
    
    // Update user name
    const newName = `Updated User ${Date.now()}`;
    await page.fill('input[name="name"]', newName);
    await page.click('button:has-text("Save")');
    
    // Wait for update
    await page.waitForSelector('[data-testid="success-toast"], .toast-success');
    
    // Check if header updated without refresh
    const updatedHeaderName = await page.locator('[data-testid="header-user-name"]').textContent();
    expect(updatedHeaderName).toContain(newName);
    
    // Navigate to different page
    await page.goto('/dashboard/cases');
    
    // Name should still be updated
    const headerNameOnNewPage = await page.locator('[data-testid="header-user-name"]').textContent();
    expect(headerNameOnNewPage).toContain(newName);
  });

  test('should sync notification state globally', async ({ page }) => {
    // Check initial notification count
    const notificationBadge = page.locator('[data-testid="notification-badge"]');
    const initialCount = await notificationBadge.textContent() || '0';
    
    // Trigger a notification (simulate via API)
    await page.request.post('/api/notifications/test', {
      data: {
        type: 'info',
        message: 'Test notification'
      }
    });
    
    // Wait for notification count to update
    await page.waitForFunction(
      (selector, oldCount) => {
        const el = document.querySelector(selector);
        return el && el.textContent !== oldCount;
      },
      '[data-testid="notification-badge"]',
      initialCount
    );
    
    // Verify count increased
    const newCount = await notificationBadge.textContent();
    expect(parseInt(newCount || '0')).toBeGreaterThan(parseInt(initialCount));
    
    // Open notification panel
    await page.click('[data-testid="notification-icon"]');
    
    // Mark as read
    await page.click('[data-testid="mark-all-read"]');
    
    // Count should reset
    await page.waitForFunction(
      (selector: string) => {
        const el = document.querySelector(selector);
        return !el || el.textContent === '0' || el.style.display === 'none';
      },
      '[data-testid="notification-badge"]'
    );
  });

  test('should maintain case selection state', async ({ page }) => {
    // Navigate to cases
    await page.goto('/dashboard/cases');
    
    // Select a case
    const firstCase = page.locator('[data-testid="case-item"]').first();
    const caseId = await firstCase.getAttribute('data-case-id');
    await firstCase.click();
    
    // Verify case is selected in global state
    await page.waitForSelector('[data-testid="selected-case-indicator"]');
    
    // Navigate to documents
    await page.goto('/dashboard/documents');
    
    // Selected case should be maintained
    const selectedCaseInfo = page.locator('[data-testid="current-case-info"]');
    await expect(selectedCaseInfo).toBeVisible();
    const displayedCaseId = await selectedCaseInfo.getAttribute('data-case-id');
    expect(displayedCaseId).toBe(caseId);
    
    // Documents should be filtered by selected case
    const documents = page.locator('[data-testid="document-item"]');
    const docCount = await documents.count();
    if (docCount > 0) {
      const firstDoc = documents.first();
      const docCaseId = await firstDoc.getAttribute('data-case-id');
      expect(docCaseId).toBe(caseId);
    }
  });

  test('should persist UI preferences in global store', async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/dashboard');
    
    // Toggle sidebar
    const sidebarToggle = page.locator('[data-testid="sidebar-toggle"]');
    await sidebarToggle.click();
    
    // Check sidebar state
    const sidebar = page.locator('[data-testid="sidebar"]');
    const isCollapsed = await sidebar.getAttribute('data-collapsed');
    
    // Navigate to different page
    await page.goto('/dashboard/cases');
    
    // Sidebar state should persist
    const sidebarOnNewPage = page.locator('[data-testid="sidebar"]');
    const isStillCollapsed = await sidebarOnNewPage.getAttribute('data-collapsed');
    expect(isStillCollapsed).toBe(isCollapsed);
    
    // Change view mode
    const viewModeToggle = page.locator('[data-testid="view-mode-toggle"]');
    if (await viewModeToggle.isVisible()) {
      await viewModeToggle.click();
      await page.click('[data-testid="view-mode-grid"]');
      
      // Navigate away and back
      await page.goto('/dashboard/documents');
      await page.goto('/dashboard/cases');
      
      // View mode should persist
      const currentViewMode = await page.locator('[data-testid="current-view"]').getAttribute('data-view');
      expect(currentViewMode).toBe('grid');
    }
  });

  test('should handle optimistic updates in global state', async ({ page }) => {
    // Navigate to cases
    await page.goto('/dashboard/cases');
    
    // Create a new case with optimistic update
    await page.click('[data-testid="create-case-button"]');
    
    const caseTitle = `Test Case ${Date.now()}`;
    await page.fill('input[name="title"]', caseTitle);
    await page.fill('textarea[name="description"]', 'Test description');
    
    // Submit form
    await page.click('button[type="submit"]');
    
    // Case should appear immediately (optimistic update)
    const newCase = page.locator(`[data-testid="case-item"]:has-text("${caseTitle}")`);
    await expect(newCase).toBeVisible({ timeout: 1000 });
    
    // Check for pending state indicator
    const pendingIndicator = newCase.locator('[data-testid="pending-indicator"]');
    if (await pendingIndicator.isVisible()) {
      // Wait for it to disappear (confirming server sync)
      await expect(pendingIndicator).toBeHidden({ timeout: 5000 });
    }
    
    // Verify case is actually saved
    await page.reload();
    await expect(newCase).toBeVisible();
  });

  test('should synchronize filter state across views', async ({ page }) => {
    // Set filters on cases page
    await page.goto('/dashboard/cases');
    
    // Apply status filter
    await page.click('[data-testid="filter-button"]');
    await page.click('[data-testid="status-filter-active"]');
    await page.click('[data-testid="apply-filters"]');
    
    // Navigate to different view
    await page.goto('/dashboard/analytics');
    
    // Check if filters are reflected
    const activeFilterBadge = page.locator('[data-testid="active-filters-badge"]');
    if (await activeFilterBadge.isVisible()) {
      const filterCount = await activeFilterBadge.textContent();
      expect(parseInt(filterCount || '0')).toBeGreaterThan(0);
    }
    
    // Go back to cases
    await page.goto('/dashboard/cases');
    
    // Filters should still be applied
    const statusFilter = page.locator('[data-testid="status-filter-active"]');
    const isChecked = await statusFilter.isChecked();
    expect(isChecked).toBe(true);
  });

  test('should handle real-time updates in global state', async ({ page, context }) => {
    // Open two tabs
    const page2 = await context.newPage();
    
    // Login on second tab
    await page2.goto('/dashboard/cases');
    
    // Create a case in first tab
    const caseTitle = `Real-time Test ${Date.now()}`;
    await page.click('[data-testid="create-case-button"]');
    await page.fill('input[name="title"]', caseTitle);
    await page.click('button[type="submit"]');
    
    // Check if it appears in second tab (real-time sync)
    const newCaseInTab2 = page2.locator(`[data-testid="case-item"]:has-text("${caseTitle}")`);
    await expect(newCaseInTab2).toBeVisible({ timeout: 10000 });
    
    await page2.close();
  });

  test('should persist global search state', async ({ page }) => {
    // Perform a global search
    const searchInput = page.locator('[data-testid="global-search"]');
    await searchInput.fill('contract law');
    await searchInput.press('Enter');
    
    // Wait for search results
    await page.waitForSelector('[data-testid="search-results"]');
    
    // Navigate to a result
    const firstResult = page.locator('[data-testid="search-result-item"]').first();
    await firstResult.click();
    
    // Go back using browser back button
    await page.goBack();
    
    // Search query should be preserved
    const currentSearchValue = await searchInput.inputValue();
    expect(currentSearchValue).toBe('contract law');
    
    // Results should still be visible
    await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
  });

  test('should handle error states in global store', async ({ page }) => {
    // Trigger an error by making an invalid request
    await page.goto('/dashboard/cases');
    
    // Try to create a case with invalid data
    await page.click('[data-testid="create-case-button"]');
    await page.fill('input[name="title"]', ''); // Empty title
    await page.click('button[type="submit"]');
    
    // Error should be displayed globally
    const errorToast = page.locator('[data-testid="error-toast"], .toast-error');
    await expect(errorToast).toBeVisible();
    
    // Error should be in global error state
    const globalErrorIndicator = page.locator('[data-testid="global-error-indicator"]');
    if (await globalErrorIndicator.isVisible()) {
      await globalErrorIndicator.click();
      
      // Should show error details
      const errorDetails = page.locator('[data-testid="error-details"]');
      await expect(errorDetails).toBeVisible();
      
      // Clear errors
      await page.click('[data-testid="clear-errors"]');
      await expect(errorToast).toBeHidden();
    }
  });

  test('should track analytics events in global state', async ({ page }) => {
    // Enable analytics tracking
    await page.goto('/dashboard/settings');
    const analyticsToggle = page.locator('[data-testid="analytics-toggle"]');
    if (await analyticsToggle.isVisible()) {
      await analyticsToggle.check();
    }
    
    // Perform tracked actions
    await page.goto('/dashboard/cases');
    await page.click('[data-testid="case-item"]').first();
    
    // Check analytics state
    await page.goto('/dashboard/analytics');
    
    // Recent events should be displayed
    const recentEvents = page.locator('[data-testid="recent-events"]');
    if (await recentEvents.isVisible()) {
      const eventCount = await recentEvents.locator('.event-item').count();
      expect(eventCount).toBeGreaterThan(0);
      
      // Check for case view event
      const caseViewEvent = recentEvents.locator('.event-item:has-text("Case Viewed")');
      await expect(caseViewEvent).toBeVisible();
    }
  });
});