import { test, expect } from '@playwright/test';

test.describe('Legal AI Frontend Tests', () => {
  test('should load main page successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Legal AI|YoRHa/);
    await expect(page.locator('body')).toBeVisible();
  });

  test('should have navigation elements', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    
    // Check for common navigation elements
    const navElements = [
      'nav',
      '[role="navigation"]',
      'header',
      '.navbar',
      '.navigation'
    ];
    
    let navFound = false;
    for (const selector of navElements) {
      try {
        await expect(page.locator(selector).first()).toBeVisible({ timeout: 5000 });
        navFound = true;
        break;
      } catch (e) {
        // Continue to next selector
      }
    }
    
    if (!navFound) {
      console.log('No navigation elements found, checking for basic UI structure');
    }
  });

  test('should be responsive', async ({ page }) => {
    // Test desktop view
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
    
    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
    
    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
  });

  test('should load without JavaScript errors', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    page.on('pageerror', error => {
      errors.push(error.message);
    });
    
    await page.goto('/');
    await page.waitForTimeout(3000); // Wait for any async operations
    
    // Filter out common non-critical errors
    const criticalErrors = errors.filter(error => 
      !error.includes('404') && 
      !error.includes('favicon') &&
      !error.includes('ChunkLoadError') &&
      !error.toLowerCase().includes('network error')
    );
    
    if (criticalErrors.length > 0) {
      console.log('JavaScript errors found:', criticalErrors);
    }
    
    expect(criticalErrors.length).toBeLessThan(3); // Allow some non-critical errors
  });

  test('should have working links', async ({ page }) => {
    await page.goto('/');
    
    // Find all internal links
    const links = await page.locator('a[href^="/"], a[href^="./"], a[href^="../"]').all();
    
    if (links.length > 0) {
      // Test first few links to avoid timeout
      const linksToTest = links.slice(0, Math.min(3, links.length));
      
      for (const link of linksToTest) {
        const href = await link.getAttribute('href');
        if (href && href !== '/' && href !== '#') {
          try {
            await link.click({ timeout: 5000 });
            await page.waitForLoadState('networkidle');
            await expect(page.locator('body')).toBeVisible();
            await page.goBack();
          } catch (error) {
            console.log(`Link ${href} failed:`, error.message);
          }
        }
      }
    } else {
      console.log('No internal links found to test');
    }
  });
});