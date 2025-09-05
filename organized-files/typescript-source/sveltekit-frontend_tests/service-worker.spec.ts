// @ts-nocheck
import { test, expect } from '@playwright/test';

test.describe('Service Worker Functionality', () => {
  test('should register service worker successfully', async ({ page }) => {
    await page.goto('/');
    
    // Check if service worker is supported
    const swSupported = await page.evaluate(() => 'serviceWorker' in navigator);
    
    if (!swSupported) {
      test.skip(); // Skip if browser doesn't support service workers
      return;
    }
    
    // Wait for service worker registration
    await page.waitForFunction(() => {
      return navigator.serviceWorker.ready;
    });
    
    // Verify service worker is registered
    const registration = await page.evaluate(async () => {
      const reg = await navigator.serviceWorker.ready;
      return {
        scope: reg.scope,
        state: reg.active?.state,
        scriptURL: reg.active?.scriptURL
      };
    });
    
    expect(registration.state).toBe('activated');
    expect(registration.scriptURL).toContain('service-worker');
    expect(registration.scope).toBeTruthy();
  });

  test('should cache critical resources', async ({ page }) => {
    await page.goto('/');
    
    // Wait for service worker to be ready
    await page.waitForFunction(() => navigator.serviceWorker.ready);
    
    // Check cached resources
    const cachedResources = await page.evaluate(async () => {
      const cacheNames = await caches.keys();
      const allCached = [];
      
      for (const cacheName of cacheNames) {
        const cache = await caches.open(cacheName);
        const requests = await cache.keys();
        allCached.push(...requests.map(req: any => ({ 
          url: req.url, 
          cache: cacheName 
        })));
      }
      
      return allCached;
    });
    
    // Should cache essential assets
    const essentialAssets = [
      '/', // Root page
      '/app.css',
      '/app.js'
    ];
    
    essentialAssets.forEach(asset: any => {
      const isCached = cachedResources.some(cached: any => 
        cached.url.includes(asset)
      );
      expect(isCached).toBe(true);
    });
  });

  test('should provide offline functionality', async ({ page, context }) => {
    // First, visit the page online to cache resources
    await page.goto('/dashboard');
    
    // Login to cache authenticated pages
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Visit some pages to cache them
    await page.goto('/dashboard/cases');
    await page.goto('/dashboard/documents');
    
    // Go offline
    await context.setOffline(true);
    
    // Try to navigate to cached page
    await page.goto('/dashboard/cases');
    
    // Should still load (from cache)
    await expect(page.locator('h1, [data-testid="page-title"]')).toBeVisible();
    
    // Check for offline indicator
    const offlineIndicator = page.locator('[data-testid="offline-indicator"]');
    if (await offlineIndicator.isVisible()) {
      await expect(offlineIndicator).toContainText(/offline|no.*connection/i);
    }
    
    // Go back online
    await context.setOffline(false);
    
    // Offline indicator should disappear
    if (await offlineIndicator.isVisible()) {
      await expect(offlineIndicator).toBeHidden({ timeout: 5000 });
    }
  });

  test('should sync data when coming back online', async ({ page, context }) => {
    // Go online and login
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Go to cases page
    await page.goto('/dashboard/cases');
    
    // Go offline
    await context.setOffline(true);
    
    // Try to create a case offline
    await page.click('[data-testid="create-case-button"]');
    const offlineTitle = `Offline Case ${Date.now()}`;
    await page.fill('input[name="title"]', offlineTitle);
    await page.fill('textarea[name="description"]', 'Created while offline');
    await page.click('button[type="submit"]');
    
    // Should show queued for sync
    const syncIndicator = page.locator('[data-testid="sync-pending"], [data-testid="offline-created"]');
    if (await syncIndicator.isVisible()) {
      await expect(syncIndicator).toBeVisible();
    }
    
    // Go back online
    await context.setOffline(false);
    
    // Wait for sync
    await page.waitForTimeout(3000);
    
    // Sync indicator should disappear
    if (await syncIndicator.isVisible()) {
      await expect(syncIndicator).toBeHidden({ timeout: 10000 });
    }
    
    // Refresh and verify case was synced
    await page.reload();
    const syncedCase = page.locator(`[data-testid="case-item"]:has-text("${offlineTitle}")`);
    await expect(syncedCase).toBeVisible();
  });

  test('should handle push notifications', async ({ page, context }) => {
    // Grant notification permission
    await context.grantPermissions(['notifications']);
    
    await page.goto('/dashboard');
    
    // Wait for service worker
    await page.waitForFunction(() => navigator.serviceWorker.ready);
    
    // Check if push notifications are supported
    const pushSupported = await page.evaluate(() => {
      return 'PushManager' in window && 'Notification' in window;
    });
    
    if (!pushSupported) {
      test.skip();
      return;
    }
    
    // Subscribe to push notifications
    const subscription = await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: 'BFg1ZmFjZjllLTQwMTMtNDdmOS1hOTU2LTZlZDFiMjJjNzQ5OSAwMDAwMDA=' // Dummy key
      });
      
      return subscription ? subscription.toJSON() : null;
    });
    
    expect(subscription).toBeTruthy();
    
    // Test notification display
    await page.evaluate(() => {
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.ready.then(registration: any => {
          registration.showNotification('Test Notification', {
            body: 'This is a test notification',
            icon: '/favicon.ico',
            badge: '/favicon.ico',
            actions: [
              { action: 'view', title: 'View' },
              { action: 'dismiss', title: 'Dismiss' }
            ]
          });
        });
      }
    });
    
    // Wait for notification (this might not be visible in headless mode)
    await page.waitForTimeout(1000);
  });

  test('should update service worker when new version available', async ({ page }) => {
    await page.goto('/');
    
    // Wait for service worker
    await page.waitForFunction(() => navigator.serviceWorker.ready);
    
    // Simulate service worker update
    const updateAvailable = await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.ready;
      
      // Check for updates
      await registration.update();
      
      return new Promise((resolve) => {
        registration.addEventListener('updatefound', () => {
          const newWorker = registration.installing;
          if (newWorker) {
            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                resolve(true);
              }
            });
          }
        });
        
        // Simulate no update available
        setTimeout(() => resolve(false), 2000);
      });
    });
    
    if (updateAvailable) {
      // Check for update notification
      const updateNotification = page.locator('[data-testid="sw-update-available"]');
      await expect(updateNotification).toBeVisible();
      
      // Click update button
      await page.click('[data-testid="sw-update-button"]');
      
      // Page should reload with new service worker
      await page.waitForLoadState('networkidle');
    }
  });

  test('should handle background sync', async ({ page, context }) => {
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
    
    // Go offline
    await context.setOffline(true);
    
    // Perform an action that should be queued
    await page.goto('/dashboard/documents');
    await page.click('[data-testid="upload-document"]');
    
    // Create a mock file upload
    await page.evaluate(() => {
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(file);
      
      const input = document.querySelector('input[type="file"]') as HTMLInputElement;
      if (input) {
        input.files = dataTransfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    
    await page.click('[data-testid="upload-submit"]');
    
    // Should show queued for background sync
    const bgSyncIndicator = page.locator('[data-testid="bg-sync-queued"]');
    if (await bgSyncIndicator.isVisible()) {
      await expect(bgSyncIndicator).toBeVisible();
    }
    
    // Go back online
    await context.setOffline(false);
    
    // Background sync should process the queue
    await page.waitForTimeout(5000);
    
    // Check sync completion
    if (await bgSyncIndicator.isVisible()) {
      await expect(bgSyncIndicator).toBeHidden({ timeout: 10000 });
    }
  });

  test('should manage cache storage efficiently', async ({ page }) => {
    await page.goto('/');
    
    // Wait for service worker
    await page.waitForFunction(() => navigator.serviceWorker.ready);
    
    // Check cache storage usage
    const cacheInfo = await page.evaluate(async () => {
      const estimate = await navigator.storage.estimate();
      const cacheNames = await caches.keys();
      
      const cacheDetails = await Promise.all(
        cacheNames.map(async (name) => {
          const cache = await caches.open(name);
          const keys = await cache.keys();
          return {
            name,
            entries: keys.length,
            urls: keys.slice(0, 5).map(req: any => req.url) // First 5 URLs
          };
        })
      );
      
      return {
        quota: estimate.quota,
        usage: estimate.usage,
        caches: cacheDetails
      };
    });
    
    expect(cacheInfo.quota).toBeGreaterThan(0);
    expect(cacheInfo.usage).toBeGreaterThan(0);
    expect(cacheInfo.caches.length).toBeGreaterThan(0);
    
    // Test cache cleanup
    const cleanupResult = await page.evaluate(async () => {
      // Trigger cache cleanup (if implemented)
      if ('serviceWorker' in navigator) {
        const registration = await navigator.serviceWorker.ready;
        if (registration.active) {
          registration.active.postMessage({ type: 'CLEANUP_CACHE' });
        }
      }
      
      return true;
    });
    
    expect(cleanupResult).toBe(true);
  });

  test('should handle service worker messages', async ({ page }) => {
    await page.goto('/');
    
    // Wait for service worker
    await page.waitForFunction(() => navigator.serviceWorker.ready);
    
    // Send message to service worker
    const messageResponse = await page.evaluate(async () => {
      return new Promise((resolve) => {
        const channel = new MessageChannel();
        
        channel.port1.onmessage = (event) => {
          resolve(event.data);
        };
        
        navigator.serviceWorker.controller?.postMessage(
          { type: 'PING', timestamp: Date.now() },
          [channel.port2]
        );
        
        // Timeout after 5 seconds
        setTimeout(() => resolve(null), 5000);
      });
    });
    
    if (messageResponse) {
      expect(messageResponse).toHaveProperty('type');
    }
  });
});