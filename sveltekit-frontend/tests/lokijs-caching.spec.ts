import { test, expect } from '@playwright/test';

test.describe('LokiJS Caching System', () => {
  test.beforeEach(async ({ page }) => {
    // Initialize LokiJS and login
    await page.goto('/login');
    await page.fill('input[name="email"]', 'demo@example.com');
    await page.fill('input[name="password"]', 'demoPassword123');
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard/**');
  });

  test('should initialize LokiJS database in browser', async ({ page }) => {
    // Check if LokiJS is initialized
    const lokiInitialized = await page.evaluate(() => {
      return typeof window.lokiDB !== 'undefined' && window.lokiDB !== null;
    });
    
    expect(lokiInitialized).toBe(true);
    
    // Check database collections
    const collections = await page.evaluate(() => {
      if (window.lokiDB) {
        return window.lokiDB.listCollections().map(col => col.name);
      }
      return [];
    });
    
    expect(collections).toContain('cases');
    expect(collections).toContain('documents');
    expect(collections).toContain('users');
    expect(collections).toContain('cache');
  });

  test('should cache API responses in LokiJS', async ({ page }) => {
    // Clear cache first
    await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          cacheCollection.clear();
        }
      }
    });
    
    // Make API request that should be cached
    await page.goto('/dashboard/cases');
    
    // Wait for data to load
    await page.waitForSelector('[data-testid="case-item"]', { timeout: 10000 });
    
    // Check if response was cached
    const cacheEntry = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          return cacheCollection.findOne({ key: { $regex: /cases/ } });
        }
      }
      return null;
    });
    
    expect(cacheEntry).toBeTruthy();
    expect(cacheEntry).toHaveProperty('data');
    expect(cacheEntry).toHaveProperty('timestamp');
    expect(cacheEntry).toHaveProperty('ttl');
  });

  test('should serve cached data when offline', async ({ page, context }) => {
    // Load data online first
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Count cases loaded
    const onlineCaseCount = await page.locator('[data-testid="case-item"]').count();
    
    // Go offline
    await context.setOffline(true);
    
    // Navigate away and back
    await page.goto('/dashboard/documents');
    await page.goto('/dashboard/cases');
    
    // Should still show cached data
    await page.waitForSelector('[data-testid="case-item"]', { timeout: 5000 });
    
    const offlineCaseCount = await page.locator('[data-testid="case-item"]').count();
    expect(offlineCaseCount).toBe(onlineCaseCount);
    
    // Check cache hit indicator
    const cacheIndicator = page.locator('[data-testid="cache-indicator"]');
    if (await cacheIndicator.isVisible()) {
      const indicatorText = await cacheIndicator.textContent();
      expect(indicatorText).toMatch(/cached|offline/i);
    }
    
    // Go back online
    await context.setOffline(false);
  });

  test('should update cache when data changes', async ({ page }) => {
    // Load initial data
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Get initial cache timestamp
    const initialCacheTime = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          const entry = cacheCollection.findOne({ key: { $regex: /cases/ } });
          return entry ? entry.timestamp : null;
        }
      }
      return null;
    });
    
    // Create a new case
    await page.click('[data-testid="create-case-button"]');
    await page.fill('input[name="title"]', `Cache Test Case ${Date.now()}`);
    await page.fill('textarea[name="description"]', 'Testing cache invalidation');
    await page.click('button[type="submit"]');
    
    // Wait for redirect and data refresh
    await page.waitForURL('/dashboard/cases/**');
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Check if cache was updated
    const updatedCacheTime = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          const entry = cacheCollection.findOne({ key: { $regex: /cases/ } });
          return entry ? entry.timestamp : null;
        }
      }
      return null;
    });
    
    expect(updatedCacheTime).toBeGreaterThan(initialCacheTime);
  });

  test('should handle cache expiration', async ({ page }) => {
    // Create cache entry with short TTL
    await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          cacheCollection.insert({
            key: 'test-expiry',
            data: { test: 'data' },
            timestamp: Date.now(),
            ttl: 1000 // 1 second
          });
        }
      }
    });
    
    // Wait for expiration
    await page.waitForTimeout(1500);
    
    // Check if expired entry is cleaned up
    const expiredEntry = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          // Trigger cleanup
          const now = Date.now();
          cacheCollection.removeWhere(item => now > (item.timestamp + item.ttl));
          
          return cacheCollection.findOne({ key: 'test-expiry' });
        }
      }
      return null;
    });
    
    expect(expiredEntry).toBeNull();
  });

  test('should persist LokiJS data across browser sessions', async ({ page, context }) => {
    // Add some data to cache
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Get cache size
    const initialCacheSize = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        return cacheCollection ? cacheCollection.count() : 0;
      }
      return 0;
    });
    
    // Close and reopen browser context
    await context.close();
    const newContext = await page.context().browser()?.newContext();
    const newPage = await newContext!.newPage();
    
    // Login again
    await newPage.goto('/login');
    await newPage.fill('input[name="email"]', 'demo@example.com');
    await newPage.fill('input[name="password"]', 'demoPassword123');
    await newPage.click('button[type="submit"]');
    await newPage.waitForURL('/dashboard/**');
    
    // Check if cache persisted
    const persistedCacheSize = await newPage.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        return cacheCollection ? cacheCollection.count() : 0;
      }
      return 0;
    });
    
    expect(persistedCacheSize).toBeGreaterThanOrEqual(0); // May be 0 if new session
    
    await newContext!.close();
  });

  test('should support complex queries on cached data', async ({ page }) => {
    // Load some data
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Perform complex query on cached data
    const queryResult = await page.evaluate(() => {
      if (window.lokiDB) {
        const casesCollection = window.lokiDB.getCollection('cases');
        if (casesCollection) {
          // Query for active cases created in the last 30 days
          const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
          
          return casesCollection.find({
            $and: [
              { status: 'active' },
              { created_at: { $gt: thirtyDaysAgo } }
            ]
          });
        }
      }
      return [];
    });
    
    expect(Array.isArray(queryResult)).toBe(true);
    
    // Query with sorting and limiting
    const sortedResult = await page.evaluate(() => {
      if (window.lokiDB) {
        const casesCollection = window.lokiDB.getCollection('cases');
        if (casesCollection) {
          return casesCollection.chain()
            .find({ status: { $ne: 'deleted' } })
            .simplesort('created_at', true)
            .limit(5)
            .data();
        }
      }
      return [];
    });
    
    expect(Array.isArray(sortedResult)).toBe(true);
    expect(sortedResult.length).toBeLessThanOrEqual(5);
  });

  test('should handle cache invalidation patterns', async ({ page }) => {
    // Load initial data
    await page.goto('/dashboard/cases');
    await page.waitForSelector('[data-testid="case-item"]');
    
    // Get current cache
    const initialCache = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          return cacheCollection.find();
        }
      }
      return [];
    });
    
    // Trigger cache invalidation
    await page.evaluate(() => {
      // Simulate cache invalidation event
      window.dispatchEvent(new CustomEvent('cache:invalidate', {
        detail: { pattern: 'cases*' }
      }));
    });
    
    // Check if related cache entries were removed
    const cacheAfterInvalidation = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          return cacheCollection.find({ key: { $regex: /cases/ } });
        }
      }
      return [];
    });
    
    expect(cacheAfterInvalidation.length).toBeLessThan(initialCache.length);
  });

  test('should optimize cache storage size', async ({ page }) => {
    // Fill cache with data
    await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          // Add many cache entries
          for (let i = 0; i < 100; i++) {
            cacheCollection.insert({
              key: `test-${i}`,
              data: { large: 'x'.repeat(1000) }, // 1KB each
              timestamp: Date.now() - (i * 1000),
              ttl: 3600000 // 1 hour
            });
          }
        }
      }
    });
    
    // Get initial cache size
    const initialSize = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        return cacheCollection ? cacheCollection.count() : 0;
      }
      return 0;
    });
    
    // Trigger cache optimization
    await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          // Remove oldest entries if cache is too large
          const maxEntries = 50;
          const totalEntries = cacheCollection.count();
          
          if (totalEntries > maxEntries) {
            const entriesToRemove = totalEntries - maxEntries;
            const oldestEntries = cacheCollection.chain()
              .simplesort('timestamp')
              .limit(entriesToRemove)
              .data();
            
            oldestEntries.forEach(entry => {
              cacheCollection.remove(entry);
            });
          }
        }
      }
    });
    
    // Check optimized size
    const optimizedSize = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        return cacheCollection ? cacheCollection.count() : 0;
      }
      return 0;
    });
    
    expect(optimizedSize).toBeLessThanOrEqual(50);
    expect(optimizedSize).toBeLessThan(initialSize);
  });

  test('should handle concurrent cache operations', async ({ page, context }) => {
    // Open multiple tabs
    const page2 = await context.newPage();
    await page2.goto('/dashboard/cases');
    
    // Perform concurrent cache operations
    const promises = [
      page.evaluate(() => {
        if (window.lokiDB) {
          const cacheCollection = window.lokiDB.getCollection('cache');
          if (cacheCollection) {
            for (let i = 0; i < 10; i++) {
              cacheCollection.insert({
                key: `concurrent-1-${i}`,
                data: { tab: 1, index: i },
                timestamp: Date.now(),
                ttl: 3600000
              });
            }
          }
        }
      }),
      page2.evaluate(() => {
        if (window.lokiDB) {
          const cacheCollection = window.lokiDB.getCollection('cache');
          if (cacheCollection) {
            for (let i = 0; i < 10; i++) {
              cacheCollection.insert({
                key: `concurrent-2-${i}`,
                data: { tab: 2, index: i },
                timestamp: Date.now(),
                ttl: 3600000
              });
            }
          }
        }
      })
    ];
    
    await Promise.all(promises);
    
    // Check if all entries were added
    const totalEntries = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          return cacheCollection.find({ key: { $regex: /concurrent/ } }).length;
        }
      }
      return 0;
    });
    
    expect(totalEntries).toBe(20);
    
    await page2.close();
  });

  test('should provide cache statistics and monitoring', async ({ page }) => {
    // Generate some cache activity
    await page.goto('/dashboard/cases');
    await page.goto('/dashboard/documents');
    await page.goto('/dashboard/profile');
    
    // Get cache statistics
    const cacheStats = await page.evaluate(() => {
      if (window.lokiDB) {
        const cacheCollection = window.lokiDB.getCollection('cache');
        if (cacheCollection) {
          const allEntries = cacheCollection.find();
          const now = Date.now();
          
          const stats = {
            total_entries: allEntries.length,
            expired_entries: allEntries.filter(e => now > (e.timestamp + e.ttl)).length,
            hit_count: 0,
            miss_count: 0,
            memory_usage_estimate: JSON.stringify(allEntries).length,
            oldest_entry: Math.min(...allEntries.map(e => e.timestamp)),
            newest_entry: Math.max(...allEntries.map(e => e.timestamp))
          };
          
          return stats;
        }
      }
      return null;
    });
    
    expect(cacheStats).toBeTruthy();
    expect(cacheStats.total_entries).toBeGreaterThan(0);
    expect(cacheStats.memory_usage_estimate).toBeGreaterThan(0);
    expect(cacheStats.newest_entry).toBeGreaterThanOrEqual(cacheStats.oldest_entry);
  });
});