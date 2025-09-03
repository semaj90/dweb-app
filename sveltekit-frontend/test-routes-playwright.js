// Playwright Automation for Route Testing
// Tests all routes are clickable and accessible
import { chromium } from 'playwright';
import fs from 'fs';

const BASE_URL = 'http://localhost:5173';
const ALL_ROUTES_PATH = '/all-routes';

class RouteTestingAutomator {
  constructor() {
    this.browser = null;
    this.page = null;
    this.testResults = {};
    this.totalRoutes = 0;
    this.successfulRoutes = 0;
    this.failedRoutes = 0;
    this.timeoutRoutes = 0;
  }

  async initialize() {
    console.log('🚀 Initializing Playwright Route Testing...');
    
    // Launch browser with proper configuration
    this.browser = await chromium.launch({
      headless: false, // Set to true for headless mode
      slowMo: 100, // Slow down actions for visibility
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--allow-running-insecure-content'
      ]
    });
    
    // Create new page
    this.page = await this.browser.newPage();
    
    // Set viewport for consistent testing
    await this.page.setViewportSize({ width: 1920, height: 1080 });
    
    // Set reasonable timeout
    this.page.setDefaultTimeout(30000);
    
    console.log('✅ Browser initialized');
  }

  async navigateToAllRoutes() {
    console.log(`📍 Navigating to ${BASE_URL}${ALL_ROUTES_PATH}...`);
    
    try {
      await this.page.goto(`${BASE_URL}${ALL_ROUTES_PATH}`, { 
        waitUntil: 'networkidle', 
        timeout: 60000 
      });
      
      // Wait for the page to fully load
      await this.page.waitForSelector('h1:has-text("BITS-UI V2 ROUTES CENTER")', { timeout: 30000 });
      
      console.log('✅ All Routes page loaded successfully');
      return true;
    } catch (error) {
      console.error('❌ Failed to load All Routes page:', error.message);
      return false;
    }
  }

  async getDiscoveredRoutes() {
    console.log('🔍 Discovering available routes...');
    
    try {
      // Wait for route cards to load
      await this.page.waitForSelector('[class*="Card"]', { timeout: 15000 });
      
      // Extract all route information from the page
      const routes = await this.page.evaluate(() => {
        const routeCards = document.querySelectorAll('[class*="Card"]:has(code)');
        const routes = [];
        
        routeCards.forEach(card => {
          const codeElement = card.querySelector('code');
          const titleElement = card.querySelector('h3');
          const statusBadges = card.querySelectorAll('[class*="Badge"]');
          const navigateButton = card.querySelector('button:has-text("Navigate"), button:has-text("NAVIGATE")');
          
          if (codeElement && titleElement) {
            const route = codeElement.textContent.trim();
            const title = titleElement.textContent.trim();
            
            // Check if route is available
            let available = true;
            let status = 'unknown';
            
            statusBadges.forEach(badge => {
              const text = badge.textContent.trim().toLowerCase();
              if (text.includes('missing') || text.includes('unavailable')) {
                available = false;
              } else if (text.includes('active') || text.includes('beta') || text.includes('experimental')) {
                status = text;
              }
            });
            
            // Also check button state
            if (navigateButton && (navigateButton.disabled || navigateButton.textContent.includes('Unavailable'))) {
              available = false;
            }
            
            routes.push({
              route,
              title,
              available,
              status,
              hasNavigateButton: !!navigateButton
            });
          }
        });
        
        return routes;
      });
      
      this.totalRoutes = routes.length;
      console.log(`✅ Found ${this.totalRoutes} routes to test`);
      
      // Log discovered routes
      routes.forEach(route => {
        console.log(`   ${route.available ? '✅' : '❌'} ${route.route} - ${route.title}`);
      });
      
      return routes;
    } catch (error) {
      console.error('❌ Failed to discover routes:', error.message);
      return [];
    }
  }

  async testRouteClickability(route) {
    console.log(`🧪 Testing route: ${route.route}`);
    
    if (!route.available) {
      console.log(`   ⏭️ Skipping unavailable route: ${route.route}`);
      this.testResults[route.route] = {
        status: 'skipped',
        reason: 'Route marked as unavailable',
        timestamp: new Date().toISOString()
      };
      return 'skipped';
    }
    
    try {
      // Find the navigate button for this specific route
      const routeCard = await this.page.locator(`code:has-text("${route.route}")`).locator('..').locator('..');
      const navigateButton = routeCard.locator('button:has-text("Navigate"), button:has-text("NAVIGATE")').first();
      
      // Check if button exists and is enabled
      const buttonExists = await navigateButton.count() > 0;
      if (!buttonExists) {
        throw new Error('Navigate button not found');
      }
      
      const isDisabled = await navigateButton.isDisabled();
      if (isDisabled) {
        throw new Error('Navigate button is disabled');
      }
      
      // Record starting URL
      const startUrl = this.page.url();
      
      // Click the navigate button
      await navigateButton.click();
      
      // Wait for navigation or page change
      const navigationPromise = this.page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => null);
      const urlChangePromise = this.page.waitForFunction(
        (startUrl) => window.location.href !== startUrl,
        startUrl,
        { timeout: 10000 }
      ).catch(() => null);
      
      await Promise.race([navigationPromise, urlChangePromise]);
      
      // Check final URL
      const finalUrl = this.page.url();
      const expectedPath = route.route;
      
      if (finalUrl.includes(expectedPath) || finalUrl !== startUrl) {
        console.log(`   ✅ Successfully navigated to: ${finalUrl}`);
        this.successfulRoutes++;
        this.testResults[route.route] = {
          status: 'success',
          finalUrl,
          expectedPath,
          timestamp: new Date().toISOString()
        };
        
        // Navigate back to all-routes page
        await this.page.goto(`${BASE_URL}${ALL_ROUTES_PATH}`, { waitUntil: 'networkidle', timeout: 15000 });
        await this.page.waitForSelector('h1:has-text("BITS-UI V2 ROUTES CENTER")', { timeout: 10000 });
        
        return 'success';
      } else {
        throw new Error(`Navigation failed: expected ${expectedPath}, got ${finalUrl}`);
      }
      
    } catch (error) {
      console.log(`   ❌ Failed: ${error.message}`);
      this.failedRoutes++;
      this.testResults[route.route] = {
        status: 'error',
        error: error.message,
        timestamp: new Date().toISOString()
      };
      
      // Try to get back to all-routes page if we're lost
      try {
        await this.page.goto(`${BASE_URL}${ALL_ROUTES_PATH}`, { waitUntil: 'networkidle', timeout: 10000 });
      } catch (navError) {
        console.log(`   ⚠️ Could not return to all-routes page: ${navError.message}`);
      }
      
      return 'error';
    }
  }

  async testAllRoutes() {
    console.log('🎯 Starting comprehensive route testing...');
    
    const routes = await this.getDiscoveredRoutes();
    if (routes.length === 0) {
      console.log('❌ No routes discovered, aborting test');
      return;
    }
    
    // Filter to only available routes for testing
    const availableRoutes = routes.filter(r => r.available);
    console.log(`🚀 Testing ${availableRoutes.length} available routes out of ${routes.length} total`);
    
    let testIndex = 0;
    for (const route of availableRoutes) {
      testIndex++;
      console.log(`\\n[${testIndex}/${availableRoutes.length}] Testing: ${route.route}`);
      
      const result = await this.testRouteClickability(route);
      
      // Add small delay between tests
      await this.page.waitForTimeout(1000);
      
      // Log progress
      console.log(`Progress: ${testIndex}/${availableRoutes.length} (${((testIndex/availableRoutes.length)*100).toFixed(1)}%)`);
    }
    
    console.log('\\n🏁 Route testing completed!');
  }

  async useBuiltInTester() {
    console.log('🔧 Using built-in route tester...');
    
    try {
      // Look for the "Test All" button
      const testAllButton = this.page.locator('button:has-text("Test All"), button:has-text("🧪 Test All")');
      
      if (await testAllButton.count() > 0) {
        console.log('✅ Found built-in test button, clicking...');
        await testAllButton.click();
        
        // Wait for testing to start
        await this.page.waitForSelector('text="ROUTE TESTING IN PROGRESS"', { timeout: 10000 });
        console.log('🔄 Built-in testing started...');
        
        // Wait for testing to complete (look for completion indicators)
        const maxWaitTime = 300000; // 5 minutes
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
          const isTestingActive = await this.page.locator('text="ROUTE TESTING IN PROGRESS"').count() > 0;
          if (!isTestingActive) {
            console.log('✅ Built-in testing completed!');
            break;
          }
          await this.page.waitForTimeout(2000);
        }
        
        // Extract test results from the page
        const builtInResults = await this.page.evaluate(() => {
          const results = {};
          const routeCards = document.querySelectorAll('[class*="Card"]:has(code)');
          
          routeCards.forEach(card => {
            const codeElement = card.querySelector('code');
            const badges = card.querySelectorAll('[class*="Badge"]');
            
            if (codeElement) {
              const route = codeElement.textContent.trim();
              
              badges.forEach(badge => {
                const text = badge.textContent.trim().toLowerCase();
                if (text === 'success' || text === 'error' || text === 'timeout') {
                  results[route] = text;
                }
              });
            }
          });
          
          return results;
        });
        
        console.log('📊 Built-in test results:', builtInResults);
        return builtInResults;
      } else {
        console.log('⚠️ Built-in test button not found, using manual testing');
        return null;
      }
    } catch (error) {
      console.log('❌ Built-in tester failed:', error.message);
      return null;
    }
  }

  generateReport() {
    console.log('\\n📊 GENERATING TEST REPORT');
    console.log('========================');
    
    const report = {
      summary: {
        totalRoutes: this.totalRoutes,
        successfulRoutes: this.successfulRoutes,
        failedRoutes: this.failedRoutes,
        timeoutRoutes: this.timeoutRoutes,
        skippedRoutes: Object.values(this.testResults).filter(r => r.status === 'skipped').length,
        successRate: this.totalRoutes > 0 ? ((this.successfulRoutes / this.totalRoutes) * 100).toFixed(2) + '%' : '0%'
      },
      testResults: this.testResults,
      timestamp: new Date().toISOString()
    };
    
    console.log('\\n📈 SUMMARY:');
    console.log(`   Total Routes: ${report.summary.totalRoutes}`);
    console.log(`   ✅ Successful: ${report.summary.successfulRoutes}`);
    console.log(`   ❌ Failed: ${report.summary.failedRoutes}`);
    console.log(`   ⏱️ Timeouts: ${report.summary.timeoutRoutes}`);
    console.log(`   ⏭️ Skipped: ${report.summary.skippedRoutes}`);
    console.log(`   📊 Success Rate: ${report.summary.successRate}`);
    
    console.log('\\n📋 DETAILED RESULTS:');
    Object.entries(this.testResults).forEach(([route, result]) => {
      const icon = result.status === 'success' ? '✅' : 
                   result.status === 'error' ? '❌' : 
                   result.status === 'skipped' ? '⏭️' : '⏱️';
      console.log(`   ${icon} ${route}: ${result.status}`);
      if (result.error) {
        console.log(`      Error: ${result.error}`);
      }
    });
    
    // Save report to file
    const reportPath = 'route-test-report.json';
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\\n📄 Report saved to: ${reportPath}`);
    
    return report;
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
      console.log('🧹 Browser closed');
    }
  }
}

// Main execution
async function runRouteTests() {
  const tester = new RouteTestingAutomator();
  
  try {
    await tester.initialize();
    
    const loaded = await tester.navigateToAllRoutes();
    if (!loaded) {
      throw new Error('Could not load all-routes page');
    }
    
    // Try built-in tester first
    const builtInResults = await tester.useBuiltInTester();
    
    if (!builtInResults) {
      // Fall back to manual testing
      await tester.testAllRoutes();
    } else {
      // Process built-in results
      Object.entries(builtInResults).forEach(([route, status]) => {
        tester.testResults[route] = {
          status,
          source: 'built-in-tester',
          timestamp: new Date().toISOString()
        };
        
        if (status === 'success') tester.successfulRoutes++;
        else if (status === 'error') tester.failedRoutes++;
        else if (status === 'timeout') tester.timeoutRoutes++;
      });
      tester.totalRoutes = Object.keys(builtInResults).length;
    }
    
    const report = tester.generateReport();
    
    console.log('\\n🎉 Route testing completed successfully!');
    console.log(`📊 Final Success Rate: ${report.summary.successRate}`);
    
  } catch (error) {
    console.error('💥 Route testing failed:', error.message);
    console.error(error.stack);
  } finally {
    await tester.cleanup();
  }
}

// Check if running directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runRouteTests().catch(console.error);
}

export { RouteTestingAutomator, runRouteTests };