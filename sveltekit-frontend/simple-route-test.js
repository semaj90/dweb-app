// Simple Route Testing Script
import { chromium } from 'playwright';

async function testAllRoutesPage() {
  console.log('🚀 Starting simple route test...');
  
  let browser;
  try {
    // Launch browser
    browser = await chromium.launch({ 
      headless: false,
      slowMo: 1000 
    });
    
    const page = await browser.newPage();
    
    // Navigate to all-routes page
    console.log('📍 Navigating to http://localhost:5178/all-routes');
    await page.goto('http://localhost:5178/all-routes', { 
      waitUntil: 'networkidle',
      timeout: 30000 
    });
    
    // Take screenshot
    await page.screenshot({ path: 'all-routes-screenshot.png' });
    console.log('📸 Screenshot saved as all-routes-screenshot.png');
    
    // Get page title
    const title = await page.title();
    console.log('📄 Page title:', title);
    
    // Check if page loaded correctly
    const hasRouteCards = await page.locator('[class*="Card"]').count();
    console.log('🔍 Found route cards:', hasRouteCards);
    
    // Look for the main heading
    const heading = await page.locator('h1').first().textContent();
    console.log('📝 Page heading:', heading);
    
    // Count navigation buttons
    const navButtons = await page.locator('button:has-text("Navigate"), button:has-text("NAVIGATE")').count();
    console.log('🔘 Navigation buttons found:', navButtons);
    
    // Try to click the first navigate button if available
    if (navButtons > 0) {
      console.log('🔘 Attempting to click first navigation button...');
      const firstButton = page.locator('button:has-text("Navigate"), button:has-text("NAVIGATE")').first();
      
      // Get button text and status
      const buttonText = await firstButton.textContent();
      const isDisabled = await firstButton.isDisabled();
      console.log(`   Button text: "${buttonText}", Disabled: ${isDisabled}`);
      
      if (!isDisabled) {
        await firstButton.click();
        console.log('✅ Successfully clicked navigation button');
        
        // Wait a moment and check URL
        await page.waitForTimeout(2000);
        const newUrl = page.url();
        console.log('🔄 New URL after click:', newUrl);
      } else {
        console.log('⚠️ Button is disabled');
      }
    }
    
    console.log('✅ Route testing completed successfully!');
    
  } catch (error) {
    console.error('❌ Route testing failed:', error.message);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

testAllRoutesPage();