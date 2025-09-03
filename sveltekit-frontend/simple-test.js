import { chromium } from 'playwright';

async function simpleTest() {
  console.log('🚀 Starting simple Playwright test...');
  
  try {
    console.log('📱 Launching browser...');
    const browser = await chromium.launch({ headless: false });
    const page = await browser.newPage();
    
    console.log('🌐 Navigating to localhost:5174...');
    await page.goto('http://localhost:5174');
    
    console.log('📄 Getting page title...');
    const title = await page.title();
    console.log('✅ Page title:', title);
    
    console.log('🔍 Looking for all-routes link...');
    const allRoutesLink = page.locator('a[href="/all-routes"]');
    const linkCount = await allRoutesLink.count();
    console.log(`📊 Found ${linkCount} all-routes links`);
    
    if (linkCount > 0) {
      console.log('🎯 Clicking first all-routes link...');
      await allRoutesLink.first().click();
      await page.waitForTimeout(2000);
      
      const newUrl = page.url();
      console.log('🌍 Current URL:', newUrl);
    }
    
    console.log('🧹 Closing browser...');
    await browser.close();
    console.log('✅ Test completed successfully!');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
  }
}

simpleTest();