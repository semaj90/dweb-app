import { chromium } from 'playwright';

console.log('🧪 Starting Session Debug Test...');

const browser = await chromium.launch({ headless: false, devtools: true });
const page = await browser.newPage();

try {
  // Test 1: Homepage not logged in
  console.log('📝 Test 1: Checking homepage when not logged in...');
  await page.goto('http://localhost:5173/');
  await page.waitForSelector('.session-debug-panel', { timeout: 10000 });
  
  const debugPanel = await page.locator('.session-debug-panel').isVisible();
  console.log(`✅ Session debug panel visible: ${debugPanel}`);
  
  const statusMessage = await page.locator('.status-message').textContent();
  console.log(`📊 Status message: "${statusMessage}"`);
  
  if (statusMessage.includes('Not logged in')) {
    console.log('✅ Test 1 PASSED: Shows "Not logged in" correctly');
  } else {
    console.log('❌ Test 1 FAILED: Should show "Not logged in"');
  }
  
  // Test 2: Login flow
  console.log('📝 Test 2: Testing login flow...');
  await page.click('.action-btn.login');
  await page.waitForURL('**/auth/login');
  console.log('✅ Navigated to login page');
  
  await page.fill('input[name="email"]', 'demo@legalai.gov');
  await page.fill('input[name="password"]', 'demo123');
  console.log('✅ Filled login form');
  
  await page.click('button[type="submit"]');
  await page.waitForURL('**/dashboard', { timeout: 15000 });
  console.log('✅ Login successful - redirected to dashboard');
  
  // Test 3: Check dashboard session info
  console.log('📝 Test 3: Checking dashboard session info...');
  const dashboardStatusMessage = await page.locator('.status-message').textContent();
  console.log(`📊 Dashboard status: "${dashboardStatusMessage}"`);
  
  if (dashboardStatusMessage.includes('Logged in as: demo@legalai.gov')) {
    console.log('✅ Test 3 PASSED: Dashboard shows correct user info');
  } else {
    console.log('❌ Test 3 FAILED: Dashboard should show user info');
  }
  
  // Test 4: Session ID verification
  const sessionId = await page.locator('.session-id').textContent();
  console.log(`📊 Session ID: ${sessionId}`);
  
  if (sessionId && sessionId.length > 20) {
    console.log('✅ Test 4 PASSED: Session ID looks valid');
  } else {
    console.log('❌ Test 4 FAILED: Session ID should be present');
  }
  
  // Test 5: Navigate back to homepage and verify session persistence
  console.log('📝 Test 5: Testing session persistence on homepage...');
  await page.goto('http://localhost:5173/');
  await page.waitForSelector('.session-debug-panel');
  
  const homeStatusMessage = await page.locator('.status-message').textContent();
  console.log(`📊 Homepage status after login: "${homeStatusMessage}"`);
  
  if (homeStatusMessage.includes('Logged in as: demo@legalai.gov')) {
    console.log('✅ Test 5 PASSED: Session persisted on homepage');
  } else {
    console.log('❌ Test 5 FAILED: Session should persist on homepage');
  }
  
  // Test 6: Logout functionality
  console.log('📝 Test 6: Testing logout functionality...');
  await page.click('.action-btn.logout');
  await page.waitForURL('http://localhost:5173/');
  await page.waitForTimeout(1000); // Give time for UI to update
  
  const loggedOutStatus = await page.locator('.status-message').textContent();
  console.log(`📊 Status after logout: "${loggedOutStatus}"`);
  
  if (loggedOutStatus.includes('Not logged in')) {
    console.log('✅ Test 6 PASSED: Logout successful');
  } else {
    console.log('❌ Test 6 FAILED: Should show "Not logged in" after logout');
  }
  
  console.log('\n🎉 SESSION DEBUG TESTS COMPLETE!');
  
} catch (error) {
  console.error('❌ Test failed with error:', error.message);
} finally {
  await browser.close();
}