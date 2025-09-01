
import { test, expect } from '@playwright/test';
import { URL } from "url";

test.describe('Complete User Flow E2E Tests', () => {
  
  test('User Registration → Login → Profile → CRUD Operations', async ({ page }) => {
    const timestamp = Date.now();
    const testEmail = `test-${timestamp}@example.com`;
    
    // 1. Navigate to registration
    await page.goto('http://localhost:5175/auth/register');
    await expect(page).toHaveTitle(/Register/);
    
    // 2. Register new user
    await page.fill('[data-testid="email"]', testEmail);
    await page.fill('[data-testid="password"]', 'TestPassword123!');
    await page.fill('[data-testid="firstName"]', 'Test');
    await page.fill('[data-testid="lastName"]', 'User');
    await page.fill('[data-testid="username"]', `testuser${timestamp}`);
    
    await page.click('[data-testid="register-button"]');
    
    // Should redirect to login or dashboard
    await page.waitForURL(/(login|dashboard)/);
    
    // 3. Login with new credentials
    const currentUrl = page.url();
    if (currentUrl.includes('/login')) {
      await page.fill('[data-testid="email"]', testEmail);
      await page.fill('[data-testid="password"]', 'TestPassword123!');
      await page.click('[data-testid="login-button"]');
    }
    
    // 4. Verify dashboard access
    await page.waitForURL(/\/dashboard/);
    await expect(page.locator('[data-testid="user-profile"]')).toBeVisible();
    
    // 5. Navigate to user profile
    await page.click('[data-testid="profile-link"]');
    await page.waitForURL(/\/profile/);
    
    // 6. Test CRUD operations
    
    // CREATE: Add a new case/document
    await page.click('[data-testid="new-case-button"]');
    await page.fill('[data-testid="case-title"]', 'Test Legal Case');
    await page.fill('[data-testid="case-description"]', 'E2E Test Case Description');
    await page.click('[data-testid="save-case"]');
    
    // READ: Verify case appears in list
    await expect(page.locator('[data-testid="case-list"]')).toContainText('Test Legal Case');
    
    // UPDATE: Edit the case
    await page.click('[data-testid="edit-case-button"]');
    await page.fill('[data-testid="case-title"]', 'Updated Test Case');
    await page.click('[data-testid="update-case"]');
    
    // Verify update
    await expect(page.locator('[data-testid="case-list"]')).toContainText('Updated Test Case');
    
    // DELETE: Remove the case
    await page.click('[data-testid="delete-case-button"]');
    await page.click('[data-testid="confirm-delete"]');
    
    // Verify deletion
    await expect(page.locator('[data-testid="case-list"]')).not.toContainText('Updated Test Case');
  });
  
  test('API Integration Tests', async ({ page }) => {
    // Test API endpoints directly
    
    // 1. Test user registration API
    const response = await page.request.post('http://localhost:5175/api/auth/register', {
      data: {
        email: `api-test-${Date.now()}@example.com`,
        password: 'ApiTest123!',
        firstName: 'API',
        lastName: 'Test'
      }
    });
    
    expect(response.ok()).toBeTruthy();
    const userData = await response.json();
    expect(userData.user).toBeDefined();
    expect(userData.user.email).toBeDefined();
    
    // 2. Test vector service proxy
    const vectorResponse = await page.request.get('http://localhost:5175/api/vectors');
    expect(vectorResponse.ok()).toBeTruthy();
    
    const vectorData = await vectorResponse.json();
    expect(vectorData.service).toBe('vector-proxy');
  });
  
  test('Database Integration Test', async ({ page }) => {
    // Test database operations through API
    
    // 1. Get users list
    const usersResponse = await page.request.get('http://localhost:5175/api/users?limit=5');
    expect(usersResponse.ok()).toBeTruthy();
    
    const usersData = await usersResponse.json();
    expect(usersData.users).toBeDefined();
    expect(Array.isArray(usersData.users)).toBeTruthy();
  });
  
  test('Go Microservices Integration', async ({ page }) => {
    // Test Go services directly
    
    // 1. Enhanced RAG Service
    const ragResponse = await page.request.get('http://localhost:8094/api/health');
    expect(ragResponse.ok()).toBeTruthy();
    
    // 2. Vector Service
    const vectorResponse = await page.request.get('http://localhost:8095/api/health');
    expect(vectorResponse.ok()).toBeTruthy();
    
    // 3. Upload Service
    const uploadResponse = await page.request.get('http://localhost:8093/health');
    expect(uploadResponse.ok()).toBeTruthy();
  });
  
});
