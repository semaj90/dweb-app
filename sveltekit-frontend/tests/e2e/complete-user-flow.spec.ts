import { test, expect, type Page } from '@playwright/test';
import { faker } from '@faker-js/faker';

/**
 * Complete End-to-End User Flow Test Suite
 * Tests: Register → Login → Profile → CRUD Operations → Database Persistence
 */

// Test data generation
const testUser = {
  email: faker.internet.email(),
  password: 'TestPassword123!',
  firstName: faker.person.firstName(),
  lastName: faker.person.lastName(),
  role: 'attorney' as const
};

// Helper function to wait for navigation and loading
async function waitForPageLoad(page: Page) {
  await page.waitForLoadState('networkidle');
  await page.waitForSelector('body', { state: 'attached' });
}

test.describe('Complete Legal AI Platform User Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Start from the homepage
    await page.goto('/');
    await waitForPageLoad(page);
  });

  test('Complete User Journey: Register → Login → Profile → CRUD Operations', async ({ page }) => {
    // ===== STEP 1: User Registration =====
    test.step('User Registration', async () => {
      await page.goto('/auth/register');
      await waitForPageLoad(page);

      // Fill registration form
      await page.fill('input[name="email"]', testUser.email);
      await page.fill('input[name="password"]', testUser.password);
      await page.fill('input[name="firstName"]', testUser.firstName);
      await page.fill('input[name="lastName"]', testUser.lastName);
      await page.selectOption('select[name="role"]', testUser.role);

      // Submit registration
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify successful registration
      await expect(page).toHaveURL(/\/auth\/login/);
      await expect(page.locator('.success-message')).toContainText('registered successfully');
    });

    // ===== STEP 2: User Login =====
    test.step('User Login', async () => {
      // Fill login form
      await page.fill('input[name="email"]', testUser.email);
      await page.fill('input[name="password"]', testUser.password);
      
      // Submit login
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify successful login and redirect to dashboard
      await expect(page).toHaveURL(/\/dashboard/);
      await expect(page.locator('[data-testid="user-welcome"]')).toContainText(testUser.firstName);
    });

    // ===== STEP 3: Profile Management =====
    test.step('Profile Access and Update', async () => {
      // Navigate to profile
      await page.click('[data-testid="profile-link"]');
      await waitForPageLoad(page);

      // Verify profile data
      await expect(page.locator('input[name="firstName"]')).toHaveValue(testUser.firstName);
      await expect(page.locator('input[name="lastName"]')).toHaveValue(testUser.lastName);
      await expect(page.locator('input[name="email"]')).toHaveValue(testUser.email);

      // Update profile
      const updatedBio = faker.lorem.paragraph();
      await page.fill('textarea[name="bio"]', updatedBio);
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify profile update success
      await expect(page.locator('.success-message')).toContainText('Profile updated');
      await expect(page.locator('textarea[name="bio"]')).toHaveValue(updatedBio);
    });

    // ===== STEP 4: Case CRUD Operations =====
    test.step('Case CRUD Operations', async () => {
      // Navigate to cases
      await page.click('[data-testid="cases-link"]');
      await waitForPageLoad(page);

      // CREATE: New case
      const caseTitle = faker.company.name() + ' vs ' + faker.company.name();
      const caseDescription = faker.lorem.paragraphs(2);

      await page.click('[data-testid="new-case-button"]');
      await page.fill('input[name="title"]', caseTitle);
      await page.fill('textarea[name="description"]', caseDescription);
      await page.selectOption('select[name="priority"]', 'high');
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify case creation
      await expect(page.locator('[data-testid="case-list"]')).toContainText(caseTitle);

      // READ: View case details
      await page.click(`[data-testid="case-item"]:has-text("${caseTitle}")`);
      await waitForPageLoad(page);
      await expect(page.locator('[data-testid="case-title"]')).toContainText(caseTitle);
      await expect(page.locator('[data-testid="case-description"]')).toContainText(caseDescription);

      // UPDATE: Edit case
      await page.click('[data-testid="edit-case-button"]');
      const updatedTitle = caseTitle + ' [Updated]';
      await page.fill('input[name="title"]', updatedTitle);
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify case update
      await expect(page.locator('[data-testid="case-title"]')).toContainText('[Updated]');
    });

    // ===== STEP 5: Evidence CRUD Operations =====
    test.step('Evidence CRUD Operations', async () => {
      // Navigate to evidence
      await page.click('[data-testid="evidence-link"]');
      await waitForPageLoad(page);

      // CREATE: Upload evidence
      const evidenceTitle = 'Test Document ' + Date.now();
      await page.click('[data-testid="upload-evidence-button"]');
      
      // Simulate file upload
      const fileContent = faker.lorem.paragraphs(3);
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles({
        name: 'test-document.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from(fileContent)
      });

      await page.fill('input[name="title"]', evidenceTitle);
      await page.fill('textarea[name="description"]', 'Test evidence description');
      await page.click('button[type="submit"]');
      await waitForPageLoad(page);

      // Verify evidence upload
      await expect(page.locator('[data-testid="evidence-list"]')).toContainText(evidenceTitle);

      // READ: View evidence
      await page.click(`[data-testid="evidence-item"]:has-text("${evidenceTitle}")`);
      await waitForPageLoad(page);
      await expect(page.locator('[data-testid="evidence-title"]')).toContainText(evidenceTitle);
    });

    // ===== STEP 6: Database Persistence Verification =====
    test.step('Database Persistence Verification', async () => {
      // Test API endpoints to verify database storage
      const response = await page.request.get('/api/users/me');
      expect(response.status()).toBe(200);
      const userData = await response.json();
      expect(userData.email).toBe(testUser.email);
      expect(userData.firstName).toBe(testUser.firstName);
      expect(userData.lastName).toBe(testUser.lastName);

      // Test cases API
      const casesResponse = await page.request.get('/api/cases');
      expect(casesResponse.status()).toBe(200);
      const casesData = await casesResponse.json();
      expect(casesData.length).toBeGreaterThan(0);

      // Test evidence API
      const evidenceResponse = await page.request.get('/api/evidence');
      expect(evidenceResponse.status()).toBe(200);
      const evidenceData = await evidenceResponse.json();
      expect(evidenceData.length).toBeGreaterThan(0);
    });

    // ===== STEP 7: Vector Search Integration =====
    test.step('Vector Search and AI Integration', async () => {
      // Navigate to search
      await page.click('[data-testid="search-link"]');
      await waitForPageLoad(page);

      // Perform semantic search
      const searchQuery = 'contract liability terms';
      await page.fill('input[data-testid="search-input"]', searchQuery);
      await page.click('button[data-testid="search-button"]');
      await waitForPageLoad(page);

      // Verify search results
      await expect(page.locator('[data-testid="search-results"]')).toBeVisible();
      
      // Test AI analysis
      await page.click('[data-testid="ai-analysis-button"]');
      await waitForPageLoad(page);
      
      // Verify AI analysis results
      await expect(page.locator('[data-testid="ai-analysis-results"]')).toBeVisible();
      await expect(page.locator('[data-testid="ai-analysis-results"]')).toContainText('Analysis');
    });

    // ===== STEP 8: User Logout =====
    test.step('User Logout', async () => {
      await page.click('[data-testid="user-menu"]');
      await page.click('[data-testid="logout-button"]');
      await waitForPageLoad(page);

      // Verify logout and redirect
      await expect(page).toHaveURL(/\/auth\/login/);
      await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    });
  });

  test('Database Connection and API Health Check', async ({ page }) => {
    // Test database connection
    const dbHealthResponse = await page.request.get('/api/db/health');
    expect(dbHealthResponse.status()).toBe(200);
    const dbHealth = await dbHealthResponse.json();
    expect(dbHealth.status).toBe('connected');

    // Test Go microservices
    const ragHealthResponse = await page.request.get('/api/go/enhanced-rag/health');
    expect(ragHealthResponse.status()).toBe(200);

    // Test vector operations
    const vectorHealthResponse = await page.request.get('/api/vectors/health');
    expect(vectorHealthResponse.status()).toBe(200);
  });

  test('Performance and Load Testing', async ({ page }) => {
    // Test page load performance
    const startTime = Date.now();
    await page.goto('/dashboard');
    await waitForPageLoad(page);
    const loadTime = Date.now() - startTime;
    
    expect(loadTime).toBeLessThan(5000); // Page should load within 5 seconds

    // Test API response times
    const apiStartTime = Date.now();
    const response = await page.request.get('/api/cases');
    const apiLoadTime = Date.now() - apiStartTime;
    
    expect(response.status()).toBe(200);
    expect(apiLoadTime).toBeLessThan(2000); // API should respond within 2 seconds
  });
});

test.describe('Error Handling and Edge Cases', () => {
  test('Invalid registration data', async ({ page }) => {
    await page.goto('/auth/register');
    
    // Test with invalid email
    await page.fill('input[name="email"]', 'invalid-email');
    await page.fill('input[name="password"]', 'short');
    await page.click('button[type="submit"]');
    
    await expect(page.locator('.error-message')).toContainText('Invalid email');
    await expect(page.locator('.error-message')).toContainText('Password must be');
  });

  test('Invalid login credentials', async ({ page }) => {
    await page.goto('/auth/login');
    
    await page.fill('input[name="email"]', 'nonexistent@example.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');
    
    await expect(page.locator('.error-message')).toContainText('Invalid credentials');
  });

  test('Unauthorized access protection', async ({ page }) => {
    // Try to access protected route without authentication
    await page.goto('/dashboard');
    await waitForPageLoad(page);
    
    // Should redirect to login
    await expect(page).toHaveURL(/\/auth\/login/);
  });
});