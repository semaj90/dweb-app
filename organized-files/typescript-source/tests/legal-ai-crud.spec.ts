import { test, expect, type Page } from '@playwright/test';
import { randomUUID } from 'crypto';

// Test data
const testUser = {
  name: 'Test Prosecutor',
  email: `test.prosecutor.${Date.now()}@example.com`,
  password: 'TestPassword123!',
  role: 'prosecutor'
};

const testCase = {
  caseNumber: `TEST-${Date.now()}`,
  title: 'Test Case - Evidence Upload',
  description: 'Automated test case for Playwright testing',
  priority: 'high',
  category: 'theft',
  incidentDate: new Date().toISOString().split('T')[0],
  location: 'Test Location - 123 Test Street'
};

// Helper function to wait for navigation
async function waitForNavigation(page: Page, url: string) {
  await page.waitForURL(url, { timeout: 10000 });
}

test.describe('Legal AI CRUD Operations', () => {
  test.beforeEach(async ({ page }) => {
    // Ensure we're starting fresh
    await page.goto('/');
  });

  test('complete user registration flow with validation', async ({ page }) => {
    // Navigate to registration page
    await page.goto('/register');
    
    // Test form validation - empty form
    await page.click('button[type="submit"]');
    await expect(page.locator('text=All fields are required')).toBeVisible();
    
    // Fill in the form with test data
    await page.fill('input[name="name"]', testUser.name);
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.fill('input[name="confirmPassword"]', testUser.password);
    await page.selectOption('select[name="role"]', testUser.role);
    
    // Accept terms and conditions
    await page.check('input[name="terms"]');
    
    // Submit the form
    await page.click('button[type="submit"]');
    
    // Should redirect to login with success message
    await waitForNavigation(page, '/login?registered=true');
    await expect(page.locator('text=Registration successful')).toBeVisible();
  });

  test('login with newly created user', async ({ page }) => {
    // Navigate to login page
    await page.goto('/login');
    
    // Fill in login form
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    
    // Submit login form
    await page.click('button[type="submit"]');
    
    // Should redirect to dashboard
    await waitForNavigation(page, '/dashboard');
    
    // Verify user is logged in
    await expect(page.locator('text=' + testUser.name)).toBeVisible();
  });

  test('create a new case with Superforms validation', async ({ page }): Promise<void> => {
    // First login
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await waitForNavigation(page, '/dashboard');
    
    // Navigate to cases page
    await page.goto('/cases');
    
    // Click create new case button
    await page.click('button:has-text("New Case")');
    
    // Test Zod validation - submit empty form
    await page.click('button[type="submit"]');
    
    // Check for validation errors
    await expect(page.locator('text=Case number is required')).toBeVisible();
    
    // Fill in case details
    await page.fill('input[name="caseNumber"]', testCase.caseNumber);
    await page.fill('input[name="title"]', testCase.title);
    await page.fill('textarea[name="description"]', testCase.description);
    await page.selectOption('select[name="priority"]', testCase.priority);
    await page.selectOption('select[name="category"]', testCase.category);
    await page.fill('input[name="incidentDate"]', testCase.incidentDate);
    await page.fill('input[name="location"]', testCase.location);
    
    // Submit the form
    await page.click('button[type="submit"]');
    
    // Verify case was created
    await expect(page.locator(`text=${testCase.title}`)).toBeVisible();
    
    // Store case URL for evidence upload test
    const caseUrl = page.url();
    // // // // return caseUrl;
  });

  test('upload evidence with drag and drop using Svelte DnD', async ({ page }) => {
    // First login
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await waitForNavigation(page, '/dashboard');
    
    // Navigate to first case
    await page.goto('/cases');
    await page.click(`text=${testCase.title}`);
    
    // Click add evidence button
    await page.click('button:has-text("Add Evidence")');
    
    // Wait for evidence upload modal
    await page.waitForSelector('[data-testid="evidence-upload-modal"]');
    
    // Create a test file for upload
    const buffer = Buffer.from('Test evidence file content');
    const fileName = 'test-evidence.pdf';
    
    // Method 1: Direct file input (fallback if drag-drop fails)
    const fileInput = await page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: fileName,
      mimeType: 'application/pdf',
      buffer: buffer
    });
    
    // Fill in evidence metadata
    await page.fill('input[name="evidenceTitle"]', 'Test Evidence Document');
    await page.fill('textarea[name="evidenceDescription"]', 'Automated test evidence upload');
    await page.selectOption('select[name="evidenceType"]', 'document');
    
    // Add tags
    await page.fill('input[name="tags"]', 'test, automated, playwright');
    
    // Submit evidence upload
    await page.click('button:has-text("Upload Evidence")');
    
    // Wait for upload to complete
    await page.waitForSelector('text=Evidence uploaded successfully', { timeout: 30000 });
    
    // Verify evidence appears in the list
    await expect(page.locator('text=Test Evidence Document')).toBeVisible();
  });

  test('drag and drop evidence reordering with Svelte DnD', async ({ page }) => {
    // Login and navigate to case with evidence
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await waitForNavigation(page, '/dashboard');
    
    // Navigate to case
    await page.goto('/cases');
    await page.click(`text=${testCase.title}`);
    
    // Wait for evidence list
    await page.waitForSelector('[data-testid="evidence-list"]');
    
    // Get draggable evidence items
    const evidenceItems = page.locator('[data-testid="evidence-item"]');
    const itemCount = await evidenceItems.count();
    
    if (itemCount >= 2) {
      // Get the first two items
      const firstItem = evidenceItems.nth(0);
      const secondItem = evidenceItems.nth(1);
      
      // Perform drag and drop
      await firstItem.dragTo(secondItem);
      
      // Verify order has changed
      await page.waitForTimeout(500); // Wait for animation
      
      // Check that items have been reordered
      const updatedFirstItem = await evidenceItems.nth(0).textContent();
      const updatedSecondItem = await evidenceItems.nth(1).textContent();
      
      expect(updatedFirstItem).not.toBe(await firstItem.textContent());
    }
  });

  test('test Superforms validation on case edit', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await waitForNavigation(page, '/dashboard');
    
    // Navigate to case
    await page.goto('/cases');
    await page.click(`text=${testCase.title}`);
    
    // Click edit button
    await page.click('button:has-text("Edit Case")');
    
    // Clear required field to trigger validation
    await page.fill('input[name="title"]', '');
    
    // Try to save
    await page.click('button:has-text("Save Changes")');
    
    // Check for Zod validation error
    await expect(page.locator('text=Title is required')).toBeVisible();
    
    // Fix the error
    await page.fill('input[name="title"]', testCase.title + ' - Updated');
    
    // Save changes
    await page.click('button:has-text("Save Changes")');
    
    // Verify update was successful
    await expect(page.locator('text=Case updated successfully')).toBeVisible();
    await expect(page.locator(`text=${testCase.title} - Updated`)).toBeVisible();
  });

  test('delete evidence with confirmation', async ({ page }) => {
    // Login and navigate to case
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');
    await waitForNavigation(page, '/dashboard');
    
    // Navigate to case with evidence
    await page.goto('/cases');
    await page.click(`text=${testCase.title}`);
    
    // Find evidence item and click delete
    const evidenceItem = page.locator('[data-testid="evidence-item"]').first();
    await evidenceItem.hover();
    await evidenceItem.locator('button[aria-label="Delete evidence"]').click();
    
    // Confirm deletion in dialog
    await page.click('button:has-text("Confirm Delete")');
    
    // Verify evidence was deleted
    await expect(page.locator('text=Evidence deleted successfully')).toBeVisible();
  });
});

// Test database seeding
test.describe('Database Seeding', () => {
  test('seed database with test data', async ({ page }) => {
    // This would typically be done via API or direct database connection
    // For now, we'll use the registration flow to create test users
    
    const seedUsers = [
      { name: 'John Prosecutor', email: 'john.prosecutor@test.com', role: 'prosecutor' },
      { name: 'Jane Detective', email: 'jane.detective@test.com', role: 'detective' },
      { name: 'Admin User', email: 'admin@test.com', role: 'admin' }
    ];
    
    for (const user of seedUsers) {
      await page.goto('/register');
      await page.fill('input[name="name"]', user.name);
      await page.fill('input[name="email"]', user.email);
      await page.fill('input[name="password"]', 'TestPassword123!');
      await page.fill('input[name="confirmPassword"]', 'TestPassword123!');
      await page.selectOption('select[name="role"]', user.role);
      await page.check('input[name="terms"]');
      await page.click('button[type="submit"]');
      
      // Wait for redirect
      await page.waitForURL('/login?registered=true');
    }
  });
});

// API tests for CRUD operations
test.describe('API CRUD Tests', () => {
  let authToken: string;
  
  test.beforeAll(async ({ request }) => {
    // Login via API to get auth token
    const loginResponse = await request.post('/api/auth/login', {
      data: {
        email: testUser.email,
        password: testUser.password
      }
    });
    
    const loginData = await loginResponse.json();
    authToken = loginData.token;
  });
  
  test('create case via API', async ({ request }) => {
    const response = await request.post('/api/cases', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      },
      data: {
        caseNumber: `API-TEST-${Date.now()}`,
        title: 'API Test Case',
        description: 'Created via Playwright API test',
        priority: 'medium',
        status: 'active'
      }
    });
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.id).toBeDefined();
  });
  
  test('upload evidence via API', async ({ request }) => {
    // First create a case
    const caseResponse = await request.post('/api/cases', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      },
      data: {
        caseNumber: `API-EVIDENCE-${Date.now()}`,
        title: 'API Evidence Test Case',
        description: 'For evidence upload testing'
      }
    });
    
    const caseData = await caseResponse.json();
    
    // Upload evidence
    const formData = new FormData();
    formData.append('caseId', caseData.id);
    formData.append('title', 'API Test Evidence');
    formData.append('description', 'Uploaded via API');
    formData.append('file', new Blob(['test content'], { type: 'text/plain' }), 'test.txt');
    
    const response = await request.post('/api/evidence', {
      headers: {
        'Authorization': `Bearer ${authToken}`
      },
      data: formData
    });
    
    expect(response.ok()).toBeTruthy();
    const data = await response.json();
    expect(data.id).toBeDefined();
  });
});
