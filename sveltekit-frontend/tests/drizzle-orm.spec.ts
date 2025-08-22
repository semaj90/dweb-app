
import { test, expect } from '@playwright/test';

test.describe('Drizzle ORM Database Operations', () => {
  let testUserId: string;
  let testCaseId: string;

  test.beforeAll(async ({ request }) => {
    // Create a test user for the suite
    const response = await request.post('/api/test/setup-user', {
      data: {
        email: `test-${Date.now()}@example.com`,
        password: 'TestPassword123!',
        name: 'Test User'
      }
    });
    
    if (response.ok()) {
      const data = await response.json();
      testUserId = data.userId;
    }
  });

  test.afterAll(async ({ request }) => {
    // Clean up test data
    if (testUserId) {
      await request.delete(`/api/test/cleanup-user/${testUserId}`);
    }
  });

  test('should perform CRUD operations on users table', async ({ page }) => {
    // Create user
    const createResponse = await page.request.post('/api/users', {
      data: {
        email: `drizzle-test-${Date.now()}@example.com`,
        password: 'SecurePass123!',
        name: 'Drizzle Test User',
        role: 'user'
      }
    });
    
    expect(createResponse.status()).toBe(201);
    const user = await createResponse.json();
    
    expect(user).toHaveProperty('id');
    expect(user).toHaveProperty('email');
    expect(user).toHaveProperty('name');
    expect(user).toHaveProperty('created_at');
    expect(user).not.toHaveProperty('password'); // Should not expose password
    
    // Read user
    const getResponse = await page.request.get(`/api/users/${user.id}`);
    expect(getResponse.status()).toBe(200);
    
    const fetchedUser = await getResponse.json();
    expect(fetchedUser.id).toBe(user.id);
    expect(fetchedUser.email).toBe(user.email);
    
    // Update user
    const updateResponse = await page.request.patch(`/api/users/${user.id}`, {
      data: {
        name: 'Updated Drizzle User',
        metadata: {
          last_login: new Date().toISOString()
        }
      }
    });
    
    expect(updateResponse.status()).toBe(200);
    const updatedUser = await updateResponse.json();
    expect(updatedUser.name).toBe('Updated Drizzle User');
    expect(updatedUser.metadata.last_login).toBeDefined();
    
    // Delete user
    const deleteResponse = await page.request.delete(`/api/users/${user.id}`);
    expect(deleteResponse.status()).toBe(204);
    
    // Verify deletion
    const verifyResponse = await page.request.get(`/api/users/${user.id}`);
    expect(verifyResponse.status()).toBe(404);
  });

  test('should handle transactions correctly', async ({ page }) => {
    // Test transaction rollback on error
    const response = await page.request.post('/api/cases/create-with-documents', {
      data: {
        case: {
          title: 'Transaction Test Case',
          description: 'Testing Drizzle transactions',
          user_id: testUserId
        },
        documents: [
          {
            title: 'Document 1',
            content: 'Content 1'
          },
          {
            title: 'Document 2',
            content: 'Content 2',
            invalid_field: 'This should cause an error'
          }
        ]
      }
    });
    
    // Transaction should rollback due to invalid document
    expect(response.status()).toBe(400);
    
    // Verify case was not created
    const casesResponse = await page.request.get('/api/cases', {
      params: {
        user_id: testUserId,
        title: 'Transaction Test Case'
      }
    });
    
    const cases = await casesResponse.json();
    expect(cases.length).toBe(0);
  });

  test('should handle complex queries with joins', async ({ page }) => {
    // Create a case with related data
    const caseResponse = await page.request.post('/api/cases', {
      data: {
        title: 'Complex Query Test',
        description: 'Testing Drizzle joins',
        user_id: testUserId,
        status: 'active'
      }
    });
    
    const caseData = await caseResponse.json();
    testCaseId = caseData.id;
    
    // Add documents to the case
    for (let i = 0; i < 3; i++) {
      await page.request.post('/api/documents', {
        data: {
          case_id: testCaseId,
          title: `Document ${i + 1}`,
          content: `Content for document ${i + 1}`,
          type: 'evidence'
        }
      });
    }
    
    // Query case with all related data
    const response = await page.request.get(`/api/cases/${testCaseId}/full`);
    expect(response.status()).toBe(200);
    
    const fullCase = await response.json();
    expect(fullCase).toHaveProperty('id');
    expect(fullCase).toHaveProperty('title');
    expect(fullCase).toHaveProperty('user');
    expect(fullCase).toHaveProperty('documents');
    
    expect(fullCase.user.id).toBe(testUserId);
    expect(fullCase.documents).toHaveLength(3);
    
    // Verify document structure
    fullCase.documents.forEach((doc: unknown) => {
      expect(doc).toHaveProperty('id');
      expect(doc).toHaveProperty('title');
      expect(doc).toHaveProperty('content');
      expect(doc).toHaveProperty('case_id');
      expect(doc.case_id).toBe(testCaseId);
    });
  });

  test('should handle pagination correctly', async ({ page }) => {
    // Create multiple documents
    const numDocs = 25;
    for (let i = 0; i < numDocs; i++) {
      await page.request.post('/api/documents', {
        data: {
          case_id: testCaseId,
          title: `Pagination Test Doc ${i}`,
          content: `Content ${i}`,
          type: 'document'
        }
      });
    }
    
    // Test pagination
    const pageSize = 10;
    let allDocuments: unknown[] = [];
    let currentPage = 1;
    let hasMore = true;
    
    while (hasMore) {
      const response = await page.request.get('/api/documents', {
        params: {
          case_id: testCaseId,
          page: currentPage,
          limit: pageSize
        }
      });
      
      expect(response.status()).toBe(200);
      const data = await response.json();
      
      expect(data).toHaveProperty('items');
      expect(data).toHaveProperty('total');
      expect(data).toHaveProperty('page');
      expect(data).toHaveProperty('pages');
      
      allDocuments = allDocuments.concat(data.items);
      hasMore = currentPage < data.pages;
      currentPage++;
    }
    
    // Verify we got all documents
    expect(allDocuments.length).toBeGreaterThanOrEqual(numDocs);
  });

  test('should handle database migrations', async ({ page }) => {
    // Check migration status
    const response = await page.request.get('/api/db/migrations/status');
    expect(response.status()).toBe(200);
    
    const status = await response.json();
    expect(status).toHaveProperty('current_version');
    expect(status).toHaveProperty('pending_migrations');
    expect(status).toHaveProperty('applied_migrations');
    
    // All migrations should be applied in test environment
    expect(status.pending_migrations).toHaveLength(0);
  });

  test('should handle complex filtering and sorting', async ({ page }) => {
    // Create test data with different attributes
    const testData = [
      { title: 'Alpha Case', priority: 'high', created_at: '2024-01-01' },
      { title: 'Beta Case', priority: 'medium', created_at: '2024-01-15' },
      { title: 'Gamma Case', priority: 'low', created_at: '2024-02-01' },
      { title: 'Delta Case', priority: 'high', created_at: '2024-02-15' }
    ];
    
    for (const data of testData) {
      await page.request.post('/api/cases', {
        data: {
          ...data,
          user_id: testUserId,
          description: 'Test case for filtering'
        }
      });
    }
    
    // Test complex filtering
    const filterResponse = await page.request.get('/api/cases', {
      params: {
        user_id: testUserId,
        priority: 'high',
        created_after: '2024-01-01',
        created_before: '2024-03-01',
        sort_by: 'created_at',
        order: 'desc'
      }
    });
    
    expect(filterResponse.status()).toBe(200);
    const filtered = await filterResponse.json();
    
    expect(filtered.items).toHaveLength(2);
    expect(filtered.items[0].title).toBe('Delta Case');
    expect(filtered.items[1].title).toBe('Alpha Case');
  });

  test('should handle bulk operations efficiently', async ({ page }) => {
    // Bulk insert
    const bulkData = Array(50).fill(null).map((_, i) => ({
      case_id: testCaseId,
      title: `Bulk Doc ${i}`,
      content: `Bulk content ${i}`,
      type: 'document'
    }));
    
    const startTime = Date.now();
    
    const bulkResponse = await page.request.post('/api/documents/bulk', {
      data: { documents: bulkData }
    });
    
    const endTime = Date.now();
    
    expect(bulkResponse.status()).toBe(201);
    const result = await bulkResponse.json();
    
    expect(result.created).toBe(50);
    expect(endTime - startTime).toBeLessThan(5000); // Should be fast
    
    // Bulk update
    const updateResponse = await page.request.patch('/api/documents/bulk', {
      data: {
        filter: { case_id: testCaseId, type: 'document' },
        update: { status: 'archived' }
      }
    });
    
    expect(updateResponse.status()).toBe(200);
    const updateResult = await updateResponse.json();
    expect(updateResult.updated).toBeGreaterThan(0);
  });

  test('should handle JSON fields correctly', async ({ page }) => {
    // Create document with JSON metadata
    const jsonData = {
      tags: ['legal', 'contract', 'review'],
      properties: {
        confidential: true,
        department: 'legal',
        review_date: '2024-03-01'
      },
      history: [
        { action: 'created', timestamp: '2024-01-01T10:00:00Z' },
        { action: 'reviewed', timestamp: '2024-01-15T14:00:00Z' }
      ]
    };
    
    const response = await page.request.post('/api/documents', {
      data: {
        case_id: testCaseId,
        title: 'JSON Test Document',
        content: 'Testing JSON fields',
        metadata: jsonData
      }
    });
    
    expect(response.status()).toBe(201);
    const document = await response.json();
    
    // Query by JSON fields
    const queryResponse = await page.request.get('/api/documents', {
      params: {
        'metadata.properties.confidential': true,
        'metadata.tags': 'contract'
      }
    });
    
    expect(queryResponse.status()).toBe(200);
    const results = await queryResponse.json();
    
    expect(results.items.length).toBeGreaterThan(0);
    const found = results.items.find((d: unknown) => d.id === document.id);
    expect(found).toBeDefined();
    expect(found.metadata.tags).toContain('contract');
  });

  test('should handle database constraints', async ({ page }) => {
    // Test unique constraint
    const email = `unique-test-${Date.now()}@example.com`;
    
    const firstResponse = await page.request.post('/api/users', {
      data: {
        email: email,
        password: 'Pass123!',
        name: 'First User'
      }
    });
    
    expect(firstResponse.status()).toBe(201);
    
    // Try to create another user with same email
    const secondResponse = await page.request.post('/api/users', {
      data: {
        email: email,
        password: 'Pass456!',
        name: 'Second User'
      }
    });
    
    expect(secondResponse.status()).toBe(409); // Conflict
    const error = await secondResponse.json();
    expect(error.error).toContain('unique');
    
    // Test foreign key constraint
    const invalidCaseResponse = await page.request.post('/api/documents', {
      data: {
        case_id: 'non-existent-case-id',
        title: 'Orphan Document',
        content: 'This should fail'
      }
    });
    
    expect(invalidCaseResponse.status()).toBe(400);
  });

  test('should handle optimistic locking', async ({ page }) => {
    // Create a document
    const createResponse = await page.request.post('/api/documents', {
      data: {
        case_id: testCaseId,
        title: 'Optimistic Lock Test',
        content: 'Original content',
        type: 'document'
      }
    });
    
    const document = await createResponse.json();
    const originalVersion = document.version || document.updated_at;
    
    // First update should succeed
    const update1Response = await page.request.patch(`/api/documents/${document.id}`, {
      data: {
        content: 'Updated content 1',
        version: originalVersion
      }
    });
    
    expect(update1Response.status()).toBe(200);
    
    // Second update with old version should fail
    const update2Response = await page.request.patch(`/api/documents/${document.id}`, {
      data: {
        content: 'Updated content 2',
        version: originalVersion // Using outdated version
      }
    });
    
    expect(update2Response.status()).toBe(409); // Conflict
    const error = await update2Response.json();
    expect(error.error).toContain('version');
  });
});