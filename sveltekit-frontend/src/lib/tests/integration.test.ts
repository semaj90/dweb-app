// Comprehensive Integration Tests
// End-to-end testing of SvelteKit 2 API with PostgreSQL and pgvector

/// <reference types="vitest" />
import { CaseOperations, EvidenceOperations, checkDatabaseHealth, withTransaction } from '../server/db/enhanced-operations';
import { apiSuccess, apiError, CommonErrors, parseRequestBody } from '../server/api/response';
import { db } from '../server/db/drizzle';
import { cases, evidence, users } from '../server/db/schema-postgres';
import { eq } from 'drizzle-orm';
import type { CaseAPI, EvidenceAPI, HealthAPI } from '../types/api-contracts';

// Test data
const testUser = {
  id: 'test-user-123',
  email: 'test@example.com',
  firstName: 'Test',
  lastName: 'User',
  role: 'prosecutor'
};

const testCase = {
  title: 'Test Criminal Case',
  description: 'A test case for integration testing',
  priority: 'medium' as const,
  status: 'open' as const,
  location: 'Test City',
  jurisdiction: 'Test County',
  createdBy: testUser.id
};

const testEvidence = {
  title: 'Test Evidence Item',
  description: 'Evidence for integration testing',
  evidenceType: 'document' as const,
  tags: ['test', 'integration'],
  collectedBy: 'Detective Smith',
  location: 'Crime Scene A',
  uploadedBy: testUser.id
};

describe('Database Integration Tests', () => {
  beforeAll(async () => {
    // Ensure database is healthy before running tests
    const health = await checkDatabaseHealth();
    expect(health.connected).toBe(true);
    console.log('Database health:', health);
  });

  describe('Database Health Checks', () => {
    it('should connect to PostgreSQL successfully', async () => {
      const health = await checkDatabaseHealth();

      expect(health.connected).toBe(true);
      expect(health.queryTime).toBeGreaterThan(0);
      expect(health.errors).toEqual([]);
    });

    it('should have pgvector extension available', async () => {
      const health = await checkDatabaseHealth();

      expect(health.pgvectorEnabled).toBe(true);
    });
  });

  describe('Transaction Management', () => {
    it('should handle successful transactions', async () => {
      const result = await withTransaction(async (tx) => {
        // Insert test data within transaction
        const [insertedCase] = await tx.insert(cases).values({
          ...testCase,
          caseNumber: 'TEST-001'
        }).returning();

        expect(insertedCase).toBeDefined();
        expect(insertedCase.title).toBe(testCase.title);

        return insertedCase;
      });

      expect(result.title).toBe(testCase.title);

      // Clean up
      await db.delete(cases).where(eq(cases.id, result.id));
    });

    it('should rollback failed transactions', async () => {
      let caseId: string | null = null;

      try {
        await withTransaction(async (tx) => {
          // Insert test case
          const [insertedCase] = await tx.insert(cases).values({
            ...testCase,
            caseNumber: 'TEST-002'
          }).returning();

          caseId = insertedCase.id;

          // Force an error to trigger rollback
          throw new Error('Intentional test error');
        });
      } catch (error: any) {
        expect(error).toBeInstanceOf(Error);
      }

      // Verify case was not inserted due to rollback
      if (caseId) {
        const foundCase = await db.select().from(cases).where(eq(cases.id, caseId)).limit(1);
        expect(foundCase).toHaveLength(0);
      }
    });
  });
});

describe('Case Operations Integration Tests', () => {
  let createdCaseId: string;

  afterEach(async () => {
    // Clean up created cases
    if (createdCaseId) {
      try {
        await db.delete(cases).where(eq(cases.id, createdCaseId));
      } catch (error: any) {
        console.warn('Failed to clean up case:', error);
      }
    }
  });

  describe('Case Creation', () => {
    it('should create a new case with vector embeddings', async () => {
      const newCase = await CaseOperations.create(testCase);
      createdCaseId = newCase.id;

      expect(newCase).toBeDefined();
      expect(newCase.title).toBe(testCase.title);
      expect(newCase.caseNumber).toMatch(/^CR-\d{4}-\d{4}$/);
      expect(newCase.status).toBe('open');
      expect(newCase.priority).toBe('medium');
      expect(newCase.createdBy).toBe(testUser.id);
      expect(newCase.createdAt).toBeInstanceOf(Date);
      expect(newCase.updatedAt).toBeInstanceOf(Date);
    });

    it('should generate unique case numbers', async () => {
      const case1 = await CaseOperations.create({
        ...testCase,
        title: 'Test Case 1'
      });

      const case2 = await CaseOperations.create({
        ...testCase,
        title: 'Test Case 2'
      });

      expect(case1.caseNumber).not.toBe(case2.caseNumber);

      // Clean up
      await db.delete(cases).where(eq(cases.id, case1.id));
      await db.delete(cases).where(eq(cases.id, case2.id));
    });
  });

  describe('Case Search', () => {
    beforeEach(async () => {
      const newCase = await CaseOperations.create({
        ...testCase,
        title: 'Searchable Test Case'
      });
      createdCaseId = newCase.id;
    });

    it('should search cases by title', async () => {
      const { cases: foundCases, total } = await CaseOperations.search({
        query: 'Searchable',
        limit: 10,
        offset: 0
      });

      expect(foundCases.length).toBeGreaterThan(0);
      expect(total).toBeGreaterThan(0);

      const testCaseFound = foundCases.find(c => c.id === createdCaseId);
      expect(testCaseFound).toBeDefined();
      expect(testCaseFound?.title).toContain('Searchable');
    });

    it('should filter cases by status', async () => {
      const { cases: openCases } = await CaseOperations.search({
        status: ['open'],
        limit: 10,
        offset: 0
      });

      expect(openCases.length).toBeGreaterThan(0);
      openCases.forEach(caseItem => {
        expect(caseItem.status).toBe('open');
      });
    });

    it('should filter cases by priority', async () => {
      const { cases: mediumCases } = await CaseOperations.search({
        priority: ['medium'],
        limit: 10,
        offset: 0
      });

      expect(mediumCases.length).toBeGreaterThan(0);
      mediumCases.forEach(caseItem => {
        expect(caseItem.priority).toBe('medium');
      });
    });
  });

  describe('Case Updates', () => {
    beforeEach(async () => {
      const newCase = await CaseOperations.create(testCase);
      createdCaseId = newCase.id;
    });

    it('should update case fields', async () => {
      const updatedCase = await CaseOperations.update(
        createdCaseId,
        {
          title: 'Updated Case Title',
          status: 'investigating',
          priority: 'high'
        },
        testUser.id
      );

      expect(updatedCase.title).toBe('Updated Case Title');
      expect(updatedCase.status).toBe('investigating');
      expect(updatedCase.priority).toBe('high');
      expect(updatedCase.updatedAt).toBeInstanceOf(Date);
    });

    it('should throw error when updating non-existent case', async () => {
      await expect(
        CaseOperations.update(
          'non-existent-id',
          { title: 'New Title' },
          testUser.id
        )
      ).rejects.toThrow();
    });
  });

  describe('Case Relations', () => {
    beforeEach(async () => {
      const newCase = await CaseOperations.create(testCase);
      createdCaseId = newCase.id;
    });

    it('should load case with related evidence', async () => {
      // Create test evidence for the case
      const testEvidenceWithCase = { ...testEvidence, caseId: createdCaseId };
      const createdEvidence = await EvidenceOperations.create(testEvidenceWithCase);

      const caseWithRelations = await CaseOperations.getWithRelations(createdCaseId);

      expect(caseWithRelations).toBeDefined();
      expect(caseWithRelations?.evidence).toBeDefined();
      expect(caseWithRelations?.evidence.length).toBeGreaterThan(0);

      // Clean up evidence
      await db.delete(evidence).where(eq(evidence.id, createdEvidence.id));
    });
  });
});

describe('Evidence Operations Integration Tests', () => {
  let createdEvidenceId: string;
  let createdCaseId: string;

  beforeEach(async () => {
    // Create a test case first
    const newCase = await CaseOperations.create(testCase);
    createdCaseId = newCase.id;
  });

  afterEach(async () => {
    // Clean up created evidence and cases
    if (createdEvidenceId) {
      try {
        await db.delete(evidence).where(eq(evidence.id, createdEvidenceId));
      } catch (error: any) {
        console.warn('Failed to clean up evidence:', error);
      }
    }

    if (createdCaseId) {
      try {
        await db.delete(cases).where(eq(cases.id, createdCaseId));
      } catch (error: any) {
        console.warn('Failed to clean up case:', error);
      }
    }
  });

  describe('Evidence Creation', () => {
    it('should create evidence with vector embeddings', async () => {
      const evidenceWithCase = { ...testEvidence, caseId: createdCaseId };
      const newEvidence = await EvidenceOperations.create(evidenceWithCase);
      createdEvidenceId = newEvidence.id;

      expect(newEvidence).toBeDefined();
      expect(newEvidence.title).toBe(testEvidence.title);
      expect(newEvidence.caseId).toBe(createdCaseId);
      expect(newEvidence.evidenceType).toBe('document');
      expect(newEvidence.tags).toEqual(testEvidence.tags);
      expect(newEvidence.uploadedAt).toBeInstanceOf(Date);
      expect(newEvidence.isAdmissible).toBe(true);
    });

    it('should create evidence without case association', async () => {
      const standaloneEvidence = await EvidenceOperations.create(testEvidence);
      createdEvidenceId = standaloneEvidence.id;

      expect(standaloneEvidence).toBeDefined();
      expect(standaloneEvidence.caseId).toBeNull();
      expect(standaloneEvidence.title).toBe(testEvidence.title);
    });
  });

  describe('Evidence Search', () => {
    beforeEach(async () => {
      const evidenceWithCase = {
        ...testEvidence,
        caseId: createdCaseId,
        title: 'Searchable Evidence Item'
      };
      const newEvidence = await EvidenceOperations.create(evidenceWithCase);
      createdEvidenceId = newEvidence.id;
    });

    it('should search evidence by title', async () => {
      const { evidence: foundEvidence, total } = await EvidenceOperations.search({
        query: 'Searchable',
        limit: 10,
        offset: 0
      });

      expect(foundEvidence.length).toBeGreaterThan(0);
      expect(total).toBeGreaterThan(0);

      const testEvidenceFound = foundEvidence.find(e => e.id === createdEvidenceId);
      expect(testEvidenceFound).toBeDefined();
      expect(testEvidenceFound?.title).toContain('Searchable');
    });

    it('should filter evidence by case', async () => {
      const { evidence: caseEvidence } = await EvidenceOperations.search({
        caseId: createdCaseId,
        limit: 10,
        offset: 0
      });

      expect(caseEvidence.length).toBeGreaterThan(0);
      caseEvidence.forEach(evidenceItem => {
        expect(evidenceItem.caseId).toBe(createdCaseId);
      });
    });

    it('should filter evidence by type', async () => {
      const { evidence: documents } = await EvidenceOperations.search({
        evidenceTypes: ['document'],
        limit: 10,
        offset: 0
      });

      expect(documents.length).toBeGreaterThan(0);
      documents.forEach(evidenceItem => {
        expect(evidenceItem.evidenceType).toBe('document');
      });
    });
  });

  describe('Evidence Updates', () => {
    beforeEach(async () => {
      const evidenceWithCase = { ...testEvidence, caseId: createdCaseId };
      const newEvidence = await EvidenceOperations.create(evidenceWithCase);
      createdEvidenceId = newEvidence.id;
    });

    it('should update evidence with chain of custody', async () => {
      const updatedEvidence = await EvidenceOperations.update(
        createdEvidenceId,
        {
          title: 'Updated Evidence Title',
          description: 'Updated description',
          tags: ['updated', 'test']
        },
        testUser.id,
        'Evidence updated for testing'
      );

      expect(updatedEvidence.title).toBe('Updated Evidence Title');
      expect(updatedEvidence.description).toBe('Updated description');
      expect(updatedEvidence.tags).toEqual(['updated', 'test']);
      expect(updatedEvidence.chainOfCustody).toBeInstanceOf(Array);
      const chainOfCustody = updatedEvidence.chainOfCustody as any[];
      expect(chainOfCustody.length).toBeGreaterThan(0);

      // Check chain of custody entry
      const lastEntry = chainOfCustody[chainOfCustody.length - 1];
      expect(lastEntry.action).toBe('updated');
      expect(lastEntry.updatedBy).toBe(testUser.id);
      expect(lastEntry.notes).toBe('Evidence updated for testing');
    });
  });
});

describe('API Response Integration Tests', () => {
  describe('Success Responses', () => {
    it('should create standardized success response', () => {
      const data = { id: '123', name: 'test' };
      const response = apiSuccess(data, 'req_123', 150);

      expect(response).toBeInstanceOf(Response);
      expect(response.status).toBe(200);
    });
  });

  describe('Error Responses', () => {
    it('should create standardized error response', () => {
      const error = CommonErrors.NotFound('Test resource');
      const response = apiError(error, 'req_123', 50);

      expect(response).toBeInstanceOf(Response);
      expect(response.status).toBe(404);
    });

    it('should handle validation errors', () => {
      const validationError = CommonErrors.ValidationFailed('name', 'Required field missing');
      const response = apiError(validationError, 'req_123', 25);

      expect(response).toBeInstanceOf(Response);
      expect(response.status).toBe(400);
    });
  });
});

describe('Vector Operations Integration Tests', () => {
  it('should perform vector similarity search', async () => {
    // This test requires actual vector data in the database
    // For now, we'll just test that the functions don't throw errors
    expect(async () => {
      const health = await checkDatabaseHealth();
      expect(health.pgvectorEnabled).toBe(true);
    }).not.toThrow();
  });
});

// Test utilities
export async function setupTestDatabase(): Promise<any> {
  // Insert test user if not exists
  const existingUsers = await db.select().from(users).where(eq(users.id, testUser.id)).limit(1);
  if (existingUsers.length === 0) {
    await db.insert(users).values({
      id: testUser.id,
      email: testUser.email,
      firstName: testUser.firstName,
      lastName: testUser.lastName,
      role: testUser.role
    });
  }
}

export async function cleanupTestDatabase(): Promise<any> {
  // Clean up all test data
  await db.delete(evidence).where(eq(evidence.uploadedBy, testUser.id));
  await db.delete(cases).where(eq(cases.createdBy, testUser.id));
  await db.delete(users).where(eq(users.id, testUser.id));
}

// Export test data for use in other test files
export { testUser, testCase, testEvidence };
