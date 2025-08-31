import { test, expect } from '@playwright/test';

test.describe('Legal AI System - Complete E2E Tests', () => {
  const API_URL = 'http://localhost:5173/api';
  let testCaseId: string;
  let testEvidenceId: string;
  let testReportId: string;
  let testCitationId: string;

  test.beforeAll(async ({ request }) => {
    // Check system health before running tests
    const health = await request.get(`${API_URL}/health`);
    expect(health.ok()).toBeTruthy();
    const healthData = await health.json();
    console.log('System health:', healthData.status);
  });

  test.describe('Cases Management', () => {
    test('should create a new legal case', async ({ request }) => {
      const response = await request.post(`${API_URL}/cases`, {
        data: {
          caseNumber: `TEST-${Date.now()}`,
          title: 'Playwright Test Case - Contract Dispute',
          clientName: 'Test Client Corp',
          opposingParty: 'Defendant Inc',
          caseType: 'civil',
          practiceArea: 'corporate litigation',
          priority: 'high',
          status: 'active',
          jurisdiction: 'federal',
          courtName: 'U.S. District Court',
          description: 'Test case for automated testing'
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data).toHaveProperty('id');
      expect(body.data.caseNumber).toContain('TEST-');
      testCaseId = body.data.id;
      console.log('Created case:', testCaseId);
    });

    test('should retrieve the created case', async ({ request }) => {
      const response = await request.get(`${API_URL}/cases?id=${testCaseId}`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data.id).toBe(testCaseId);
      expect(body.data.priority).toBe('high');
    });

    test('should list all cases with pagination', async ({ request }) => {
      const response = await request.get(`${API_URL}/cases?limit=10&offset=0`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(Array.isArray(body.data)).toBe(true);
    });

    test('should update case status', async ({ request }) => {
      const response = await request.put(`${API_URL}/cases?id=${testCaseId}`, {
        data: {
          status: 'in_progress',
          priority: 'critical'
        }
      });
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
    });
  });

  test.describe('Evidence Management', () => {
    test('should upload evidence file', async ({ request }) => {
      const formData = {
        file: {
          name: 'test-document.pdf',
          mimeType: 'application/pdf',
          buffer: Buffer.from('Test PDF content for evidence')
        },
        caseId: testCaseId,
        title: 'Contract Agreement',
        description: 'Original contract between parties'
      };

      const response = await request.post(`${API_URL}/evidence`, {
        multipart: formData
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data).toHaveProperty('evidence_number');
      expect(body.data.processing_status).toBe('pending');
      testEvidenceId = body.data.id;
      console.log('Uploaded evidence:', body.data.evidence_number);
    });

    test('should retrieve evidence for case', async ({ request }) => {
      const response = await request.get(`${API_URL}/evidence?caseId=${testCaseId}`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(Array.isArray(body.data)).toBe(true);
      expect(body.data.length).toBeGreaterThan(0);
    });

    test('should check evidence processing queue', async ({ request }) => {
      const response = await request.get(`${API_URL}/ingestion?resourceId=${testEvidenceId}`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
    });
  });

  test.describe('Reports Generation', () => {
    test('should create manual report', async ({ request }) => {
      const response = await request.post(`${API_URL}/reports`, {
        data: {
          title: 'Case Progress Report',
          type: 'progress_report',
          content: '# Case Progress\n\n## Summary\nCase is progressing well...',
          caseId: testCaseId,
          format: 'markdown'
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data).toHaveProperty('id');
      testReportId = body.data.id;
    });

    test('should generate AI report', async ({ request }) => {
      const response = await request.post(`${API_URL}/reports`, {
        data: {
          title: 'AI Legal Analysis',
          type: 'legal_research',
          caseId: testCaseId,
          format: 'markdown',
          aiGenerated: true,
          aiPrompt: 'Analyze contract dispute case and provide legal recommendations'
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data.ai_generated).toBe(true);
    });

    test('should retrieve reports for case', async ({ request }) => {
      const response = await request.get(`${API_URL}/reports?caseId=${testCaseId}`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(Array.isArray(body.data)).toBe(true);
      expect(body.data.length).toBeGreaterThanOrEqual(2);
    });
  });

  test.describe('Citations Management', () => {
    test('should add citation', async ({ request }) => {
      const response = await request.post(`${API_URL}/citations`, {
        data: {
          caseId: testCaseId,
          reportId: testReportId,
          caseNumber: '567 F.3d 890',
          caseName: 'Smith v. Jones Corp',
          citation: '567 F.3d 890 (9th Cir. 2022)',
          courtName: '9th Circuit Court of Appeals',
          decisionDate: '2022-06-15',
          precedentLevel: 'binding',
          legalIssues: ['contract breach', 'damages'],
          holding: 'Court held that consequential damages are recoverable',
          reasoning: 'Based on established precedent and contract terms'
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.data).toHaveProperty('relevance_score');
      testCitationId = body.data.id;
    });

    test('should search citations with vector similarity', async ({ request }) => {
      const response = await request.get(
        `${API_URL}/citations?search=contract damages breach`
      );
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(Array.isArray(body.data)).toBe(true);
    });
  });

  test.describe('Vector Search', () => {
    test('should perform cross-collection vector search', async ({ request }) => {
      const response = await request.post(`${API_URL}/search/vector`, {
        data: {
          query: 'contract breach damages legal precedent',
          collections: ['cases', 'evidence', 'reports', 'citations'],
          limit: 10
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.results).toBeDefined();
      expect(body.sources).toBeDefined();
      expect(body.sources.pgvector).toBeGreaterThanOrEqual(0);
    });

    test('should filter vector search by case', async ({ request }) => {
      const response = await request.post(`${API_URL}/search/vector`, {
        data: {
          query: 'evidence contract',
          collections: ['evidence', 'reports'],
          limit: 5,
          filters: {
            caseId: testCaseId
          }
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
    });
  });

  test.describe('Processing Queue', () => {
    test('should queue multiple processing operations', async ({ request }) => {
      const response = await request.post(`${API_URL}/ingestion`, {
        data: {
          resourceType: 'evidence',
          resourceId: testEvidenceId,
          operations: ['ocr', 'embedding', 'analysis', 'auto-tagging']
        }
      });

      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(body.taskIds).toBeDefined();
      expect(body.taskIds.length).toBe(4);
    });

    test('should check queue status', async ({ request }) => {
      const response = await request.get(`${API_URL}/ingestion?status=queued`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(body.success).toBe(true);
      expect(Array.isArray(body.data)).toBe(true);
    });
  });

  test.describe('System Health', () => {
    test('should verify all services are healthy', async ({ request }) => {
      const response = await request.get(`${API_URL}/health`);
      
      expect(response.ok()).toBeTruthy();
      const body = await response.json();
      expect(['healthy', 'degraded']).toContain(body.status);
      expect(body.checks.database).toBe(true);
    });
  });

  test.describe('Cleanup', () => {
    test('should delete test data', async ({ request }) => {
      // Delete citation
      if (testCitationId) {
        const citationResponse = await request.delete(
          `${API_URL}/citations?id=${testCitationId}`
        );
        expect(citationResponse.ok()).toBeTruthy();
      }

      // Delete report
      if (testReportId) {
        const reportResponse = await request.delete(
          `${API_URL}/reports?id=${testReportId}`
        );
        expect(reportResponse.ok()).toBeTruthy();
      }

      // Delete case (will cascade delete evidence)
      if (testCaseId) {
        const caseResponse = await request.delete(
          `${API_URL}/cases?id=${testCaseId}`
        );
        expect(caseResponse.ok()).toBeTruthy();
      }

      console.log('Test cleanup completed');
    });
  });
});

test.describe('Performance Tests', () => {
  test('should handle concurrent requests', async ({ request }) => {
    const promises = Array.from({ length: 5 }, (_, i) => 
      request.post('http://localhost:5173/api/cases', {
        data: {
          caseNumber: `PERF-${Date.now()}-${i}`,
          title: `Performance Test ${i}`,
          clientName: `Client ${i}`,
          caseType: 'civil',
          practiceArea: 'test'
        }
      })
    );
    
    const responses = await Promise.all(promises);
    responses.forEach(response => {
      expect(response.ok()).toBeTruthy();
    });
  });

  test('should complete vector search within 3 seconds', async ({ request }) => {
    const startTime = Date.now();
    
    const response = await request.post('http://localhost:5173/api/search/vector', {
      data: {
        query: 'complex legal analysis with multiple terms',
        collections: ['cases', 'evidence', 'reports', 'citations'],
        limit: 20
      }
    });
    
    const duration = Date.now() - startTime;
    
    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(3000);
    console.log(`Vector search completed in ${duration}ms`);
  });
});