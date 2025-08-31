#!/usr/bin/env node

/**
 * Comprehensive CRUD System Test for Legal AI Platform
 * Tests all user workflows, authentication, evidence upload, auto-tagging,
 * reports, citations, and complete API architecture.
 * 
 * Architecture Test Coverage:
 * - Authenticated User CRUD
 * - Cases with Evidence attachment
 * - Evidence upload with auto-tagging (pgvector)
 * - Reports and Citations CRUD
 * - JSONB PostgreSQL integration
 * - GPU acceleration pipeline
 * - Microservices (Ollama, Redis, XState, etc.)
 */

import fs from 'fs';
import path from 'path';

// Polyfill fetch for Node.js if not available
if (!global.fetch) {
  try {
    global.fetch = (await import('node-fetch')).default;
  } catch (e) {
    console.warn('node-fetch not available, using native fetch (Node.js 18+)');
  }
}

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const TEST_USER = {
  email: 'test@legalai.com',
  password: 'TestPass123',
  firstName: 'Test',
  lastName: 'User',
  role: 'attorney'
};

class LegalAISystemTest {
  constructor() {
    this.sessionToken = null;
    this.testResults = {
      passed: 0,
      failed: 0,
      details: []
    };
    this.testUser = null;
    this.testCase = null;
    this.testEvidence = [];
    this.testReports = [];
    this.testCitations = [];
  }

  async log(message, level = 'INFO') {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level}] ${message}`);
    this.testResults.details.push({ timestamp, level, message });
  }

  async makeRequest(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...(this.sessionToken && { 'Authorization': `Bearer ${this.sessionToken}` })
      }
    };

    const finalOptions = {
      ...defaultOptions,
      ...options,
      headers: { ...defaultOptions.headers, ...options.headers }
    };

    try {
      const response = await fetch(url, finalOptions);
      const data = await response.text();
      
      let parsedData;
      try {
        parsedData = JSON.parse(data);
      } catch {
        parsedData = { raw: data, status: response.status };
      }

      return {
        ok: response.ok,
        status: response.status,
        data: parsedData
      };
    } catch (error) {
      await this.log(`Request failed: ${url} - ${error.message}`, 'ERROR');
      return {
        ok: false,
        status: 0,
        data: { error: error.message }
      };
    }
  }

  async assert(condition, testName, details = '') {
    if (condition) {
      this.testResults.passed++;
      await this.log(`✅ PASS: ${testName}${details ? ' - ' + details : ''}`, 'PASS');
      return true;
    } else {
      this.testResults.failed++;
      await this.log(`❌ FAIL: ${testName}${details ? ' - ' + details : ''}`, 'FAIL');
      return false;
    }
  }

  // Test 1: Authentication System
  async testAuthentication() {
    await this.log('=== Testing Authentication System ===');

    // Register new user
    const registerResponse = await this.makeRequest('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(TEST_USER)
    });

    await this.assert(
      registerResponse.ok || registerResponse.status === 409, // 409 if user exists
      'User Registration',
      `Status: ${registerResponse.status}`
    );

    // Login
    const loginResponse = await this.makeRequest('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        email: TEST_USER.email,
        password: TEST_USER.password
      })
    });

    await this.assert(
      loginResponse.ok,
      'User Login',
      `Status: ${loginResponse.status}`
    );

    if (loginResponse.ok && loginResponse.data.sessionToken) {
      this.sessionToken = loginResponse.data.sessionToken;
    } else if (loginResponse.ok && loginResponse.data.token) {
      this.sessionToken = loginResponse.data.token;
    }

    // Get user profile
    const profileResponse = await this.makeRequest('/api/auth/me', {
      method: 'GET'
    });

    await this.assert(
      profileResponse.ok,
      'User Profile Retrieval',
      `Status: ${profileResponse.status}`
    );

    if (profileResponse.ok) {
      this.testUser = profileResponse.data.user || profileResponse.data;
    }

    return true;
  }

  // Test 2: Cases CRUD Operations
  async testCasesCRUD() {
    await this.log('=== Testing Cases CRUD Operations ===');

    const testCaseData = {
      title: 'Test Legal Case - System Integration',
      description: 'Comprehensive test case for system validation',
      caseNumber: `TEST-${Date.now()}`,
      status: 'active',
      priority: 'high',
      assignedTo: this.testUser?.id || 'test-user',
      isConfidential: false,
      tags: ['test', 'integration', 'system-validation'],
      metadata: {
        jurisdiction: 'Federal',
        courtLevel: 'district',
        practiceArea: ['criminal law', 'constitutional law']
      }
    };

    // Create Case
    const createResponse = await this.makeRequest('/api/cases', {
      method: 'POST',
      body: JSON.stringify(testCaseData)
    });

    await this.assert(
      createResponse.ok,
      'Case Creation',
      `Status: ${createResponse.status}`
    );

    if (createResponse.ok) {
      this.testCase = createResponse.data.case || createResponse.data;
    }

    // Read Cases
    const readResponse = await this.makeRequest('/api/cases', {
      method: 'GET'
    });

    await this.assert(
      readResponse.ok,
      'Cases Retrieval',
      `Status: ${readResponse.status}`
    );

    // Update Case
    if (this.testCase?.id) {
      const updateResponse = await this.makeRequest(`/api/cases/${this.testCase.id}`, {
        method: 'PUT',
        body: JSON.stringify({
          ...testCaseData,
          description: 'Updated test case description',
          status: 'investigating'
        })
      });

      await this.assert(
        updateResponse.ok,
        'Case Update',
        `Status: ${updateResponse.status}`
      );
    }

    return true;
  }

  // Test 3: Evidence Upload and Management
  async testEvidenceOperations() {
    await this.log('=== Testing Evidence Operations ===');

    if (!this.testCase?.id) {
      await this.log('No test case available, skipping evidence tests', 'WARN');
      return false;
    }

    const testEvidenceData = {
      caseId: this.testCase.id,
      title: 'Test Evidence Document',
      description: 'System integration test evidence',
      evidenceType: 'document',
      category: 'physical',
      status: 'authenticated',
      metadata: {
        fileType: 'text',
        source: 'system_test',
        confidentialityLevel: 'public',
        chainOfCustody: [`Test upload at ${new Date().toISOString()}`]
      },
      content: 'This is a test evidence document for system validation. It contains sample legal text for AI processing and vector indexing.'
    };

    // Create Evidence
    const createResponse = await this.makeRequest('/api/evidence', {
      method: 'POST',
      body: JSON.stringify(testEvidenceData)
    });

    await this.assert(
      createResponse.ok,
      'Evidence Creation',
      `Status: ${createResponse.status}`
    );

    if (createResponse.ok) {
      const evidence = createResponse.data.evidence || createResponse.data;
      this.testEvidence.push(evidence);
    }

    // Test Evidence Upload with Auto-tagging
    const uploadData = new FormData();
    uploadData.append('file', new Blob(['Test evidence content for AI analysis'], { type: 'text/plain' }), 'test-evidence.txt');
    uploadData.append('caseId', this.testCase.id);
    uploadData.append('title', 'Auto-tagged Test Evidence');
    uploadData.append('description', 'Evidence for testing auto-tagging system');

    const uploadResponse = await this.makeRequest('/api/evidence/upload', {
      method: 'POST',
      body: uploadData,
      headers: {} // Remove Content-Type to let browser set it
    });

    await this.assert(
      uploadResponse.ok,
      'Evidence Upload with Auto-tagging',
      `Status: ${uploadResponse.status}`
    );

    // Test AI Processing Trigger
    if (this.testEvidence.length > 0) {
      const processResponse = await this.makeRequest('/api/ai/process-evidence', {
        method: 'POST',
        body: JSON.stringify({
          evidenceId: this.testEvidence[0].id,
          analysisType: 'full',
          useGPUAcceleration: true
        })
      });

      await this.assert(
        processResponse.ok,
        'AI Evidence Processing',
        `Status: ${processResponse.status}`
      );
    }

    return true;
  }

  // Test 4: Reports CRUD Operations
  async testReportsCRUD() {
    await this.log('=== Testing Reports CRUD Operations ===');

    const testReportData = {
      title: 'System Integration Test Report',
      content: 'This is a comprehensive test report generated during system validation.',
      reportType: 'case_analysis',
      status: 'draft',
      caseId: this.testCase?.id,
      metadata: {
        generatedBy: 'system_test',
        analysisType: 'integration_test',
        timestamp: new Date().toISOString(),
        sections: ['summary', 'evidence_analysis', 'recommendations']
      }
    };

    // Create Report
    const createResponse = await this.makeRequest('/api/reports', {
      method: 'POST',
      body: JSON.stringify(testReportData)
    });

    await this.assert(
      createResponse.ok,
      'Report Creation',
      `Status: ${createResponse.status}`
    );

    if (createResponse.ok) {
      const report = createResponse.data.report || createResponse.data;
      this.testReports.push(report);
    }

    // Read Reports
    const readResponse = await this.makeRequest('/api/reports', {
      method: 'GET'
    });

    await this.assert(
      readResponse.ok,
      'Reports Retrieval',
      `Status: ${readResponse.status}`
    );

    // Update Report
    if (this.testReports.length > 0) {
      const updateResponse = await this.makeRequest(`/api/reports/${this.testReports[0].id}`, {
        method: 'PUT',
        body: JSON.stringify({
          ...testReportData,
          content: 'Updated report content with additional findings',
          status: 'completed'
        })
      });

      await this.assert(
        updateResponse.ok,
        'Report Update',
        `Status: ${updateResponse.status}`
      );
    }

    return true;
  }

  // Test 5: Citations CRUD Operations
  async testCitationsCRUD() {
    await this.log('=== Testing Citations CRUD Operations ===');

    const testCitationData = {
      title: 'Miranda v. Arizona Test Citation',
      content: 'Test citation for system validation - Fifth Amendment protection against self-incrimination.',
      author: 'U.S. Supreme Court',
      source: '384 U.S. 436 (1966)',
      citationType: 'case_law',
      jurisdiction: 'federal',
      tags: ['constitutional_law', 'criminal_procedure', 'miranda_rights'],
      metadata: {
        court: 'Supreme Court',
        year: 1966,
        relevanceScore: 0.95,
        practiceArea: ['criminal law']
      }
    };

    // Create Citation
    const createResponse = await this.makeRequest('/api/citations', {
      method: 'POST',
      body: JSON.stringify(testCitationData)
    });

    await this.assert(
      createResponse.ok,
      'Citation Creation',
      `Status: ${createResponse.status}`
    );

    if (createResponse.ok) {
      const citation = createResponse.data.citation || createResponse.data;
      this.testCitations.push(citation);
    }

    // Read Citations
    const readResponse = await this.makeRequest('/api/citations', {
      method: 'GET'
    });

    await this.assert(
      readResponse.ok,
      'Citations Retrieval',
      `Status: ${readResponse.status}`
    );

    return true;
  }

  // Test 6: Vector Search and AI Integration
  async testVectorSearchAI() {
    await this.log('=== Testing Vector Search and AI Integration ===');

    // Test semantic search
    const searchResponse = await this.makeRequest('/api/ai/vector-search', {
      method: 'POST',
      body: JSON.stringify({
        query: 'constitutional rights and criminal procedure',
        limit: 10,
        threshold: 0.3,
        filters: {
          documentType: 'evidence',
          caseId: this.testCase?.id
        }
      })
    });

    await this.assert(
      searchResponse.ok,
      'Vector Search',
      `Status: ${searchResponse.status}`
    );

    // Test AI Chat
    const chatResponse = await this.makeRequest('/api/ai/chat', {
      method: 'POST',
      body: JSON.stringify({
        messages: [
          {
            role: 'user',
            content: 'Analyze the legal implications of Miranda rights in criminal cases'
          }
        ],
        context: {
          caseId: this.testCase?.id,
          useRAG: true,
          useGPU: true
        }
      })
    });

    await this.assert(
      chatResponse.ok,
      'AI Chat Integration',
      `Status: ${chatResponse.status}`
    );

    // Test Ollama Health
    const ollamaResponse = await this.makeRequest('/api/ai/health/local', {
      method: 'GET'
    });

    await this.assert(
      ollamaResponse.ok,
      'Ollama Service Health',
      `Status: ${ollamaResponse.status}`
    );

    return true;
  }

  // Test 7: GPU Acceleration and Microservices
  async testGPUAndMicroservices() {
    await this.log('=== Testing GPU Acceleration and Microservices ===');

    // Test GPU acceleration
    const gpuResponse = await this.makeRequest('/api/ai/gpu', {
      method: 'POST',
      body: JSON.stringify({
        operation: 'inference',
        model: 'gemma3-legal',
        input: 'Test GPU acceleration for legal AI processing',
        options: {
          useWebGPU: true,
          useCUDA: true,
          optimization: 'speed'
        }
      })
    });

    await this.assert(
      gpuResponse.ok,
      'GPU Acceleration Test',
      `Status: ${gpuResponse.status}`
    );

    // Test Redis integration
    const redisResponse = await this.makeRequest('/api/v1/redis/publish', {
      method: 'POST',
      body: JSON.stringify({
        channel: 'test_channel',
        message: { test: 'system_integration', timestamp: Date.now() }
      })
    });

    await this.assert(
      redisResponse.ok,
      'Redis Integration',
      `Status: ${redisResponse.status}`
    );

    // Test Worker processing
    const workerResponse = await this.makeRequest('/api/worker/autotag/trigger', {
      method: 'POST',
      body: JSON.stringify({
        caseId: this.testCase?.id,
        priority: 'high',
        operation: 'full_analysis'
      })
    });

    await this.assert(
      workerResponse.ok,
      'Worker Processing',
      `Status: ${workerResponse.status}`
    );

    return true;
  }

  // Test 8: Database Architecture and JSONB
  async testDatabaseArchitecture() {
    await this.log('=== Testing Database Architecture and JSONB ===');

    // Test JSONB legal metadata
    const jsonbResponse = await this.makeRequest('/api/jsonb/legal', {
      method: 'POST',
      body: JSON.stringify({
        operation: 'search',
        criteria: {
          'metadata.practiceArea': ['criminal law'],
          'metadata.jurisdiction': 'federal'
        },
        limit: 10
      })
    });

    await this.assert(
      jsonbResponse.ok,
      'JSONB Legal Metadata Query',
      `Status: ${jsonbResponse.status}`
    );

    // Test database health
    const healthResponse = await this.makeRequest('/api/admin/health', {
      method: 'GET'
    });

    await this.assert(
      healthResponse.ok,
      'Database Health Check',
      `Status: ${healthResponse.status}`
    );

    return true;
  }

  // Main test runner
  async runAllTests() {
    await this.log('🚀 Starting Comprehensive Legal AI System Test');
    await this.log(`Testing against: ${BASE_URL}`);

    const tests = [
      () => this.testAuthentication(),
      () => this.testCasesCRUD(),
      () => this.testEvidenceOperations(),
      () => this.testReportsCRUD(),
      () => this.testCitationsCRUD(),
      () => this.testVectorSearchAI(),
      () => this.testGPUAndMicroservices(),
      () => this.testDatabaseArchitecture()
    ];

    for (const test of tests) {
      try {
        await test();
      } catch (error) {
        await this.log(`Test error: ${error.message}`, 'ERROR');
        this.testResults.failed++;
      }
      await new Promise(resolve => setTimeout(resolve, 1000)); // Brief pause between tests
    }

    await this.generateReport();
  }

  async generateReport() {
    const total = this.testResults.passed + this.testResults.failed;
    const successRate = total > 0 ? (this.testResults.passed / total * 100).toFixed(2) : 0;

    const report = {
      summary: {
        total,
        passed: this.testResults.passed,
        failed: this.testResults.failed,
        successRate: `${successRate}%`,
        timestamp: new Date().toISOString(),
        testDuration: Date.now() - this.startTime
      },
      testUser: this.testUser,
      testCase: this.testCase,
      testEvidence: this.testEvidence.length,
      testReports: this.testReports.length,
      testCitations: this.testCitations.length,
      details: this.testResults.details
    };

    const reportPath = `test-report-${Date.now()}.json`;
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    await this.log('📊 Test Summary:');
    await this.log(`Total Tests: ${total}`);
    await this.log(`Passed: ${this.testResults.passed}`);
    await this.log(`Failed: ${this.testResults.failed}`);
    await this.log(`Success Rate: ${successRate}%`);
    await this.log(`Report saved: ${reportPath}`);

    if (this.testResults.failed === 0) {
      await this.log('🎉 ALL TESTS PASSED! Legal AI System is fully operational.');
    } else {
      await this.log('⚠️  Some tests failed. Check the report for details.');
    }

    return report;
  }
}

// Run tests
async function main() {
  const tester = new LegalAISystemTest();
  tester.startTime = Date.now();
  
  try {
    await tester.runAllTests();
    process.exit(tester.testResults.failed > 0 ? 1 : 0);
  } catch (error) {
    console.error('Fatal test error:', error);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export default LegalAISystemTest;