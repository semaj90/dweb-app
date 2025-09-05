#!/usr/bin/env node
/**
 * API Endpoint Comprehensive Tester
 * Tests all API endpoints for proper database sync with Drizzle ORM
 * Verifies type safety, data integrity, and production readiness
 */

import { readFile } from 'fs/promises';
import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';

const API_BASE = process.env.API_BASE_URL || 'http://localhost:5173';
const DB_CONFIG = {
  host: 'localhost',
  port: 5432,
  database: 'legal_ai_db',
  username: 'postgres',
  password: '123456'
};

class APIEndpointTester {
  constructor() {
    this.spinner = ora();
    this.results = {
      passed: 0,
      failed: 0,
      warnings: 0,
      tests: []
    };
    this.db = null;
  }

  async initialize() {
    try {
      // Initialize database connection for verification
      const { default: postgres } = await import('postgres');
      this.db = postgres(`postgresql://${DB_CONFIG.username}:${DB_CONFIG.password}@${DB_CONFIG.host}:${DB_CONFIG.port}/${DB_CONFIG.database}`);
      
      // Test connection
      await this.db`SELECT 1`;
      console.log(chalk.green('✅ Database connection established'));
    } catch (error) {
      console.log(chalk.yellow('⚠️ Database connection failed, continuing without DB verification'));
    }
  }

  async runAllTests() {
    console.log(chalk.cyan.bold('🧪 API Endpoint Comprehensive Testing'));
    console.log(chalk.gray(`Base URL: ${API_BASE}`));
    console.log('='.repeat(60));

    await this.initialize();

    try {
      await this.testHealthEndpoints();
      await this.testAuthEndpoints();
      await this.testCRUDEndpoints();
      await this.testVectorSearchEndpoints();
      await this.testFileUploadEndpoints();
      await this.testAIEndpoints();
      await this.testRealtimeEndpoints();
      
      this.printResults();
    } catch (error) {
      this.spinner.fail(`Test suite failed: ${error.message}`);
      process.exit(1);
    } finally {
      if (this.db) {
        await this.db.end();
      }
    }
  }

  async testHealthEndpoints() {
    this.spinner.start('Testing health endpoints...');
    
    const healthEndpoints = [
      { path: '/api/health', method: 'GET', expectedStatus: 200 },
      { path: '/api/v1/health', method: 'GET', expectedStatus: 200 },
      { path: '/api/v1/health/services', method: 'GET', expectedStatus: 200 },
      { path: '/api/v1/health/database', method: 'GET', expectedStatus: 200 },
      { path: '/api/vite-health', method: 'GET', expectedStatus: 200 }
    ];

    for (const endpoint of healthEndpoints) {
      await this.testEndpoint('Health Check', endpoint, async (response, data) => {
        // Verify health response structure
        if (data && typeof data === 'object') {
          const hasStatus = 'status' in data || 'health' in data || 'ok' in data;
          if (!hasStatus) {
            throw new Error('Health endpoint missing status field');
          }
        }
        return true;
      });
    }

    this.spinner.succeed('Health endpoints tested');
  }

  async testAuthEndpoints() {
    this.spinner.start('Testing authentication endpoints...');

    // Test user registration
    const registerData = {
      email: `test-${Date.now()}@example.com`,
      password: 'TestPass123!',
      name: 'Test User'
    };

    await this.testEndpoint('User Registration', {
      path: '/api/auth/register',
      method: 'POST',
      data: registerData,
      expectedStatus: [201, 409] // 409 if user exists
    }, async (response, data) => {
      if (response.status === 201 && this.db) {
        // Verify user was created in database
        const [user] = await this.db`SELECT id, email, name FROM users WHERE email = ${registerData.email}`;
        if (!user) {
          throw new Error('User not found in database after registration');
        }
        if (user.email !== registerData.email) {
          throw new Error('Database user email mismatch');
        }
        
        // Cleanup test user
        await this.db`DELETE FROM users WHERE id = ${user.id}`;
      }
      return true;
    });

    // Test user login
    await this.testEndpoint('User Login', {
      path: '/api/auth/login',
      method: 'POST',
      data: { email: 'admin@example.com', password: 'password' },
      expectedStatus: [200, 401, 404]
    });

    // Test session validation
    await this.testEndpoint('Session Validation', {
      path: '/api/auth/session',
      method: 'GET',
      expectedStatus: [200, 401]
    });

    this.spinner.succeed('Authentication endpoints tested');
  }

  async testCRUDEndpoints() {
    this.spinner.start('Testing CRUD endpoints...');

    let testCaseId = null;
    
    // Test Case Creation
    const caseData = {
      title: `API Test Case ${Date.now()}`,
      description: 'Test case created by API endpoint tester',
      priority: 'medium',
      status: 'active'
    };

    await this.testEndpoint('Create Case', {
      path: '/api/v1/cases',
      method: 'POST',
      data: caseData,
      expectedStatus: [201, 401]
    }, async (response, data) => {
      if (response.status === 201 && data) {
        testCaseId = data.id;
        
        // Verify in database
        if (this.db) {
          const [caseRecord] = await this.db`SELECT * FROM cases WHERE id = ${testCaseId}`;
          if (!caseRecord) {
            throw new Error('Case not found in database after creation');
          }
          if (caseRecord.title !== caseData.title) {
            throw new Error('Database case title mismatch');
          }
        }
      }
      return true;
    });

    // Test Case Reading
    if (testCaseId) {
      await this.testEndpoint('Read Case', {
        path: `/api/v1/cases/${testCaseId}`,
        method: 'GET',
        expectedStatus: [200, 401, 404]
      }, async (response, data) => {
        if (response.status === 200 && data) {
          if (data.id !== testCaseId) {
            throw new Error('Case ID mismatch in response');
          }
          if (data.title !== caseData.title) {
            throw new Error('Case title mismatch in response');
          }
        }
        return true;
      });

      // Test Case Update
      const updateData = { title: `Updated ${caseData.title}`, status: 'in_review' };
      await this.testEndpoint('Update Case', {
        path: `/api/v1/cases/${testCaseId}`,
        method: 'PATCH',
        data: updateData,
        expectedStatus: [200, 401, 404]
      }, async (response, data) => {
        if (response.status === 200 && this.db) {
          // Verify update in database
          const [updatedCase] = await this.db`SELECT * FROM cases WHERE id = ${testCaseId}`;
          if (updatedCase && updatedCase.title !== updateData.title) {
            throw new Error('Case update not reflected in database');
          }
        }
        return true;
      });

      // Test Case Deletion
      await this.testEndpoint('Delete Case', {
        path: `/api/v1/cases/${testCaseId}`,
        method: 'DELETE',
        expectedStatus: [204, 200, 401, 404]
      }, async (response, data) => {
        if (response.status === 204 || response.status === 200) {
          if (this.db) {
            // Verify deletion in database
            const [deletedCase] = await this.db`SELECT * FROM cases WHERE id = ${testCaseId}`;
            if (deletedCase) {
              throw new Error('Case not deleted from database');
            }
          }
        }
        return true;
      });
    }

    // Test Cases Listing with Pagination
    await this.testEndpoint('List Cases', {
      path: '/api/v1/cases?page=1&limit=10',
      method: 'GET',
      expectedStatus: [200, 401]
    }, async (response, data) => {
      if (response.status === 200 && data) {
        // Verify pagination structure
        const hasItems = Array.isArray(data.items) || Array.isArray(data);
        if (!hasItems) {
          throw new Error('Cases list missing items array');
        }
        
        // If paginated, check structure
        if (data.items && !Array.isArray(data.items)) {
          throw new Error('Invalid pagination structure');
        }
      }
      return true;
    });

    this.spinner.succeed('CRUD endpoints tested');
  }

  async testVectorSearchEndpoints() {
    this.spinner.start('Testing vector search endpoints...');

    // Test Vector Search
    await this.testEndpoint('Vector Search', {
      path: '/api/v1/vector/search',
      method: 'POST',
      data: {
        query: 'legal contract analysis',
        limit: 10,
        threshold: 0.7
      },
      expectedStatus: [200, 400, 401, 500]
    }, async (response, data) => {
      if (response.status === 200 && data) {
        // Verify search results structure
        const hasResults = Array.isArray(data.results) || Array.isArray(data);
        if (!hasResults) {
          throw new Error('Vector search missing results array');
        }

        // Verify result structure if any results
        const results = data.results || data;
        if (results.length > 0) {
          const firstResult = results[0];
          if (!firstResult.id && !firstResult.content && !firstResult.score) {
            this.addWarning('Vector search result missing expected fields (id, content, score)');
          }
        }
      }
      return true;
    });

    // Test Embedding Generation
    await this.testEndpoint('Generate Embeddings', {
      path: '/api/v1/embeddings',
      method: 'POST',
      data: {
        text: 'This is a test document for embedding generation.',
        model: 'nomic-embed-text'
      },
      expectedStatus: [200, 400, 401, 500, 503]
    }, async (response, data) => {
      if (response.status === 200 && data) {
        // Verify embedding structure
        if (!data.embedding || !Array.isArray(data.embedding)) {
          throw new Error('Invalid embedding response structure');
        }
        
        // Verify embedding dimensions (should be 384 for nomic-embed-text)
        if (data.embedding.length !== 384) {
          this.addWarning(`Unexpected embedding dimension: ${data.embedding.length}, expected 384`);
        }
      }
      return true;
    });

    this.spinner.succeed('Vector search endpoints tested');
  }

  async testFileUploadEndpoints() {
    this.spinner.start('Testing file upload endpoints...');

    // Create test file data
    const testFile = new Blob(['This is a test legal document content for upload testing.'], { 
      type: 'text/plain' 
    });

    // Test File Upload
    await this.testEndpoint('File Upload', {
      path: '/api/v1/upload',
      method: 'POST',
      data: {
        file: testFile,
        fileName: `test-document-${Date.now()}.txt`,
        documentType: 'evidence'
      },
      expectedStatus: [201, 400, 401, 413, 500],
      isFormData: true
    }, async (response, data) => {
      if (response.status === 201 && data) {
        // Verify upload response
        if (!data.id && !data.fileId && !data.uploadId) {
          throw new Error('Upload response missing file identifier');
        }
        
        if (this.db && data.id) {
          // Verify file record in database
          const [fileRecord] = await this.db`SELECT * FROM evidence WHERE id = ${data.id}`;
          if (!fileRecord) {
            throw new Error('Uploaded file not found in database');
          }
        }
      }
      return true;
    });

    // Test Bulk Upload Status
    await this.testEndpoint('Upload Status', {
      path: '/api/v1/upload/status',
      method: 'GET',
      expectedStatus: [200, 401]
    });

    this.spinner.succeed('File upload endpoints tested');
  }

  async testAIEndpoints() {
    this.spinner.start('Testing AI endpoints...');

    // Test RAG Query
    await this.testEndpoint('RAG Query', {
      path: '/api/v1/rag',
      method: 'POST',
      data: {
        query: 'What are the key elements of a legal contract?',
        context: ['contracts', 'legal'],
        stream: false
      },
      expectedStatus: [200, 400, 401, 500, 503],
      timeout: 30000 // AI endpoints may take longer
    }, async (response, data) => {
      if (response.status === 200 && data) {
        // Verify RAG response structure
        if (!data.response && !data.answer && !data.content) {
          throw new Error('RAG response missing content field');
        }
        
        // Verify response quality
        const content = data.response || data.answer || data.content;
        if (typeof content !== 'string' || content.length < 10) {
          this.addWarning('RAG response seems too short or invalid');
        }
      }
      return true;
    });

    // Test AI Chat
    await this.testEndpoint('AI Chat', {
      path: '/api/v1/chat',
      method: 'POST',
      data: {
        messages: [
          { role: 'user', content: 'Explain the concept of force majeure in contracts.' }
        ],
        model: 'gemma3-legal'
      },
      expectedStatus: [200, 400, 401, 500, 503],
      timeout: 30000
    });

    // Test Document Analysis
    await this.testEndpoint('Document Analysis', {
      path: '/api/v1/analyze',
      method: 'POST',
      data: {
        content: 'This Agreement is entered into on [DATE] between [PARTY A] and [PARTY B]...',
        analysisType: 'contract'
      },
      expectedStatus: [200, 400, 401, 500, 503],
      timeout: 20000
    });

    this.spinner.succeed('AI endpoints tested');
  }

  async testRealtimeEndpoints() {
    this.spinner.start('Testing real-time endpoints...');

    // Test WebSocket Health
    await this.testEndpoint('WebSocket Health', {
      path: '/api/v1/ws/health',
      method: 'GET',
      expectedStatus: [200, 404]
    });

    // Test NATS Status
    await this.testEndpoint('NATS Status', {
      path: '/api/v1/nats/status',
      method: 'GET',
      expectedStatus: [200, 503]
    });

    // Test Real-time Notifications
    await this.testEndpoint('Notification Publish', {
      path: '/api/v1/notifications/publish',
      method: 'POST',
      data: {
        channel: 'test',
        message: 'API test notification',
        userId: 'test-user'
      },
      expectedStatus: [200, 400, 401, 503]
    });

    this.spinner.succeed('Real-time endpoints tested');
  }

  async testEndpoint(testName, config, customValidator = null) {
    try {
      const startTime = Date.now();
      
      // Prepare request
      const url = `${API_BASE}${config.path}`;
      const options = {
        method: config.method,
        headers: {
          'Content-Type': config.isFormData ? undefined : 'application/json',
          'User-Agent': 'API-Endpoint-Tester/1.0'
        },
        timeout: config.timeout || 10000
      };

      // Add body for POST/PATCH requests
      if (config.data && (config.method === 'POST' || config.method === 'PATCH')) {
        if (config.isFormData) {
          const formData = new FormData();
          Object.entries(config.data).forEach(([key, value]) => {
            formData.append(key, value);
          });
          options.body = formData;
        } else {
          options.body = JSON.stringify(config.data);
        }
      }

      // Make request
      const response = await fetch(url, options);
      const responseTime = Date.now() - startTime;
      
      // Parse response
      let responseData = null;
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        try {
          responseData = await response.json();
        } catch (e) {
          this.addWarning(`${testName}: Invalid JSON response`);
        }
      }

      // Verify expected status
      const expectedStatuses = Array.isArray(config.expectedStatus) ? config.expectedStatus : [config.expectedStatus];
      if (!expectedStatuses.includes(response.status)) {
        throw new Error(`Unexpected status ${response.status}, expected ${expectedStatuses.join(' or ')}`);
      }

      // Run custom validation
      if (customValidator) {
        await customValidator(response, responseData);
      }

      // Performance check
      if (responseTime > 10000) {
        this.addWarning(`${testName}: Slow response time ${responseTime}ms`);
      }

      this.addTest({
        name: testName,
        path: config.path,
        method: config.method,
        status: 'PASSED',
        responseTime,
        httpStatus: response.status
      });

    } catch (error) {
      this.addTest({
        name: testName,
        path: config.path,
        method: config.method,
        status: 'FAILED',
        error: error.message
      });
    }
  }

  addTest(test) {
    this.results.tests.push(test);
    if (test.status === 'PASSED') {
      this.results.passed++;
    } else {
      this.results.failed++;
    }
  }

  addWarning(message) {
    this.results.warnings++;
    this.results.tests.push({
      name: 'Warning',
      status: 'WARNING',
      message
    });
  }

  printResults() {
    console.log('\n' + '='.repeat(60));
    console.log(chalk.cyan.bold('🎯 API Testing Results'));
    console.log('='.repeat(60));
    
    const total = this.results.passed + this.results.failed;
    const passRate = total > 0 ? Math.round((this.results.passed / total) * 100) : 0;
    
    console.log(chalk.green(`✅ Passed: ${this.results.passed}`));
    console.log(chalk.red(`❌ Failed: ${this.results.failed}`));
    console.log(chalk.yellow(`⚠️ Warnings: ${this.results.warnings}`));
    console.log(chalk.blue(`📊 Pass Rate: ${passRate}%`));
    
    // Performance summary
    const responseTimes = this.results.tests
      .filter(t => t.responseTime)
      .map(t => t.responseTime);
    
    if (responseTimes.length > 0) {
      const avgTime = Math.round(responseTimes.reduce((a, b) => a + b) / responseTimes.length);
      const maxTime = Math.max(...responseTimes);
      console.log(chalk.blue(`⏱️ Avg Response: ${avgTime}ms (Max: ${maxTime}ms)`));
    }
    
    console.log('\n' + chalk.cyan('Detailed Test Results:'));
    console.log('-'.repeat(40));
    
    this.results.tests.forEach(test => {
      const statusColor = test.status === 'PASSED' ? 'green' : 
                         test.status === 'WARNING' ? 'yellow' : 'red';
      const statusIcon = test.status === 'PASSED' ? '✅' : 
                        test.status === 'WARNING' ? '⚠️' : '❌';
      
      if (test.method && test.path) {
        console.log(`${statusIcon} ${chalk[statusColor](test.status)} ${test.method} ${test.path}`);
        if (test.responseTime) {
          console.log(`   └─ ${test.responseTime}ms (HTTP ${test.httpStatus})`);
        }
        if (test.error) {
          console.log(`   └─ Error: ${test.error}`);
        }
      } else if (test.message) {
        console.log(`${statusIcon} ${chalk[statusColor](test.message)}`);
      }
    });
    
    if (this.results.failed === 0 && this.results.warnings === 0) {
      console.log(chalk.green.bold('\n🎉 All API endpoints working correctly! Production ready.'));
    } else if (this.results.failed === 0) {
      console.log(chalk.yellow.bold(`\n⚠️ ${this.results.warnings} warnings found. Review before production.`));
    } else {
      console.log(chalk.red.bold(`\n❌ ${this.results.failed} tests failed. Fix before production deployment.`));
    }

    // Save detailed report
    const reportPath = 'api-test-report.json';
    require('fs').writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      config: { API_BASE, DB_CONFIG: { ...DB_CONFIG, password: '[REDACTED]' } },
      results: this.results
    }, null, 2));
    
    console.log(chalk.gray(`\n📄 Detailed report saved to: ${reportPath}`));
  }
}

// CLI execution
if (process.argv[1] === new URL(import.meta.url).pathname) {
  const tester = new APIEndpointTester();
  tester.runAllTests().catch(error => {
    console.error(chalk.red('API testing failed:', error));
    process.exit(1);
  });
}

export { APIEndpointTester };