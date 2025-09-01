<!--
  Test page for Gallery API endpoints
  Tests all gallery functionality including upload, search, and management
-->
<script lang="ts">
  import { onMount } from 'svelte';
  
  let apiTests = $state([]);
  let isTestingInProgress = $state(false);
  let testResults = $state({});
  
  interface ApiTest {
    name: string;
    endpoint: string;
    method: string;
    description: string;
    payload?: any;
    expectedStatus?: number;
  }
  
  const galleryApiTests: ApiTest[] = [
    {
      name: 'Gallery Main API - GET',
      endpoint: '/api/gallery',
      method: 'GET',
      description: 'Test fetching gallery items with default parameters',
      expectedStatus: 200
    },
    {
      name: 'Gallery Main API - GET with filters',
      endpoint: '/api/gallery?page=1&pageSize=5&type=evidence',
      method: 'GET',
      description: 'Test fetching gallery items with filters',
      expectedStatus: 200
    },
    {
      name: 'Gallery Search API - GET',
      endpoint: '/api/gallery/search?q=test&page=1&pageSize=10',
      method: 'GET',
      description: 'Test simple search functionality',
      expectedStatus: 200
    },
    {
      name: 'Gallery Search API - POST',
      endpoint: '/api/gallery/search',
      method: 'POST',
      description: 'Test advanced search with filters',
      payload: {
        filters: {
          query: 'legal document',
          contentSearch: true,
          types: ['evidence', 'document']
        },
        options: {
          page: 1,
          pageSize: 10,
          sortBy: 'uploadedAt',
          sortOrder: 'desc'
        }
      },
      expectedStatus: 200
    },
    {
      name: 'Gallery Item Detail - Non-existent',
      endpoint: '/api/gallery/non-existent-id',
      method: 'GET',
      description: 'Test fetching non-existent gallery item',
      expectedStatus: 404
    },
    {
      name: 'Gallery Upload Status',
      endpoint: '/api/gallery/upload',
      method: 'GET',
      description: 'Test upload status endpoint',
      expectedStatus: 200
    }
  ];
  
  async function runApiTest(test: ApiTest): Promise<any> {
    const startTime = Date.now();
    
    try {
      const options: RequestInit = {
        method: test.method,
        headers: {
          'Content-Type': 'application/json'
        }
      };
      
      if (test.payload && (test.method === 'POST' || test.method === 'PUT')) {
        options.body = JSON.stringify(test.payload);
      }
      
      const response = await fetch(test.endpoint, options);
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      let responseData = null;
      const contentType = response.headers.get('content-type');
      
      if (contentType && contentType.includes('application/json')) {
        try {
          responseData = await response.json();
        } catch (e) {
          responseData = { error: 'Invalid JSON response' };
        }
      } else {
        responseData = { text: await response.text() };
      }
      
      return {
        success: true,
        status: response.status,
        statusText: response.statusText,
        responseTime,
        data: responseData,
        headers: Object.fromEntries(response.headers.entries()),
        expectedStatus: test.expectedStatus,
        statusMatch: test.expectedStatus ? response.status === test.expectedStatus : true
      };
      
    } catch (error) {
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      return {
        success: false,
        error: error.message,
        responseTime,
        expectedStatus: test.expectedStatus,
        statusMatch: false
      };
    }
  }
  
  async function runAllTests() {
    isTestingInProgress = true;
    testResults = {};
    
    console.log('Starting Gallery API Tests...');
    
    for (const test of galleryApiTests) {
      console.log(`Running test: ${test.name}`);
      
      try {
        const result = await runApiTest(test);
        testResults[test.name] = {
          ...result,
          test: test
        };
        
        // Force reactivity update
        testResults = { ...testResults };
        
        // Brief delay between tests
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        testResults[test.name] = {
          success: false,
          error: error.message,
          test: test
        };
        testResults = { ...testResults };
      }
    }
    
    isTestingInProgress = false;
    console.log('All tests completed');
  }
  
  function getStatusColor(result: any): string {
    if (!result.success) return 'error';
    if (result.statusMatch) return 'success';
    if (result.status >= 200 && result.status < 300) return 'warning';
    return 'error';
  }
  
  function formatJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }
  
  onMount(() => {
    // Auto-run tests after component mounts
    setTimeout(() => {
      runAllTests();
    }, 500);
  });
</script>

<svelte:head>
  <title>Gallery API Test Suite</title>
</svelte:head>

<div class="api-test-page">
  <header class="test-header">
    <h1>🧪 Gallery API Test Suite</h1>
    <p>Comprehensive testing of all gallery endpoints and functionality</p>
    <div class="test-actions">
      <button 
        class="test-button primary" 
        onclick={runAllTests}
        disabled={isTestingInProgress}
      >
        {isTestingInProgress ? '🔄 Testing...' : '🚀 Run All Tests'}
      </button>
    </div>
  </header>

  <main class="test-content">
    <!-- Test Overview -->
    <section class="test-overview">
      <h2>Test Overview</h2>
      <div class="overview-grid">
        <div class="overview-item">
          <span class="overview-label">Total Tests</span>
          <span class="overview-value">{galleryApiTests.length}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">Completed</span>
          <span class="overview-value">{Object.keys(testResults).length}</span>
        </div>
        <div class="overview-item">
          <span class="overview-label">Success Rate</span>
          <span class="overview-value">
            {Object.keys(testResults).length > 0 
              ? Math.round((Object.values(testResults).filter(r => r.success).length / Object.keys(testResults).length) * 100)
              : 0}%
          </span>
        </div>
      </div>
    </section>

    <!-- Test Results -->
    <section class="test-results">
      <h2>Test Results</h2>
      
      {#if Object.keys(testResults).length === 0 && !isTestingInProgress}
        <div class="no-results">
          <p>No test results yet. Click "Run All Tests" to start testing.</p>
        </div>
      {:else}
        <div class="results-grid">
          {#each galleryApiTests as test}
            {@const result = testResults[test.name]}
            <div class="result-card" class:testing={!result && isTestingInProgress}>
              <div class="result-header">
                <h3>{test.name}</h3>
                {#if result}
                  <span class="status-badge {getStatusColor(result)}">
                    {result.success ? (result.statusMatch ? '✅' : '⚠️') : '❌'}
                  </span>
                {:else if isTestingInProgress}
                  <span class="status-badge testing">🔄</span>
                {:else}
                  <span class="status-badge pending">⏳</span>
                {/if}
              </div>
              
              <div class="result-details">
                <p class="test-description">{test.description}</p>
                <div class="endpoint-info">
                  <span class="method {test.method.toLowerCase()}">{test.method}</span>
                  <span class="endpoint">{test.endpoint}</span>
                </div>
                
                {#if result}
                  <div class="result-metrics">
                    <div class="metric">
                      <span class="metric-label">Status:</span>
                      <span class="metric-value {getStatusColor(result)}">
                        {result.status || 'Error'} {result.statusText || ''}
                      </span>
                    </div>
                    <div class="metric">
                      <span class="metric-label">Response Time:</span>
                      <span class="metric-value">{result.responseTime}ms</span>
                    </div>
                    {#if result.expectedStatus}
                      <div class="metric">
                        <span class="metric-label">Expected:</span>
                        <span class="metric-value">{result.expectedStatus}</span>
                      </div>
                    {/if}
                  </div>
                  
                  {#if result.error}
                    <div class="error-details">
                      <strong>Error:</strong> {result.error}
                    </div>
                  {/if}
                  
                  {#if result.data}
                    <details class="response-details">
                      <summary>Response Data</summary>
                      <pre class="response-json">{formatJson(result.data)}</pre>
                    </details>
                  {/if}
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <!-- Gallery Route Test -->
    <section class="route-test">
      <h2>Gallery Route Test</h2>
      <div class="route-buttons">
        <a href="/gallery" class="test-button secondary">
          🖼️ Open Gallery Route
        </a>
        <a href="/test/n64-button" class="test-button info">
          🎮 Test N64 Button
        </a>
        <button 
          class="test-button warning"
          onclick={() => window.open('/gallery?debug=true', '_blank')}
        >
          🐛 Open Gallery with Debug
        </button>
      </div>
    </section>
  </main>
</div>

<style>
  .api-test-page {
    min-height: 100vh;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 2rem;
  }

  .test-header {
    text-align: center;
    margin-bottom: 3rem;
  }

  .test-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .test-actions {
    margin-top: 1.5rem;
  }

  .test-button {
    display: inline-block;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.95rem;
  }

  .test-button.primary {
    background: linear-gradient(45deg, #3b82f6, #1d4ed8);
    color: white;
  }

  .test-button.secondary {
    background: linear-gradient(45deg, #6b7280, #374151);
    color: white;
  }

  .test-button.info {
    background: linear-gradient(45deg, #06b6d4, #0891b2);
    color: white;
  }

  .test-button.warning {
    background: linear-gradient(45deg, #f59e0b, #d97706);
    color: white;
  }

  .test-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
  }

  .test-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .test-content {
    max-width: 1400px;
    margin: 0 auto;
  }

  .test-overview {
    margin-bottom: 3rem;
  }

  .test-overview h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  .overview-item {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
  }

  .overview-label {
    display: block;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }

  .overview-value {
    display: block;
    font-size: 2rem;
    font-weight: bold;
    color: #60a5fa;
  }

  .test-results h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .no-results {
    text-align: center;
    padding: 3rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
  }

  .result-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
  }

  .result-card.testing {
    border-color: #60a5fa;
    background: rgba(96, 165, 250, 0.1);
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .result-header h3 {
    margin: 0;
    font-size: 1.1rem;
  }

  .status-badge {
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .status-badge.success {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .status-badge.warning {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .status-badge.error {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .status-badge.testing {
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
  }

  .status-badge.pending {
    background: rgba(156, 163, 175, 0.2);
    color: #9ca3af;
  }

  .test-description {
    color: #94a3b8;
    margin-bottom: 1rem;
    line-height: 1.5;
  }

  .endpoint-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .method {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bold;
    text-transform: uppercase;
  }

  .method.get {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .method.post {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
  }

  .method.put {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .method.delete {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .endpoint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    background: rgba(0, 0, 0, 0.2);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .result-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
  }

  .metric-value {
    font-weight: 600;
  }

  .metric-value.success {
    color: #22c55e;
  }

  .metric-value.warning {
    color: #f59e0b;
  }

  .metric-value.error {
    color: #ef4444;
  }

  .error-details {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
    color: #fca5a5;
  }

  .response-details {
    margin-top: 1rem;
  }

  .response-details summary {
    cursor: pointer;
    font-weight: 600;
    color: #60a5fa;
    padding: 0.5rem 0;
  }

  .response-json {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    overflow-x: auto;
    white-space: pre-wrap;
    color: #e2e8f0;
  }

  .route-test {
    margin-bottom: 3rem;
  }

  .route-test h2 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }

  .route-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  /* Mobile responsiveness */
  @media (max-width: 768px) {
    .api-test-page {
      padding: 1rem;
    }
    
    .results-grid {
      grid-template-columns: 1fr;
    }
    
    .overview-grid {
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    
    .route-buttons {
      flex-direction: column;
      align-items: center;
    }
  }
</style>