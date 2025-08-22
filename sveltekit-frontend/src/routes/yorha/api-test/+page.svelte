<!-- YoRHa Advanced API Testing Interface -->
<script lang="ts">
  // Svelte runes are provided globally via src/types/svelte-helpers.d.ts
  import { onMount, onDestroy } from 'svelte';
  import { yorhaAPI } from '$lib/components/three/yorha-ui/api/YoRHaAPIClient';
  import {
    Zap,
    Database,
    Cpu,
    Monitor,
    Play,
    Square,
    RotateCcw,
    CheckCircle,
    AlertTriangle,
    XCircle,
    Clock,
    Activity,
    BarChart3,
    Globe,
    Server,
    Code,
    Terminal,
    Settings,
    Download,
    Upload,
    Trash2,
    Filter,
    Search,
    RefreshCw,
    Eye,
    EyeOff,
    Copy,
    ExternalLink,
    ChevronDown,
    ChevronRight,
    Pause,
    StopCircle,
    PlayCircle,
    TestTube
  } from 'lucide-svelte';

  // Enhanced test state management
  let testResults = $state([]);
  let testSuites = $state([]);
  let isRunning = $state(false);
  let selectedEndpoint = $state('all');
  let selectedSuite = $state('all');
  let currentTest = $state(null);
  let batchTesting = $state(false);
  let autoRefresh = $state(false);
  let showAdvanced = $state(false);
  let filterStatus = $state('all');
  let searchTerm = $state('');

  // Performance tracking
  let performanceMetrics = $state({
    totalTests: 0,
    passedTests: 0,
    failedTests: 0,
    averageLatency: 0,
    p95Latency: 0,
    p99Latency: 0,
    throughput: 0,
    errorRate: 0,
    uptime: 100,
    lastReset: new Date()
  });

  // Real-time monitoring
  let realTimeStats = $state({
    totalRequests: 0,
    successRate: 100,
    averageLatency: 0,
    lastUpdate: new Date(),
    activeConnections: 0,
    queuedRequests: 0,
    dataTransferred: 0
  });

  // Enhanced API endpoint configurations
  const apiEndpoints = [
    {
      id: 'system-status',
      name: 'System Status',
      icon: Monitor,
      endpoint: '/api/yorha/system/status',
      method: 'GET',
      category: 'system',
      priority: 'high',
      timeout: 5000,
      expectedStatus: 200,
      description: 'Real-time system health and performance metrics',
      headers: {},
      validation: {
        required: ['status', 'uptime', 'services'],
        statusCodes: [200]
      }
    },
    {
      id: 'enhanced-rag',
      name: 'Enhanced RAG Service',
      icon: Cpu,
      endpoint: '/api/yorha/enhanced-rag',
      method: 'POST',
      category: 'ai',
      priority: 'high',
      timeout: 30000,
      expectedStatus: 200,
      payload: {
        query: 'Legal precedent analysis for contract liability',
        context: 'legal_analysis',
        options: {
          maxResults: 5,
          includeMetadata: true,
          model: 'gemma3-legal'
        }
      },
      description: 'AI-powered legal document analysis and retrieval',
      headers: { 'Content-Type': 'application/json' },
      validation: {
        required: ['results', 'confidence', 'sources'],
        statusCodes: [200, 202]
      }
    },
    {
      id: 'legal-data-search',
      name: 'Legal Data Search',
      icon: Database,
      endpoint: '/api/yorha/legal-data',
      method: 'GET',
      category: 'data',
      priority: 'medium',
      timeout: 10000,
      expectedStatus: 200,
      params: {
        search: 'contract liability indemnification',
        limit: 10,
        offset: 0,
        sortBy: 'relevance',
        includeMetadata: true
      },
      description: 'Semantic legal document search with vector similarity',
      validation: {
        required: ['documents', 'total', 'page'],
        statusCodes: [200]
      }
    },
    {
      id: 'cluster-health',
      name: 'Cluster Health Check',
      icon: Zap,
      endpoint: '/api/v1/cluster/health',
      method: 'GET',
      category: 'infrastructure',
      priority: 'critical',
      timeout: 5000,
      expectedStatus: 200,
      description: 'Microservices cluster status and health monitoring',
      validation: {
        required: ['services', 'healthy', 'timestamp'],
        statusCodes: [200, 503]
      }
    },
    {
      id: 'context7-status',
      name: 'Context7 Multicore Status',
      icon: Server,
      endpoint: '/api/context7/multicore/status',
      method: 'GET',
      category: 'ai',
      priority: 'high',
      timeout: 5000,
      expectedStatus: 200,
      description: 'Context7 multicore processing service health',
      validation: {
        required: ['workers', 'status', 'performance'],
        statusCodes: [200]
      }
    },
    {
      id: 'vector-search',
      name: 'Vector Search Engine',
      icon: Search,
      endpoint: '/api/v1/vector/search',
      method: 'POST',
      category: 'data',
      priority: 'medium',
      timeout: 15000,
      expectedStatus: 200,
      payload: {
        query: 'Employment law discrimination case precedents',
        limit: 20,
        threshold: 0.7,
        includeEmbeddings: false
      },
      description: 'PostgreSQL pgvector similarity search',
      headers: { 'Content-Type': 'application/json' },
      validation: {
        required: ['matches', 'scores', 'metadata'],
        statusCodes: [200]
      }
    },
    {
      id: 'ai-inference',
      name: 'AI Model Inference',
      icon: Code,
      endpoint: '/api/v1/ai/inference',
      method: 'POST',
      category: 'ai',
      priority: 'high',
      timeout: 45000,
      expectedStatus: 200,
      payload: {
        prompt: 'Analyze the legal implications of this contract clause regarding force majeure',
        model: 'gemma3-legal',
        temperature: 0.3,
        maxTokens: 1000
      },
      description: 'Ollama AI model inference with legal specialization',
      headers: { 'Content-Type': 'application/json' },
      validation: {
        required: ['response', 'model', 'usage'],
        statusCodes: [200]
      }
    },
    {
      id: 'graph-query',
      name: 'Neo4j Graph Query',
      icon: Activity,
      endpoint: '/api/yorha/system/graph',
      method: 'POST',
      category: 'data',
      priority: 'low',
      timeout: 10000,
      expectedStatus: 200,
      payload: {
        query: 'MATCH (c:Case)-[:RELATED_TO]->(d:Document) RETURN c.title, d.type LIMIT 10',
        parameters: {}
      },
      description: 'Neo4j knowledge graph traversal and analysis',
      headers: { 'Content-Type': 'application/json' },
      validation: {
        required: ['records', 'summary'],
        statusCodes: [200]
      }
    },
    {
      id: 'nats-publish',
      name: 'NATS Message Publisher',
      icon: Upload,
      endpoint: '/api/v1/nats/publish',
      method: 'POST',
      category: 'messaging',
      priority: 'medium',
      timeout: 5000,
      expectedStatus: 200,
      payload: {
        subject: 'legal.test.message',
        data: {
          type: 'api_test',
          timestamp: new Date().toISOString(),
          payload: 'Test message from YoRHa API interface'
        }
      },
      description: 'NATS messaging system publication test',
      headers: { 'Content-Type': 'application/json' },
      validation: {
        required: ['published', 'messageId'],
        statusCodes: [200]
      }
    },
    {
      id: 'database-health',
      name: 'Database Connectivity',
      icon: Database,
      endpoint: '/api/v1/database/health',
      method: 'GET',
      category: 'infrastructure',
      priority: 'critical',
      timeout: 5000,
      expectedStatus: 200,
      description: 'PostgreSQL, Redis, and Neo4j connectivity check',
      validation: {
        required: ['postgresql', 'redis', 'neo4j'],
        statusCodes: [200, 503]
      }
    }
  ];

  // Test suites for batch testing
  const testSuitesConfig = [
    {
      id: 'smoke-test',
      name: 'Smoke Test Suite',
      description: 'Basic health check of all critical endpoints',
      icon: TestTube,
      endpoints: ['system-status', 'cluster-health', 'database-health', 'context7-status'],
      timeout: 30000,
      parallel: true
    },
    {
      id: 'ai-pipeline',
      name: 'AI Processing Pipeline',
      description: 'Complete AI workflow from RAG to inference',
      icon: Cpu,
      endpoints: ['enhanced-rag', 'vector-search', 'ai-inference', 'context7-status'],
      timeout: 120000,
      parallel: false
    },
    {
      id: 'data-layer',
      name: 'Data Layer Integration',
      description: 'Database and search functionality testing',
      icon: Database,
      endpoints: ['legal-data-search', 'vector-search', 'graph-query', 'database-health'],
      timeout: 60000,
      parallel: true
    },
    {
      id: 'infrastructure',
      name: 'Infrastructure Health',
      description: 'Core system and messaging infrastructure',
      icon: Server,
      endpoints: ['system-status', 'cluster-health', 'database-health', 'nats-publish'],
      timeout: 45000,
      parallel: true
    },
    {
      id: 'performance-test',
      name: 'Performance & Load Test',
      description: 'Stress testing with concurrent requests',
      icon: BarChart3,
      endpoints: ['system-status', 'enhanced-rag', 'legal-data-search'],
      timeout: 180000,
      parallel: true,
      iterations: 10
    }
  ];

  // Performance monitoring intervals
  let monitoringInterval;
  let autoRefreshInterval;

  onMount(() => {
    initializeApiTesting();
    startPerformanceMonitoring();
    if (autoRefresh) {
      startAutoRefresh();
    }
  });

  onDestroy(() => {
    if (monitoringInterval) clearInterval(monitoringInterval);
    if (autoRefreshInterval) clearInterval(autoRefreshInterval);
  });

  function initializeApiTesting() {
    // Initialize test suites
    testSuites = testSuitesConfig.map(suite => ({
      ...suite,
      status: 'pending',
      results: [],
      lastRun: null,
      duration: 0
    }));

    // Subscribe to real-time API events
    startRealTimeMonitoring();
  }

  function startRealTimeMonitoring() {
    yorhaAPI.subscribe('api:request', (data) => {
      realTimeStats.totalRequests++;
      realTimeStats.activeConnections++;
      updateRealTimeStats();
    });

    yorhaAPI.subscribe('api:response', (data) => {
      realTimeStats.activeConnections = Math.max(0, realTimeStats.activeConnections - 1);

      if (data.success) {
        const newSuccessRate = ((realTimeStats.successRate * realTimeStats.totalRequests) + 100) / (realTimeStats.totalRequests + 1);
        realTimeStats.successRate = Math.round(newSuccessRate * 100) / 100;
      } else {
        const newSuccessRate = ((realTimeStats.successRate * realTimeStats.totalRequests) + 0) / (realTimeStats.totalRequests + 1);
        realTimeStats.successRate = Math.round(newSuccessRate * 100) / 100;
      }

      realTimeStats.averageLatency = Math.round(((realTimeStats.averageLatency * (realTimeStats.totalRequests - 1)) + data.latency) / realTimeStats.totalRequests);
      realTimeStats.lastUpdate = new Date();
      realTimeStats.dataTransferred += (data.responseSize || 0);

      updateRealTimeStats();
    });
  }

  function updateRealTimeStats() {
    realTimeStats = { ...realTimeStats };
  }

  function startPerformanceMonitoring() {
    monitoringInterval = setInterval(() => {
      updatePerformanceMetrics();
    }, 5000); // Update every 5 seconds
  }

  function startAutoRefresh() {
    autoRefreshInterval = setInterval(() => {
      if (selectedEndpoint === 'all') {
        runAllTests();
      } else {
        const endpoint = apiEndpoints.find(ep => ep.id === selectedEndpoint);
        if (endpoint) {
          runApiTest(endpoint);
        }
      }
    }, 30000); // Auto refresh every 30 seconds
  }

  function updatePerformanceMetrics() {
    const now = Date.now();
    const recentResults = testResults.filter(result =>
      new Date(result.startTime).getTime() > (now - 300000) // Last 5 minutes
    );

    if (recentResults.length > 0) {
      const latencies = recentResults
        .filter(r => r.status === 'success' && r.latency)
        .map(r => r.latency)
        .sort((a, b) => a - b);

      performanceMetrics.totalTests = testResults.length;
      performanceMetrics.passedTests = testResults.filter(r => r.status === 'success').length;
      performanceMetrics.failedTests = testResults.filter(r => r.status === 'error').length;

      if (latencies.length > 0) {
        performanceMetrics.averageLatency = Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
        performanceMetrics.p95Latency = latencies[Math.floor(latencies.length * 0.95)] || 0;
        performanceMetrics.p99Latency = latencies[Math.floor(latencies.length * 0.99)] || 0;
      }

      performanceMetrics.throughput = Math.round(recentResults.length / 5); // Requests per second over 5 minutes
      performanceMetrics.errorRate = Math.round((performanceMetrics.failedTests / performanceMetrics.totalTests) * 100);
    }

    performanceMetrics = { ...performanceMetrics };
  }

  async function runApiTest(endpoint) {
    const testId = `test-${endpoint.id}-${Date.now()}`;
    const startTime = performance.now();

    const testResult = {
      id: testId,
      endpointId: endpoint.id,
      endpoint: endpoint.name,
      url: endpoint.endpoint,
      method: endpoint.method,
      category: endpoint.category,
      priority: endpoint.priority,
      status: 'running',
      startTime: new Date(),
      endTime: null,
      latency: 0,
      response: null,
      error: null,
      statusCode: null,
      responseSize: 0,
      headers: {},
      validation: {
        passed: false,
        errors: []
      }
    };

    testResults = [testResult, ...testResults.slice(0, 199)]; // Keep last 200 results
    currentTest = testId;

    try {
      let response;
      let url = endpoint.endpoint;

      // Add query parameters for GET requests
      if (endpoint.method === 'GET' && endpoint.params) {
        const params = new URLSearchParams(endpoint.params);
        url += `?${params.toString()}`;
      }

      // Prepare request options
      const requestOptions = {
        method: endpoint.method,
        headers: endpoint.headers || {},
        signal: AbortSignal.timeout(endpoint.timeout || 30000)
      };

      // Add body for POST requests
      if (endpoint.method === 'POST' && endpoint.payload) {
        requestOptions.body = JSON.stringify(endpoint.payload);
        requestOptions.headers['Content-Type'] = 'application/json';
      }

      // Make API request
      response = await fetch(url, requestOptions);

      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);

      // Parse response
      let responseData;
      let responseSize = 0;
      const responseText = await response.text();
      responseSize = responseText.length;

      try {
        responseData = JSON.parse(responseText);
      } catch {
        responseData = responseText;
      }

      // Update test result
      testResult.status = response.ok ? 'success' : 'error';
      testResult.endTime = new Date();
      testResult.latency = latency;
      testResult.response = responseData;
      testResult.statusCode = response.status;
      testResult.responseSize = responseSize;
      testResult.headers = Object.fromEntries(response.headers.entries());

      // Validation
      if (endpoint.validation) {
        testResult.validation = validateResponse(responseData, endpoint.validation, response.status);
      } else {
        testResult.validation.passed = response.ok;
      }

      // Emit event for monitoring
      yorhaAPI.emit('api:response', {
        success: response.ok,
        latency,
        responseSize,
        endpoint: endpoint.id
      });

    } catch (error) {
      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);

      testResult.status = 'error';
      testResult.endTime = new Date();
      testResult.latency = latency;
      testResult.error = error.message;
      testResult.validation.errors.push(error.message);

      // Emit event for monitoring
      yorhaAPI.emit('api:response', {
        success: false,
        latency,
        responseSize: 0,
        endpoint: endpoint.id,
        error: error.message
      });
    } finally {
      // Update test results
      testResults = [...testResults];
      currentTest = null;
    }
  }

  function validateResponse(responseData, validation, statusCode) {
    const result = { passed: true, errors: [] };

    // Check status codes
    if (validation.statusCodes && !validation.statusCodes.includes(statusCode)) {
      result.passed = false;
      result.errors.push(`Expected status codes: ${validation.statusCodes.join(', ')}, got: ${statusCode}`);
    }

    // Check required fields
    if (validation.required && typeof responseData === 'object' && responseData !== null) {
      for (const field of validation.required) {
        if (!(field in responseData)) {
          result.passed = false;
          result.errors.push(`Missing required field: ${field}`);
        }
      }
    }

    return result;
  }

  async function runTestSuite(suite) {
    if (batchTesting) return;

    batchTesting = true;
    isRunning = true;

    const suiteStartTime = Date.now();
    const suiteResults = [];

    // Update suite status
    const suiteIndex = testSuites.findIndex(s => s.id === suite.id);
    testSuites[suiteIndex].status = 'running';
    testSuites[suiteIndex].results = [];
    testSuites = [...testSuites];

    try {
      const endpoints = apiEndpoints.filter(ep => suite.endpoints.includes(ep.id));

      if (suite.parallel) {
        // Run tests in parallel
        const promises = endpoints.map(endpoint => {
          if (suite.iterations && suite.iterations > 1) {
            // Multiple iterations
            return Promise.all(
              Array.from({ length: suite.iterations }, () => runApiTest(endpoint))
            );
          }
          return runApiTest(endpoint);
        });

        await Promise.allSettled(promises);
      } else {
        // Run tests sequentially
        for (const endpoint of endpoints) {
          if (suite.iterations && suite.iterations > 1) {
            // Multiple iterations
            for (let i = 0; i < suite.iterations; i++) {
              await runApiTest(endpoint);
              // Small delay between iterations
              await new Promise(resolve => setTimeout(resolve, 100));
            }
          } else {
            await runApiTest(endpoint);
            // Small delay between tests
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
      }

      // Calculate suite results
      const suiteTestResults = testResults.filter(result =>
        result.startTime.getTime() >= suiteStartTime &&
        suite.endpoints.includes(result.endpointId)
      );

      const successCount = suiteTestResults.filter(r => r.status === 'success').length;
      const totalCount = suiteTestResults.length;

      testSuites[suiteIndex].status = successCount === totalCount ? 'success' : 'partial';
      testSuites[suiteIndex].results = suiteTestResults;
      testSuites[suiteIndex].lastRun = new Date();
      testSuites[suiteIndex].duration = Date.now() - suiteStartTime;

    } catch (error) {
      testSuites[suiteIndex].status = 'error';
      testSuites[suiteIndex].error = error.message;
    } finally {
      testSuites = [...testSuites];
      batchTesting = false;
      isRunning = false;
    }
  }

  async function runAllTests() {
    if (isRunning) return;

    isRunning = true;

    try {
      for (const endpoint of apiEndpoints) {
        await runApiTest(endpoint);
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    } finally {
      isRunning = false;
    }
  }

  function clearResults() {
    testResults = [];
    performanceMetrics = {
      totalTests: 0,
      passedTests: 0,
      failedTests: 0,
      averageLatency: 0,
      p95Latency: 0,
      p99Latency: 0,
      throughput: 0,
      errorRate: 0,
      uptime: 100,
      lastReset: new Date()
    };
  }

  function exportResults() {
    const exportData = {
      timestamp: new Date().toISOString(),
      performanceMetrics,
      realTimeStats,
      testResults: testResults.slice(0, 50), // Last 50 results
      testSuites: testSuites.map(suite => ({
        ...suite,
        results: suite.results?.slice(0, 10) // Last 10 results per suite
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yorha-api-test-results-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;

    if (autoRefresh) {
      startAutoRefresh();
    } else if (autoRefreshInterval) {
      clearInterval(autoRefreshInterval);
    }
  }

  function getFilteredResults() {
    let filtered = testResults;

    // Filter by status
    if (filterStatus !== 'all') {
      filtered = filtered.filter(result => result.status === filterStatus);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(result =>
        result.endpoint.toLowerCase().includes(term) ||
        result.category?.toLowerCase().includes(term) ||
        result.url.toLowerCase().includes(term)
      );
    }

    return filtered;
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'running': return Clock;
      case 'success': return CheckCircle;
      case 'error': return XCircle;
      case 'partial': return AlertTriangle;
      default: return Clock;
    }
  }

  function getStatusColor(status) {
    switch (status) {
      case 'running': return 'text-blue-400';
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'partial': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function getCategoryColor(category) {
    switch (category) {
      case 'system': return 'text-blue-400';
      case 'ai': return 'text-purple-400';
      case 'data': return 'text-green-400';
      case 'infrastructure': return 'text-red-400';
      case 'messaging': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function formatLatency(latency) {
    if (latency < 1000) return `${latency}ms`;
    return `${(latency / 1000).toFixed(1)}s`;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  }

  // Check background TypeScript process
  $effect(() => {
    // This will reactively update when testResults change
    updatePerformanceMetrics();
  });
</script>

<svelte:head>
  <title>YoRHa API Testing Interface | Legal AI Platform</title>
</svelte:head>

<!-- YoRHa Cyberpunk Background -->
<div class="min-h-screen bg-black text-gray-100 font-mono">
  <!-- Animated grid background -->
  <div class="fixed inset-0 opacity-5">
    <div class="absolute inset-0" style="background-image: linear-gradient(rgba(251,191,36,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(251,191,36,0.1) 1px, transparent 1px); background-size: 20px 20px;"></div>
  </div>

  <!-- Header -->
  <header class="relative z-10 border-b border-amber-900/30 bg-black/90 backdrop-blur-sm">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4">
          <TestTube class="w-8 h-8 text-amber-400" />
          <div>
            <h1 class="text-2xl font-bold text-amber-400">YoRHa API Testing Interface</h1>
            <p class="text-gray-400 text-sm">Advanced API monitoring and performance testing</p>
          </div>
        </div>

        <div class="flex items-center space-x-4">
          <!-- Real-time status indicators -->
          <div class="flex items-center space-x-4 text-sm">
            <div class="flex items-center space-x-1">
              <div class="w-2 h-2 bg-green-400 rounded-full {realTimeStats.activeConnections > 0 ? 'animate-pulse' : ''}"></div>
              <span class="text-green-400">{realTimeStats.totalRequests} requests</span>
            </div>
            <div class="flex items-center space-x-1">
              <Activity class="w-4 h-4 text-blue-400" />
              <span class="text-blue-400">{realTimeStats.successRate.toFixed(1)}% success</span>
            </div>
            <div class="flex items-center space-x-1">
              <Clock class="w-4 h-4 text-yellow-400" />
              <span class="text-yellow-400">{formatLatency(realTimeStats.averageLatency)}</span>
            </div>
          </div>

          <!-- Control buttons -->
          <button
            class="p-2 rounded-lg bg-amber-900/20 border border-amber-400/30 hover:bg-amber-400/10 text-amber-400"
            class:animate-spin={autoRefresh}
            onclick={toggleAutoRefresh}
            title={autoRefresh ? 'Disable auto-refresh' : 'Enable auto-refresh'}
          >
            <RefreshCw class="w-4 h-4" />
          </button>

          <button
            class="p-2 rounded-lg bg-blue-900/20 border border-blue-400/30 hover:bg-blue-400/10 text-blue-400"
            onclick={exportResults}
            title="Export test results"
          >
            <Download class="w-4 h-4" />
          </button>

          <button
            class="p-2 rounded-lg bg-red-900/20 border border-red-400/30 hover:bg-red-400/10 text-red-400"
            onclick={clearResults}
            title="Clear all results"
          >
            <Trash2 class="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="relative z-10 container mx-auto px-6 py-8">

    <!-- Performance Metrics Dashboard -->
    <div class="mb-8">
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
        <!-- Total Tests -->
        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-400 text-sm">Total Tests</p>
              <p class="text-2xl font-bold text-gray-100">{performanceMetrics.totalTests}</p>
            </div>
            <TestTube class="w-8 h-8 text-gray-400" />
          </div>
        </div>

        <!-- Success Rate -->
        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-400 text-sm">Success Rate</p>
              <p class="text-2xl font-bold text-green-400">
                {performanceMetrics.totalTests > 0 ? Math.round((performanceMetrics.passedTests / performanceMetrics.totalTests) * 100) : 0}%
              </p>
            </div>
            <CheckCircle class="w-8 h-8 text-green-400" />
          </div>
        </div>

        <!-- Average Latency -->
        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-400 text-sm">Avg Latency</p>
              <p class="text-2xl font-bold text-blue-400">{formatLatency(performanceMetrics.averageLatency)}</p>
            </div>
            <Clock class="w-8 h-8 text-blue-400" />
          </div>
        </div>

        <!-- Error Rate -->
        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-400 text-sm">Error Rate</p>
              <p class="text-2xl font-bold text-red-400">{performanceMetrics.errorRate}%</p>
            </div>
            <AlertTriangle class="w-8 h-8 text-red-400" />
          </div>
        </div>
      </div>

      <!-- Advanced Metrics -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <h3 class="text-lg font-semibold text-gray-200 mb-3">Latency Distribution</h3>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">P95:</span>
              <span class="text-yellow-400">{formatLatency(performanceMetrics.p95Latency)}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">P99:</span>
              <span class="text-red-400">{formatLatency(performanceMetrics.p99Latency)}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Throughput:</span>
              <span class="text-green-400">{performanceMetrics.throughput} req/s</span>
            </div>
          </div>
        </div>

        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <h3 class="text-lg font-semibold text-gray-200 mb-3">Real-time Stats</h3>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">Active:</span>
              <span class="text-blue-400">{realTimeStats.activeConnections}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Queued:</span>
              <span class="text-yellow-400">{realTimeStats.queuedRequests}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Data:</span>
              <span class="text-green-400">{formatBytes(realTimeStats.dataTransferred)}</span>
            </div>
          </div>
        </div>

        <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-4">
          <h3 class="text-lg font-semibold text-gray-200 mb-3">System Status</h3>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-400">Uptime:</span>
              <span class="text-green-400">{performanceMetrics.uptime}%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Last Update:</span>
              <span class="text-gray-400">{realTimeStats.lastUpdate.toLocaleTimeString()}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-400">Auto Refresh:</span>
              <span class="text-{autoRefresh ? 'green' : 'red'}-400">{autoRefresh ? 'ON' : 'OFF'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Test Controls -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">

      <!-- Individual API Tests -->
      <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-amber-400 mb-4">Individual API Tests</h2>

        <div class="space-y-4">
          <!-- Test Controls -->
          <div class="flex flex-wrap gap-3 mb-4">
            <button
              class="px-4 py-2 rounded-lg bg-green-900/20 border border-green-400/30 hover:bg-green-400/10 text-green-400 disabled:opacity-50"
              onclick={runAllTests}
              disabled={isRunning}
            >
              <Play class="w-4 h-4 inline-block mr-2" />
              {isRunning ? 'Running...' : 'Run All Tests'}
            </button>

            <button
              class="px-4 py-2 rounded-lg bg-gray-700/50 border border-gray-600 hover:bg-gray-600/50 text-gray-300"
              onclick={() => showAdvanced = !showAdvanced}
            >
              {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
              {showAdvanced ? <ChevronDown class="w-4 h-4 inline-block ml-2" /> : <ChevronRight class="w-4 h-4 inline-block ml-2" />}
            </button>
          </div>

          <!-- API Endpoints -->
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each apiEndpoints as endpoint}
              <div class="flex items-center justify-between p-3 bg-black/30 border border-gray-800 rounded-lg hover:border-gray-700 transition-colors">
                <div class="flex items-center space-x-3">
                  <svelte:component this={endpoint.icon} class="w-5 h-5 {getCategoryColor(endpoint.category)}" />
                  <div>
                    <p class="font-medium text-gray-200">{endpoint.name}</p>
                    <p class="text-sm text-gray-400">{endpoint.method} {endpoint.endpoint}</p>
                  </div>
                </div>

                <div class="flex items-center space-x-2">
                  <span class="px-2 py-1 rounded text-xs {getCategoryColor(endpoint.category)} border border-gray-700">
                    {endpoint.category}
                  </span>
                  <button
                    class="px-3 py-1 rounded-lg bg-blue-900/20 border border-blue-400/30 hover:bg-blue-400/10 text-blue-400 disabled:opacity-50"
                    onclick={() => runApiTest(endpoint)}
                    disabled={isRunning && currentTest?.includes(endpoint.id)}
                  >
                    {isRunning && currentTest?.includes(endpoint.id) ?
                      <Clock class="w-4 h-4 animate-spin" /> :
                      <Play class="w-4 h-4" />
                    }
                  </button>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <!-- Test Suites -->
      <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-purple-400 mb-4">Test Suites</h2>

        <div class="space-y-4">
          {#each testSuites as suite}
            <div class="p-4 bg-black/30 border border-gray-800 rounded-lg">
              <div class="flex items-center justify-between mb-2">
                <div class="flex items-center space-x-3">
                  <svelte:component this={suite.icon} class="w-5 h-5 text-purple-400" />
                  <div>
                    <h3 class="font-medium text-gray-200">{suite.name}</h3>
                    <p class="text-sm text-gray-400">{suite.description}</p>
                  </div>
                </div>

                <div class="flex items-center space-x-2">
                  {#if suite.status !== 'pending'}
                    <svelte:component this={getStatusIcon(suite.status)} class="w-4 h-4 {getStatusColor(suite.status)}" />
                  {/if}
                  <button
                    class="px-3 py-1 rounded-lg bg-purple-900/20 border border-purple-400/30 hover:bg-purple-400/10 text-purple-400 disabled:opacity-50"
                    onclick={() => runTestSuite(suite)}
                    disabled={batchTesting}
                  >
                    {suite.status === 'running' ?
                      <Clock class="w-4 h-4 animate-spin" /> :
                      <PlayCircle class="w-4 h-4" />
                    }
                  </button>
                </div>
              </div>

              <div class="flex items-center justify-between text-sm text-gray-400">
                <span>{suite.endpoints.length} endpoints</span>
                {#if suite.lastRun}
                  <span>Last run: {suite.lastRun.toLocaleTimeString()}</span>
                {/if}
                {#if suite.duration}
                  <span>Duration: {formatLatency(suite.duration)}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    </div>

    <!-- Test Results -->
    <div class="bg-gray-900/50 border border-gray-700 rounded-lg p-6">
      <div class="flex items-center justify-between mb-6">
        <h2 class="text-xl font-semibold text-green-400">Test Results</h2>

        <div class="flex items-center space-x-4">
          <!-- Search and Filter -->
          <div class="flex items-center space-x-2">
            <div class="relative">
              <Search class="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search results..."
                bind:value={searchTerm}
                class="pl-10 pr-4 py-2 bg-black/50 border border-gray-600 rounded-lg text-gray-200 placeholder-gray-400 focus:border-amber-400 focus:outline-none"
              />
            </div>

            <select
              bind:value={filterStatus}
              class="px-3 py-2 bg-black/50 border border-gray-600 rounded-lg text-gray-200 focus:border-amber-400 focus:outline-none"
            >
              <option value="all">All Status</option>
              <option value="success">Success</option>
              <option value="error">Error</option>
              <option value="running">Running</option>
            </select>
          </div>

          <span class="text-sm text-gray-400">
            {getFilteredResults().length} of {testResults.length} results
          </span>
        </div>
      </div>

      <!-- Results Table -->
      <div class="overflow-x-auto">
        <div class="max-h-96 overflow-y-auto">
          {#if getFilteredResults().length === 0}
            <div class="text-center py-12">
              <TestTube class="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p class="text-gray-400">No test results yet. Run some API tests to see results here.</p>
            </div>
          {:else}
            <table class="w-full">
              <thead class="sticky top-0 bg-gray-900/90 backdrop-blur-sm">
                <tr class="border-b border-gray-700">
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Status</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Endpoint</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Method</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Latency</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Status Code</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Time</th>
                  <th class="text-left py-3 px-4 text-gray-400 font-medium">Size</th>
                </tr>
              </thead>
              <tbody>
                {#each getFilteredResults() as result}
                  <tr class="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                    <td class="py-3 px-4">
                      <div class="flex items-center space-x-2">
                        <svelte:component this={getStatusIcon(result.status)} class="w-4 h-4 {getStatusColor(result.status)}" />
                        <span class="text-sm {getStatusColor(result.status)} capitalize">{result.status}</span>
                      </div>
                    </td>
                    <td class="py-3 px-4">
                      <div>
                        <p class="text-gray-200 font-medium">{result.endpoint}</p>
                        <p class="text-xs text-gray-400">{result.url}</p>
                      </div>
                    </td>
                    <td class="py-3 px-4">
                      <span class="px-2 py-1 rounded text-xs font-mono bg-gray-700 text-gray-300">
                        {result.method}
                      </span>
                    </td>
                    <td class="py-3 px-4">
                      <span class="text-blue-400 font-mono">
                        {result.latency ? formatLatency(result.latency) : '-'}
                      </span>
                    </td>
                    <td class="py-3 px-4">
                      {#if result.statusCode}
                        <span class="px-2 py-1 rounded text-xs font-mono {result.statusCode >= 200 && result.statusCode < 300 ? 'bg-green-900/30 text-green-400' : result.statusCode >= 400 ? 'bg-red-900/30 text-red-400' : 'bg-yellow-900/30 text-yellow-400'}">
                          {result.statusCode}
                        </span>
                      {:else}
                        <span class="text-gray-500">-</span>
                      {/if}
                    </td>
                    <td class="py-3 px-4">
                      <span class="text-gray-400 text-sm">
                        {result.startTime.toLocaleTimeString()}
                      </span>
                    </td>
                    <td class="py-3 px-4">
                      <span class="text-gray-400 text-sm">
                        {result.responseSize ? formatBytes(result.responseSize) : '-'}
                      </span>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      </div>
    </div>

  </main>
</div>

<style>
  /* Custom scrollbar for YoRHa theme */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: rgba(0, 0, 0, 0.2);
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: rgba(251, 191, 36, 0.3);
    border-radius: 3px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: rgba(251, 191, 36, 0.5);
  }

  /* Glowing animation for active connections */
  @keyframes glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>