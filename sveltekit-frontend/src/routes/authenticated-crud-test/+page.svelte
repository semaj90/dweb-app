<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import Button from '$lib/components/ui/Button.svelte';

  // Svelte 5 runes for reactive state
  let isLoading = $state(false);
  let testResults = $state<string[]>([]);
  let cases = $state<any[]>([]);
  let currentUser = $state<any>(null);
  let isAuthenticated = $state(false);
  let authError = $state<string | null>(null);

  // Test case form data
  let newCase = $state({
    caseNumber: `CASE-${Date.now()}`,
    title: 'Authenticated Test Case',
    description: 'Testing CRUD operations with authenticated user and PostgreSQL integration',
    priority: 'medium' as 'low' | 'medium' | 'high' | 'critical',
    status: 'draft' as 'draft' | 'open' | 'in_progress' | 'review' | 'closed',
    category: 'testing',
    tags: ['test', 'authenticated', 'crud'],
    metadata: { 
      test: true, 
      pgvector: true,
      authenticated: true,
      timestamp: Date.now() 
    }
  });

  // Derived state for test summary
  let testSummary = $derived(() => {
    const total = testResults.length;
    const passed = testResults.filter(r => r.includes('✅')).length;
    const failed = testResults.filter(r => r.includes('❌')).length;
    const warnings = testResults.filter(r => r.includes('⚠️')).length;
    return { 
      total, 
      passed, 
      failed, 
      warnings, 
      successRate: total > 0 ? (passed / total * 100).toFixed(1) : '0' 
    };
  });

  function addResult(message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info') {
    const icons = { info: '📝', success: '✅', error: '❌', warning: '⚠️' };
    const timestamp = new Date().toLocaleTimeString();
    testResults = [...testResults, `[${timestamp}] ${icons[type]} ${message}`];
  }

  // Check authentication status
  async function checkAuth() {
    try {
      const response = await fetch('/api/test-cases?limit=1');
      const data = await response.json();
      
      if (response.status === 401) {
        isAuthenticated = false;
        authError = 'Authentication required - please log in';
        addResult('Authentication check failed - user not logged in', 'error');
        return false;
      }
      
      if (response.ok && data.user) {
        isAuthenticated = true;
        currentUser = data.user;
        authError = null;
        addResult(`Authentication verified - logged in as ${data.user.email}`, 'success');
        return true;
      }
      
      return false;
    } catch (error) {
      authError = 'Failed to check authentication';
      addResult(`Authentication check error: ${error instanceof Error ? error.message : 'Unknown'}`, 'error');
      return false;
    }
  }

  // Test authenticated GET operations
  async function testAuthenticatedGET() {
    if (!isAuthenticated) {
      addResult('Skipping GET test - not authenticated', 'warning');
      return;
    }

    isLoading = true;
    addResult('🔍 Testing authenticated GET operations...');
    
    try {
      // Test 1: Get user's cases
      const listResponse = await fetch('/api/test-cases?limit=10');
      const listData = await listResponse.json();
      
      if (listResponse.status === 401) {
        addResult('GET operation failed - session expired', 'error');
        isAuthenticated = false;
        return;
      }
      
      if (listResponse.ok && listData.success) {
        addResult(`GET /api/test-cases - Success (${listData.data?.length || 0} user cases)`, 'success');
        addResult(`User context: ${listData.user?.email} (role: ${listData.user?.role})`, 'info');
        cases = listData.data || [];
        
        // Test with search
        const searchResponse = await fetch('/api/test-cases?search=test&limit=5');
        const searchData = await searchResponse.json();
        
        if (searchResponse.ok && searchData.success) {
          addResult(`GET with search - Found ${searchData.data?.length || 0} matching cases`, 'success');
        }
      } else {
        addResult(`GET /api/test-cases - Failed: ${listData.error}`, 'error');
      }

      // Test 2: Get specific case (if any exist)
      if (cases.length > 0) {
        const testCaseId = cases[0].id;
        const singleResponse = await fetch(`/api/test-cases?id=${testCaseId}`);
        const singleData = await singleResponse.json();
        
        if (singleResponse.ok && singleData.success) {
          addResult(`GET specific case - Success with ${singleData.data.documents?.length || 0} docs, ${singleData.data.activities?.length || 0} activities`, 'success');
        } else {
          addResult(`GET specific case - Failed: ${singleData.error}`, 'error');
        }
      }

    } catch (error) {
      addResult(`GET operations error: ${error instanceof Error ? error.message : 'Unknown'}`, 'error');
    }
    
    isLoading = false;
  }

  // Test authenticated POST operation
  async function testAuthenticatedPOST() {
    if (!isAuthenticated) {
      addResult('Skipping POST test - not authenticated', 'warning');
      return;
    }

    isLoading = true;
    addResult('📝 Testing authenticated POST operation with pgvector embeddings...');
    
    try {
      const response = await fetch('/api/test-cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newCase,
          caseNumber: `AUTH-${Date.now()}` // Unique case number
        })
      });
      
      const data = await response.json();
      
      if (response.status === 401) {
        addResult('POST operation failed - session expired', 'error');
        isAuthenticated = false;
        return null;
      }
      
      if (response.ok && data.success) {
        addResult(`POST /api/test-cases - Success (ID: ${data.data?.id})`, 'success');
        addResult(`Embedding generated: ${data.data?.hasEmbedding ? 'Yes' : 'No'}`, 'info');
        addResult(`Created by: ${data.data?.createdBy?.name || data.data?.createdBy?.email}`, 'info');
        
        // Refresh cases list
        await testAuthenticatedGET();
        return data.data.id;
      } else {
        addResult(`POST /api/test-cases - Failed: ${data.message || data.error}`, 'error');
        if (data.details) {
          addResult(`   Details: ${JSON.stringify(data.details)}`, 'error');
        }
      }
    } catch (error) {
      addResult(`POST operation error: ${error instanceof Error ? error.message : 'Unknown'}`, 'error');
    }
    
    isLoading = false;
    return null;
  }

  // Test authenticated PUT operation
  async function testAuthenticatedPUT(caseId?: string) {
    if (!isAuthenticated) {
      addResult('Skipping PUT test - not authenticated', 'warning');
      return;
    }

    if (!caseId && cases.length === 0) {
      addResult('PUT operation skipped - no cases available to update', 'warning');
      return;
    }
    
    isLoading = true;
    const targetId = caseId || cases[0].id;
    addResult(`📝 Testing authenticated PUT operation on case ${targetId}...`);
    
    try {
      const updateData = {
        title: 'Updated Authenticated Test Case',
        description: 'Updated via authenticated API test with new embeddings',
        status: 'in_progress',
        priority: 'high',
        tags: ['updated', 'authenticated', 'pgvector'],
        metadata: { 
          updated: true, 
          timestamp: Date.now(),
          updatedViaTest: true 
        }
      };

      const response = await fetch(`/api/test-cases?id=${targetId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData)
      });
      
      const data = await response.json();
      
      if (response.status === 401) {
        addResult('PUT operation failed - session expired', 'error');
        isAuthenticated = false;
        return;
      }
      
      if (response.status === 403) {
        addResult('PUT operation failed - access denied (not case owner)', 'error');
        return;
      }
      
      if (response.ok && data.success) {
        addResult(`PUT /api/test-cases - Success`, 'success');
        addResult(`New embedding generated: ${data.data?.hasNewEmbedding ? 'Yes' : 'No'}`, 'info');
        addResult(`Updated by: ${data.data?.updatedBy?.name || data.data?.updatedBy?.email}`, 'info');
        addResult(`Changed fields: ${data.data?.changedFields?.join(', ')}`, 'info');
        
        // Refresh cases list
        await testAuthenticatedGET();
      } else {
        addResult(`PUT /api/test-cases - Failed: ${data.message || data.error}`, 'error');
      }
    } catch (error) {
      addResult(`PUT operation error: ${error instanceof Error ? error.message : 'Unknown'}`, 'error');
    }
    
    isLoading = false;
  }

  // Test authenticated DELETE operation
  async function testAuthenticatedDELETE(caseId?: string) {
    if (!isAuthenticated) {
      addResult('Skipping DELETE test - not authenticated', 'warning');
      return;
    }

    if (!caseId && cases.length === 0) {
      addResult('DELETE operation skipped - no cases available to delete', 'warning');
      return;
    }
    
    isLoading = true;
    const targetId = caseId || cases[cases.length - 1].id; // Delete the last case
    addResult(`🗑️ Testing authenticated DELETE operation on case ${targetId}...`);
    
    try {
      const response = await fetch(`/api/test-cases?id=${targetId}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (response.status === 401) {
        addResult('DELETE operation failed - session expired', 'error');
        isAuthenticated = false;
        return;
      }
      
      if (response.status === 403) {
        addResult('DELETE operation failed - access denied (not case owner or admin)', 'error');
        return;
      }
      
      if (response.ok && data.success) {
        addResult(`DELETE /api/test-cases - Success`, 'success');
        addResult(`Deleted by: ${data.data?.deletedBy?.name || data.data?.deletedBy?.email}`, 'info');
        addResult(`Related data cleaned: timeline(${data.data?.relatedDataDeleted?.timeline}), activities(${data.data?.relatedDataDeleted?.activities}), docs(${data.data?.relatedDataDeleted?.documents})`, 'info');
        
        // Refresh cases list
        await testAuthenticatedGET();
      } else {
        addResult(`DELETE /api/test-cases - Failed: ${data.message || data.error}`, 'error');
      }
    } catch (error) {
      addResult(`DELETE operation error: ${error instanceof Error ? error.message : 'Unknown'}`, 'error');
    }
    
    isLoading = false;
  }

  // Run full authenticated CRUD test suite
  async function runAuthenticatedCRUDTest() {
    testResults = [];
    addResult('🚀 Starting authenticated CRUD test suite with PostgreSQL + pgvector...');
    
    // Check authentication first
    const authOk = await checkAuth();
    if (!authOk) {
      addResult('Cannot proceed - authentication required', 'error');
      return;
    }
    
    // Test 1: GET operations
    await testAuthenticatedGET();
    
    // Test 2: POST operation
    const newCaseId = await testAuthenticatedPOST();
    
    // Test 3: PUT operation (use newly created case if available)
    if (newCaseId) {
      await testAuthenticatedPUT(newCaseId);
    } else if (cases.length > 0) {
      await testAuthenticatedPUT();
    }
    
    // Test 4: DELETE operation (clean up)
    if (newCaseId) {
      await testAuthenticatedDELETE(newCaseId);
    }
    
    addResult('✅ Authenticated CRUD test suite completed', 'success');
    addResult(`Final summary: ${testSummary.passed}/${testSummary.total} tests passed`, 'info');
  }

  function clearResults() {
    testResults = [];
  }

  function goToLogin() {
    goto('/login');
  }

  // Initialize on mount
  onMount(async () => {
    await checkAuth();
    if (isAuthenticated) {
      await testAuthenticatedGET();
    }
  });
</script>

<div class="container mx-auto p-6 space-y-6">
  <div class="text-center">
    <h1 class="text-3xl font-bold mb-2">Authenticated CRUD Operations Test</h1>
    <p class="text-gray-600">Testing authenticated CRUD operations with PostgreSQL, pgvector embeddings, and user sessions</p>
  </div>

  <!-- Authentication Status -->
  <Card>
    <CardHeader>
      <CardTitle class="flex items-center gap-2">
        {#if isAuthenticated}
          <span class="w-3 h-3 bg-green-500 rounded-full"></span>
          Authentication Status: Connected
        {:else}
          <span class="w-3 h-3 bg-red-500 rounded-full"></span>
          Authentication Status: Not Authenticated
        {/if}
      </CardTitle>
    </CardHeader>
    <CardContent>
      {#if isAuthenticated && currentUser}
        <div class="space-y-2">
          <p><strong>User:</strong> {currentUser.email}</p>
          <p><strong>Role:</strong> {currentUser.role}</p>
          <p><strong>User ID:</strong> {currentUser.id}</p>
        </div>
      {:else if authError}
        <div class="text-red-600 space-y-2">
          <p><strong>Error:</strong> {authError}</p>
          <Button on:on:on:click={goToLogin} variant="default">Go to Login</Button>
        </div>
      {:else}
        <p class="text-gray-500">Checking authentication...</p>
      {/if}
    </CardContent>
  </Card>

  <!-- Test Controls -->
  <Card>
    <CardHeader>
      <CardTitle>Test Controls</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="flex flex-wrap gap-3">
        <Button on:on:on:click={runAuthenticatedCRUDTest} disabled={isLoading || !isAuthenticated} variant="default">
          {isLoading ? '⏳ Testing...' : '🚀 Run Authenticated CRUD Test'}
        </Button>
        <Button on:on:on:click={() => checkAuth()} disabled={isLoading} variant="secondary">
          🔍 Check Auth
        </Button>
        <Button on:on:on:click={testAuthenticatedGET} disabled={isLoading || !isAuthenticated} variant="secondary">
          📋 Test GET
        </Button>
        <Button on:on:on:click={testAuthenticatedPOST} disabled={isLoading || !isAuthenticated} variant="secondary">
          📝 Test POST
        </Button>
        <Button on:on:on:click={() => testAuthenticatedPUT()} disabled={isLoading || !isAuthenticated} variant="secondary">
          ✏️ Test PUT
        </Button>
        <Button on:on:on:click={() => testAuthenticatedDELETE()} disabled={isLoading || !isAuthenticated} variant="secondary">
          🗑️ Test DELETE
        </Button>
        <Button on:on:on:click={clearResults} variant="ghost">
          🧹 Clear Results
        </Button>
      </div>
      
      {#if !isAuthenticated}
        <p class="text-sm text-gray-500 mt-2">
          ⚠️ Authentication required to run tests. Please log in first.
        </p>
      {/if}
    </CardContent>
  </Card>

  <!-- Test Results Summary -->
  <Card>
    <CardHeader>
      <CardTitle>Test Results Summary</CardTitle>
    </CardHeader>
    <CardContent>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div class="bg-blue-50 p-3 rounded">
          <div class="text-2xl font-bold text-blue-600">{testSummary.total}</div>
          <div class="text-sm text-blue-600">Total Tests</div>
        </div>
        <div class="bg-green-50 p-3 rounded">
          <div class="text-2xl font-bold text-green-600">{testSummary.passed}</div>
          <div class="text-sm text-green-600">Passed</div>
        </div>
        <div class="bg-red-50 p-3 rounded">
          <div class="text-2xl font-bold text-red-600">{testSummary.failed}</div>
          <div class="text-sm text-red-600">Failed</div>
        </div>
        <div class="bg-yellow-50 p-3 rounded">
          <div class="text-2xl font-bold text-yellow-600">{testSummary.warnings}</div>
          <div class="text-sm text-yellow-600">Warnings</div>
        </div>
        <div class="bg-purple-50 p-3 rounded">
          <div class="text-2xl font-bold text-purple-600">{testSummary.successRate}%</div>
          <div class="text-sm text-purple-600">Success Rate</div>
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- Current User's Cases -->
  <Card>
    <CardHeader>
      <CardTitle>Your Cases ({cases.length})</CardTitle>
    </CardHeader>
    <CardContent>
      {#if !isAuthenticated}
        <p class="text-gray-500">Login required to view your cases</p>
      {:else if cases.length === 0}
        <p class="text-gray-500">No cases found. Create some test cases to see them here.</p>
      {:else}
        <div class="space-y-2">
          {#each cases as caseItem}
            <div class="border rounded p-3 hover:bg-gray-50">
              <div class="flex justify-between items-start">
                <div>
                  <h4 class="font-semibold">{caseItem.title}</h4>
                  <p class="text-sm text-gray-600">Case #: {caseItem.caseNumber}</p>
                  <p class="text-sm text-gray-600">
                    Status: {caseItem.status} | Priority: {caseItem.priority}
                  </p>
                  <p class="text-sm text-gray-500">
                    Created: {new Date(caseItem.createdAt).toLocaleDateString()}
                  </p>
                  {#if caseItem.metadata?.embedding}
                    <span class="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded mt-1">
                      🧠 Has Embedding
                    </span>
                  {/if}
                </div>
                <div class="text-xs text-gray-500">
                  ID: {caseItem.id.substring(0, 8)}...
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </CardContent>
  </Card>

  <!-- Test Results Log -->
  <Card>
    <CardHeader>
      <CardTitle>Test Results Log</CardTitle>
    </CardHeader>
    <CardContent>
      {#if testResults.length === 0}
        <p class="text-gray-500">No test results yet. Run some tests to see results here.</p>
      {:else}
        <div class="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm max-h-96 overflow-y-auto space-y-1">
          {#each testResults as result}
            <div class={
              result.includes('❌') ? 'text-red-400' : 
              result.includes('⚠️') ? 'text-yellow-400' : 
              result.includes('✅') ? 'text-green-400' : 
              'text-gray-300'
            }>
              {result}
            </div>
          {/each}
        </div>
      {/if}
    </CardContent>
  </Card>
</div>