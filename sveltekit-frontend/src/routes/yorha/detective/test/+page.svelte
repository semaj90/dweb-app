<!-- Test page for YoRHa Detective functionality -->
<script>
  import { onMount } from 'svelte';
  
  let testResult = '';
  let isLoading = false;

  async function testCaseCreation() {
    isLoading = true;
    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          title: 'Test Case from YoRHa Detective',
          description: 'This is a test case created from the YoRHa Detective interface',
          priority: 'medium'
        })
      });

      const result = await response.json();
      
      if (response.ok) {
        testResult = `✅ Case created successfully!\nID: ${result.data.id}\nCase Number: ${result.data.caseNumber}\nTitle: ${result.data.title}`;
      } else {
        testResult = `❌ Error: ${result.error}\nDetails: ${JSON.stringify(result.details, null, 2)}`;
      }
    } catch (error) {
      testResult = `❌ Network error: ${error.message}`;
    } finally {
      isLoading = false;
    }
  }

  async function testCaseList() {
    isLoading = true;
    try {
      const response = await fetch('/api/cases');
      const result = await response.json();
      
      if (response.ok) {
        testResult = `✅ Cases retrieved successfully!\nTotal: ${result.data.length}\nFirst few cases:\n${JSON.stringify(result.data.slice(0, 3), null, 2)}`;
      } else {
        testResult = `❌ Error: ${result.error}`;
      }
    } catch (error) {
      testResult = `❌ Network error: ${error.message}`;
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="test-page p-8 bg-gray-900 text-green-400 min-h-screen font-mono">
  <h1 class="text-3xl font-bold mb-8 text-yellow-400">YoRHa Detective API Test</h1>
  
  <div class="space-y-4 mb-8">
    <button 
      class="px-4 py-2 bg-blue-600 text-white border border-blue-400 hover:bg-blue-700 transition-colors disabled:opacity-50"
      onclick={testCaseCreation}
      disabled={isLoading}
    >
      {isLoading ? 'Testing...' : 'Test Case Creation'}
    </button>
    
    <button 
      class="px-4 py-2 bg-green-600 text-white border border-green-400 hover:bg-green-700 transition-colors disabled:opacity-50"
      onclick={testCaseList}
      disabled={isLoading}
    >
      {isLoading ? 'Testing...' : 'Test Case Listing'}
    </button>
  </div>

  {#if testResult}
    <div class="bg-black p-4 border border-gray-600 rounded">
      <h3 class="text-lg font-bold mb-2 text-yellow-400">Test Result:</h3>
      <pre class="whitespace-pre-wrap text-sm">{testResult}</pre>
    </div>
  {/if}

  <div class="mt-8">
    <h2 class="text-xl font-bold mb-4 text-yellow-400">Route Info</h2>
    <ul class="space-y-2 text-sm">
      <li>✅ Test Page: <code>/yorha/detective/test</code></li>
      <li>🎯 Main Page: <code>/yorha/detective</code></li>
      <li>🔗 API Endpoint: <code>/api/cases</code></li>
      <li>📊 Database: PostgreSQL with Drizzle ORM</li>
    </ul>
  </div>
</div>