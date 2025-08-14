<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  
  let report: any = null;
  let loading = false;
  let error = '';
  let storedReports: any[] = [];
  
  async function generateBestPractices() {
    loading = true;
    error = '';
    
    try {
      const response = await fetch('/api/generate-best-practices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          projectPath: '.',
          options: {}
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        report = result.data.report;
        await loadStoredReports();
      } else {
        error = result.error?.message || 'Generation failed';
      }
    } catch (err) {
      error = `Request failed: ${err.message}`;
    } finally {
      loading = false;
    }
  }
  
  async function loadStoredReports() {
    try {
      const response = await fetch('/api/generate-best-practices?limit=5');
      const result = await response.json();
      
      if (result.success) {
        storedReports = result.data.reports;
      }
    } catch (err) {
      console.warn('Failed to load stored reports:', err);
    }
  }
  
  onMount(() => {
    loadStoredReports();
  });
  
  function getPriorityColor(priority: string) {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  }
  
  function getCategoryIcon(category: string) {
    switch (category) {
      case 'security': return '🔒';
      case 'performance': return '⚡';
      case 'maintainability': return '🔧';
      case 'testing': return '🧪';
      case 'architecture': return '🏗️';
      case 'accessibility': return '♿';
      default: return '📝';
    }
  }
</script>

<svelte:head>
  <title>Best Practices Generator | Legal AI</title>
  <meta name="description" content="Generate actionable best practices for your codebase" />
</svelte:head>

<div class="min-h-screen bg-gray-50 py-8">
  <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900">Best Practices Generator</h1>
      <p class="mt-2 text-gray-600">
        Generate actionable best practices and recommendations for your codebase
      </p>
    </div>

    <!-- Generate Section -->
    <div class="mb-8 bg-white rounded-lg shadow p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-semibold text-gray-900">Generate New Report</h2>
        <button
          on:click={generateBestPractices}
          disabled={loading}
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if loading}
            <svg class="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Generating...
          {:else}
            🔍 Analyze Codebase
          {/if}
        </button>
      </div>
      
      {#if error}
        <div class="bg-red-50 border border-red-200 rounded-md p-4">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <p class="text-sm text-red-800">{error}</p>
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- Generated Report -->
    {#if report}
      <div class="bg-white rounded-lg shadow mb-8">
        <!-- Report Header -->
        <div class="px-6 py-4 border-b border-gray-200">
          <div class="flex items-center justify-between">
            <div>
              <h2 class="text-xl font-semibold text-gray-900">Latest Analysis Report</h2>
              <p class="text-sm text-gray-500">Generated on {new Date(report.timestamp).toLocaleString()}</p>
            </div>
            <div class="flex space-x-4 text-sm">
              <div class="text-center">
                <div class="text-2xl font-bold text-blue-600">{report.best_practices.length}</div>
                <div class="text-gray-500">Total Practices</div>
              </div>
              <div class="text-center">
                <div class="text-2xl font-bold text-red-600">{report.summary.high_priority_count}</div>
                <div class="text-gray-500">High Priority</div>
              </div>
              <div class="text-center">
                <div class="text-2xl font-bold text-green-600">{report.summary.quick_wins.length}</div>
                <div class="text-gray-500">Quick Wins</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Codebase Analysis Summary -->
        <div class="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h3 class="text-lg font-medium text-gray-900 mb-3">Codebase Analysis</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-white rounded p-3">
              <div class="text-sm text-gray-500">Project Type</div>
              <div class="font-medium">{report.codebase_analysis.project_type}</div>
            </div>
            <div class="bg-white rounded p-3">
              <div class="text-sm text-gray-500">Files</div>
              <div class="font-medium">{report.codebase_analysis.file_count.toLocaleString()}</div>
            </div>
            <div class="bg-white rounded p-3">
              <div class="text-sm text-gray-500">Lines of Code</div>
              <div class="font-medium">{report.codebase_analysis.lines_of_code.toLocaleString()}</div>
            </div>
            <div class="bg-white rounded p-3">
              <div class="text-sm text-gray-500">Architecture</div>
              <div class="font-medium">{report.codebase_analysis.architecture_style}</div>
            </div>
          </div>
          
          {#if report.codebase_analysis.technologies?.length}
            <div class="mt-4">
              <div class="text-sm text-gray-500 mb-2">Technologies</div>
              <div class="flex flex-wrap gap-2">
                {#each report.codebase_analysis.technologies as tech}
                  <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {tech}
                  </span>
                {/each}
              </div>
            </div>
          {/if}
        </div>

        <!-- Best Practices -->
        <div class="px-6 py-4">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Best Practices Recommendations</h3>
          <div class="space-y-6">
            {#each report.best_practices as practice}
              <div class="border border-gray-200 rounded-lg p-6">
                <div class="flex items-start justify-between mb-4">
                  <div class="flex items-center space-x-3">
                    <span class="text-2xl">{getCategoryIcon(practice.category)}</span>
                    <div>
                      <h4 class="text-lg font-medium text-gray-900">{practice.title}</h4>
                      <div class="flex items-center space-x-2 mt-1">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border {getPriorityColor(practice.priority)}">
                          {practice.priority.toUpperCase()}
                        </span>
                        <span class="text-sm text-gray-500 capitalize">{practice.category}</span>
                      </div>
                    </div>
                  </div>
                  <div class="flex items-center space-x-4 text-xs text-gray-500">
                    <div class="text-center">
                      <div class="font-medium">Effort</div>
                      <div class="capitalize">{practice.estimated_effort}</div>
                    </div>
                    <div class="text-center">
                      <div class="font-medium">Impact</div>
                      <div class="capitalize">{practice.impact}</div>
                    </div>
                  </div>
                </div>
                
                <p class="text-gray-600 mb-4">{practice.description}</p>
                
                {#if practice.rationale}
                  <div class="bg-blue-50 rounded-md p-3 mb-4">
                    <div class="text-sm font-medium text-blue-800">Why this matters:</div>
                    <div class="text-sm text-blue-700 mt-1">{practice.rationale}</div>
                  </div>
                {/if}
                
                {#if practice.actionable_steps?.length}
                  <div class="mb-4">
                    <div class="text-sm font-medium text-gray-900 mb-2">Action Steps:</div>
                    <ol class="list-decimal list-inside space-y-1 text-sm text-gray-700">
                      {#each practice.actionable_steps as step}
                        <li>{step}</li>
                      {/each}
                    </ol>
                  </div>
                {/if}
                
                {#if practice.examples?.good || practice.examples?.bad}
                  <div class="grid md:grid-cols-2 gap-4">
                    {#if practice.examples.good}
                      <div>
                        <div class="text-sm font-medium text-green-800 mb-2">✅ Good Example:</div>
                        <pre class="bg-green-50 border border-green-200 rounded p-3 text-xs text-green-800 overflow-x-auto">{practice.examples.good}</pre>
                      </div>
                    {/if}
                    {#if practice.examples.bad}
                      <div>
                        <div class="text-sm font-medium text-red-800 mb-2">❌ Avoid:</div>
                        <pre class="bg-red-50 border border-red-200 rounded p-3 text-xs text-red-800 overflow-x-auto">{practice.examples.bad}</pre>
                      </div>
                    {/if}
                  </div>
                {/if}
                
                {#if practice.resources?.length}
                  <div class="mt-4 pt-4 border-t border-gray-200">
                    <div class="text-sm font-medium text-gray-900 mb-2">Resources:</div>
                    <ul class="space-y-1">
                      {#each practice.resources as resource}
                        <li class="text-sm">
                          {#if resource.startsWith('http')}
                            <a href={resource} target="_blank" rel="noopener" class="text-blue-600 hover:text-blue-800">
                              {resource}
                            </a>
                          {:else}
                            <span class="text-gray-700">{resource}</span>
                          {/if}
                        </li>
                      {/each}
                    </ul>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}

    <!-- Stored Reports -->
    {#if storedReports.length > 0}
      <div class="bg-white rounded-lg shadow">
        <div class="px-6 py-4 border-b border-gray-200">
          <h2 class="text-xl font-semibold text-gray-900">Recent Reports</h2>
        </div>
        <div class="px-6 py-4">
          <div class="space-y-4">
            {#each storedReports as storedReport}
              <div class="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                <div>
                  <div class="font-medium text-gray-900">
                    {storedReport.codebase_analysis?.project_type || 'Analysis Report'}
                  </div>
                  <div class="text-sm text-gray-500">
                    {new Date(storedReport.timestamp).toLocaleString()} • 
                    {storedReport.best_practices?.length || 0} practices
                  </div>
                </div>
                <div class="flex items-center space-x-2 text-xs text-gray-500">
                  <span class="bg-red-100 text-red-800 px-2 py-1 rounded">
                    {storedReport.summary?.high_priority_count || 0} high priority
                  </span>
                  <span class="bg-green-100 text-green-800 px-2 py-1 rounded">
                    {storedReport.summary?.quick_wins?.length || 0} quick wins
                  </span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  pre {
    white-space: pre-wrap;
    word-wrap: break-word;
  }
</style>