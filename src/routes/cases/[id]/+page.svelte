<script lang="ts">
  import type { PageData } from './$types';
  
  export let data: PageData;
  
  $: ({ user, case: caseData, evidence, report } = data);
</script>

<svelte:head>
  <title>Case {caseData.caseNumber} - Legal AI</title>
</svelte:head>

<div class="container mx-auto p-6">
  <div class="mb-6">
    <h1 class="text-3xl font-bold text-gray-900 mb-2">{caseData.title}</h1>
    <p class="text-gray-600">Case #{caseData.caseNumber}</p>
    <div class="flex items-center gap-4 mt-2">
      <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-md text-sm">
        {caseData.status}
      </span>
      <span class="text-sm text-gray-500">
        Created: {new Date(caseData.createdAt).toLocaleDateString()}
      </span>
    </div>
  </div>

  {#if caseData.description}
    <div class="mb-6">
      <h2 class="text-xl font-semibold mb-2">Description</h2>
      <p class="text-gray-700">{caseData.description}</p>
    </div>
  {/if}

  <div class="grid md:grid-cols-2 gap-6">
    <!-- Evidence Section -->
    <div>
      <h2 class="text-xl font-semibold mb-4">Evidence ({evidence.length})</h2>
      {#if evidence.length > 0}
        <div class="space-y-3">
          {#each evidence as item}
            <div class="border rounded-lg p-4">
              <h3 class="font-medium">{item.filename}</h3>
              <p class="text-sm text-gray-600 mt-1">{item.fileType}</p>
              {#if item.tags && item.tags.length > 0}
                <div class="flex flex-wrap gap-1 mt-2">
                  {#each item.tags as tag}
                    <span class="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs">
                      {tag}
                    </span>
                  {/each}
                </div>
              {/if}
              <p class="text-xs text-gray-500 mt-2">
                Added: {new Date(item.createdAt).toLocaleDateString()}
              </p>
            </div>
          {/each}
        </div>
      {:else}
        <p class="text-gray-500">No evidence uploaded yet.</p>
      {/if}
    </div>

    <!-- Report Section -->
    <div>
      <h2 class="text-xl font-semibold mb-4">Report</h2>
      {#if report}
        <div class="border rounded-lg p-4">
          <h3 class="font-medium mb-2">{report.title}</h3>
          {#if report.summary}
            <div class="mb-3">
              <h4 class="font-medium text-sm text-gray-700 mb-1">Summary</h4>
              <p class="text-sm text-gray-600">{report.summary}</p>
            </div>
          {/if}
          <p class="text-xs text-gray-500">
            Last updated: {new Date(report.updatedAt).toLocaleDateString()}
          </p>
        </div>
      {:else}
        <p class="text-gray-500">No report generated yet.</p>
        <button class="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
          Generate Report
        </button>
      {/if}
    </div>
  </div>
</div>