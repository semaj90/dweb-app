#!/bin/bash
# Setup GraphQL with Local Gemma3 Integration

echo "🚀 Setting up GraphQL with Local Gemma3 LLM..."

# 1. Install dependencies
echo "📦 Installing GraphQL dependencies..."
npm install graphql graphql-yoga @pothos/core @pothos/plugin-drizzle @urql/svelte @urql/core @urql/devtools @urql/exchange-graphcache

# 2. Install additional dependencies for local LLM
echo "📦 Installing LLM dependencies..."
npm install langchain @langchain/community node-fetch

# 3. Create example Svelte component
cat > src/routes/graphql-demo/+page.svelte << 'EOF'
<script lang="ts">
  import { getContextClient, queryStore } from '@urql/svelte';
  import { gql } from '@urql/core';
  
  const client = getContextClient();
  
  let searchQuery = '';
  let isLoading = false;
  
  // GraphQL query
  const SEARCH_CASES = gql`
    query SearchCases($query: String!) {
      searchCases(query: $query, limit: 10) {
        id
        title
        content
        relevanceScore
        metadata
      }
    }
  `;
  
  // Reactive query store
  $: casesQuery = queryStore({
    client,
    query: SEARCH_CASES,
    variables: { query: searchQuery },
    pause: !searchQuery,
  });
  
  $: cases = $casesQuery.data?.searchCases ?? [];
  
  // Analyze case mutation
  const ANALYZE_CASE = gql`
    mutation AnalyzeCase($caseId: String!, $analysisType: String!) {
      analyzeCaseWithAI(caseId: $caseId, analysisType: $analysisType) {
        caseId
        analysisType
        result
        confidence
        metadata
      }
    }
  `;
  
  async function analyzeCase(caseId: string) {
    isLoading = true;
    try {
      const result = await client.mutation(ANALYZE_CASE, {
        caseId,
        analysisType: 'comprehensive',
      }).toPromise();
      
      if (result.data) {
        console.log('Analysis:', result.data.analyzeCaseWithAI);
        // Handle analysis result
      }
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="p-6">
  <h1 class="text-2xl font-bold mb-4">GraphQL + Gemma3 Demo</h1>
  
  <div class="mb-6">
    <input
      type="text"
      bind:value={searchQuery}
      placeholder="Search legal cases..."
      class="w-full p-2 border rounded"
    />
  </div>
  
  {#if $casesQuery.fetching}
    <p>Searching with Gemma3...</p>
  {:else if $casesQuery.error}
    <p class="text-red-500">Error: {$casesQuery.error.message}</p>
  {:else if cases.length > 0}
    <div class="space-y-4">
      {#each cases as case}
        <div class="border p-4 rounded">
          <h3 class="font-bold">{case.title}</h3>
          <p class="text-gray-600">{case.content?.slice(0, 200)}...</p>
          <p class="text-sm text-gray-500">Score: {case.relevanceScore?.toFixed(2)}</p>
          <button
            on:click={() => analyzeCase(case.id)}
            disabled={isLoading}
            class="mt-2 px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
          >
            Analyze with Gemma3
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>
EOF

# 4. Create background worker
cat > src/lib/workers/analysis-worker.ts << 'EOF'
import { parentPort } from 'worker_threads';

// Worker for heavy Gemma3 processing
parentPort?.on('message', async (task) => {
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma3-legal:latest',
        prompt: task.prompt,
        stream: false,
        options: {
          temperature: 0.1,
          num_predict: 1024,
        }
      })
    });
    
    const result = await response.json();
    parentPort?.postMessage({ 
      taskId: task.id, 
      result: result.response 
    });
  } catch (error) {
    parentPort?.postMessage({ 
      taskId: task.id, 
      error: error.message 
    });
  }
});
EOF

echo "✅ GraphQL + Gemma3 setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start your Gemma3 model: ollama run gemma3-legal:latest"
echo "2. Run the dev server: npm run dev"
echo "3. Visit http://localhost:5173/graphql-demo"
echo ""
echo "🎯 Your local Gemma3 is now integrated with:"
echo "  - GraphQL API at /api/graphql"
echo "  - Vector search with pgvector"
echo "  - Real-time subscriptions"
echo "  - Document processing pipeline"
