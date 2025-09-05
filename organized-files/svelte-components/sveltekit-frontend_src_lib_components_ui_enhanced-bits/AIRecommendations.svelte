<!-- AI Recommendations: Svelte 5, Bits UI, UnoCSS, analytics logging -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { UiCard as Card, UiCardHeader as CardHeader, UiCardTitle as CardTitle, UiCardContent as CardContent } from '../index.js';

  interface Props {
    userContext?: any;
    neo4jContext?: any;
    analyticsLog?: (event: any) => void;
    onRecommendations?: (results: any) => void;
  }

  const {
    userContext = {},
    neo4jContext = {},
    analyticsLog = () => {},
    onRecommendations = () => {}
  } = $props();

  let recommendations = $state<any[]>([]);
  let loading = $state(false);

  async function fetchRecommendations() {
    loading = true;
    analyticsLog({ event: 'ai_recommendations_requested', userContext, timestamp: Date.now() });
    try {
      const res = await fetch('/api/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userContext, neo4jContext })
      });
      const data = await res.json();
      recommendations = data.recommendations || [];
      analyticsLog({ event: 'ai_recommendations_result', count: recommendations.length, timestamp: Date.now() });
      onRecommendations(recommendations);
    } catch (error) {
      analyticsLog({ event: 'ai_recommendations_error', error: error.message, timestamp: Date.now() });
    } finally {
      loading = false;
    }
  }

  onMount(fetchRecommendations);
</script>

<Card class="w-full">
  <CardHeader>
    <CardTitle>AI Recommendations</CardTitle>
  </CardHeader>
  <CardContent>
    {#if loading}
      <div class="text-gray-500">Loading recommendations...</div>
    {:else if recommendations.length === 0}
      <div class="text-gray-400">No recommendations available.</div>
    {:else}
      <ul class="space-y-2">
        {#each recommendations as rec}
          <li class="border-b last:border-b-0 pb-2">
            <div class="font-semibold">{rec.title || rec.intent || rec.id}</div>
            <div class="text-xs text-gray-500">{rec.content || rec.description}</div>
            <div class="text-xs text-gray-400">Score: {rec.finalScore ?? rec.rerankScore}</div>
          </li>
        {/each}
      </ul>
    {/if}
  </CardContent>
</Card>
