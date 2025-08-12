<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  // Props
  let { recommendations = [] } = $props();
  
  const dispatch = createEventDispatcher();
  
  // Priority ordering
  const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
  
  // Sort recommendations by priority and confidence
  let sortedRecommendations = $derived(
    recommendations.sort((a, b) => {
      const priorityDiff = (priorityOrder[b.priority] || 0) - (priorityOrder[a.priority] || 0);
      if (priorityDiff !== 0) return priorityDiff;
      return (b.confidence || 0) - (a.confidence || 0);
    })
  );
  
  // Get priority styling
  function getPriorityStyle(priority: string) {
    switch (priority) {
      case 'critical':
        return 'bg-red-500/20 border-red-500/50 text-red-300';
      case 'high':
        return 'bg-orange-500/20 border-orange-500/50 text-orange-300';
      case 'medium':
        return 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300';
      case 'low':
        return 'bg-blue-500/20 border-blue-500/50 text-blue-300';
      default:
        return 'bg-gray-500/20 border-gray-500/50 text-gray-300';
    }
  }
  
  function getPriorityIcon(priority: string) {
    switch (priority) {
      case 'critical': return '🚨';
      case 'high': return '⚠️';
      case 'medium': return '💡';
      case 'low': return 'ℹ️';
      default: return '📝';
    }
  }
  
  function getTypeIcon(type: string) {
    switch (type) {
      case 'performance': return '⚡';
      case 'security': return '🔒';
      case 'optimization': return '🎯';
      case 'system': return '🖥️';
      case 'model': return '🤖';
      case 'feature': return '✨';
      case 'workflow': return '🔄';
      case 'content': return '📄';
      default: return '🔧';
    }
  }
  
  function handleAccept(recommendation: any) {
    dispatch('accept-recommendation', recommendation);
  }
  
  function handleDismiss(recommendation: any) {
    dispatch('dismiss-recommendation', recommendation);
  }
  
  function handleViewDetails(recommendation: any) {
    dispatch('view-details', recommendation);
  }
</script>

{#if sortedRecommendations.length === 0}
  <div class="text-center py-8">
    <div class="text-6xl mb-4">✨</div>
    <p class="text-slate-400">No recommendations at this time</p>
    <p class="text-sm text-slate-500 mt-2">Your system is running optimally</p>
  </div>
{:else}
  <div class="space-y-4">
    {#each sortedRecommendations as recommendation, index}
      <div class="border border-slate-600 rounded-lg p-4 bg-slate-700/30 hover:bg-slate-700/50 transition-colors">
        <!-- Header -->
        <div class="flex items-start justify-between mb-3">
          <div class="flex items-center gap-3">
            <div class="text-2xl">{getTypeIcon(recommendation.type)}</div>
            <div>
              <div class="flex items-center gap-2">
                <h4 class="text-white font-medium">{recommendation.title}</h4>
                <span class="px-2 py-1 rounded-full text-xs border {getPriorityStyle(recommendation.priority)}">
                  {getPriorityIcon(recommendation.priority)} {recommendation.priority}
                </span>
              </div>
              {#if recommendation.category}
                <p class="text-xs text-slate-400 mt-1">{recommendation.category}</p>
              {/if}
            </div>
          </div>
          
          <!-- Confidence Score -->
          {#if recommendation.confidence}
            <div class="text-right">
              <div class="text-sm font-medium text-white">
                {Math.round(recommendation.confidence * 100)}%
              </div>
              <div class="text-xs text-slate-400">confidence</div>
            </div>
          {/if}
        </div>
        
        <!-- Description -->
        <p class="text-slate-300 text-sm mb-4">
          {recommendation.description}
        </p>
        
        <!-- Reasoning (if available) -->
        {#if recommendation.reasoning}
          <div class="bg-slate-600/30 border border-slate-600 rounded-lg p-3 mb-4">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-sm">🧠</span>
              <span class="text-xs font-medium text-slate-300">AI Reasoning</span>
            </div>
            <p class="text-xs text-slate-400">{recommendation.reasoning}</p>
          </div>
        {/if}
        
        <!-- Action Steps -->
        {#if recommendation.actionableSteps && recommendation.actionableSteps.length > 0}
          <div class="mb-4">
            <div class="text-sm font-medium text-slate-300 mb-2">📋 Action Steps:</div>
            <ol class="list-decimal list-inside space-y-1">
              {#each recommendation.actionableSteps as step}
                <li class="text-xs text-slate-400">{step}</li>
              {/each}
            </ol>
          </div>
        {/if}
        
        <!-- Expected Improvement -->
        {#if recommendation.expectedImprovement}
          <div class="mb-4">
            <div class="flex items-center gap-2">
              <span class="text-sm">📈</span>
              <span class="text-sm text-slate-300">Expected Improvement:</span>
              <span class="text-sm font-medium text-green-400">
                +{Math.round(recommendation.expectedImprovement * 100)}%
              </span>
            </div>
          </div>
        {/if}
        
        <!-- Tags -->
        {#if recommendation.tags && recommendation.tags.length > 0}
          <div class="flex flex-wrap gap-1 mb-4">
            {#each recommendation.tags as tag}
              <span class="px-2 py-1 bg-slate-600 text-slate-300 rounded-full text-xs">
                #{tag}
              </span>
            {/each}
          </div>
        {/if}
        
        <!-- Actions -->
        <div class="flex items-center gap-2">
          <button
            onclick={() => handleAccept(recommendation)}
            class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-lg transition-colors"
          >
            ✅ Accept
          </button>
          
          <button
            onclick={() => handleViewDetails(recommendation)}
            class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
          >
            👁️ Details
          </button>
          
          <button
            onclick={() => handleDismiss(recommendation)}
            class="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-slate-300 text-sm rounded-lg transition-colors"
          >
            ❌ Dismiss
          </button>
          
          <!-- Learn More Link -->
          {#if recommendation.metadata?.learnMoreUrl}
            <a
              href={recommendation.metadata.learnMoreUrl}
              target="_blank"
              rel="noopener noreferrer"
              class="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors"
            >
              📚 Learn More
            </a>
          {/if}
        </div>
      </div>
    {/each}
  </div>
  
  <!-- Summary Stats -->
  <div class="mt-6 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
    <h4 class="text-white font-medium mb-3">📊 Recommendation Summary</h4>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
      <div>
        <div class="text-2xl font-bold text-red-400">
          {sortedRecommendations.filter(r => r.priority === 'critical').length}
        </div>
        <div class="text-xs text-slate-400">Critical</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-orange-400">
          {sortedRecommendations.filter(r => r.priority === 'high').length}
        </div>
        <div class="text-xs text-slate-400">High</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-yellow-400">
          {sortedRecommendations.filter(r => r.priority === 'medium').length}
        </div>
        <div class="text-xs text-slate-400">Medium</div>
      </div>
      <div>
        <div class="text-2xl font-bold text-blue-400">
          {sortedRecommendations.filter(r => r.priority === 'low').length}
        </div>
        <div class="text-xs text-slate-400">Low</div>
      </div>
    </div>
    
    <!-- Average Confidence -->
    {#if sortedRecommendations.some(r => r.confidence)}
      <div class="mt-4 text-center">
        <div class="text-lg font-bold text-white">
          {Math.round(sortedRecommendations.reduce((sum, r) => sum + (r.confidence || 0), 0) / sortedRecommendations.length * 100)}%
        </div>
        <div class="text-xs text-slate-400">Average Confidence</div>
      </div>
    {/if}
  </div>
{/if}

<style>
  /* Smooth transitions for all interactive elements */
  button, a {
    transition: all 0.2s ease;
  }
  
  /* Hover animations */
  .hover\:bg-green-700:hover,
  .hover\:bg-blue-700:hover,
  .hover\:bg-slate-700:hover,
  .hover\:bg-purple-700:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }
</style>