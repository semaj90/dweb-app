<!-- Case Item Component - YoRHa Theme -->
<script lang="ts">
  export let caseData: {
    id: string;
    title: string;
    items: number;
    timeAgo: string;
    priority: 'low' | 'medium' | 'high' | 'critical';
    status: 'active' | 'pending' | 'completed' | 'archived';
  };

  // Status badges mapping
  const statusBadges = {
    active: { label: 'ACTIVE', class: 'bg-green-600 text-white' },
    pending: { label: 'PENDING', class: 'bg-yellow-600 text-white' },
    completed: { label: 'COMPLETED', class: 'bg-blue-600 text-white' },
    archived: { label: 'ARCHIVED', class: 'bg-stone-600 text-white' }
  };

  const priorityBadges = {
    low: { label: 'LOW', class: 'bg-stone-600 text-white' },
    medium: { label: 'MEDIUM', class: 'bg-yellow-500 text-black' },
    high: { label: 'HIGH', class: 'bg-red-600 text-white' },
    critical: { label: 'CRITICAL', class: 'bg-red-800 text-white animate-pulse' }
  };

  function handleCaseClick() {
    console.log('Opening case:', caseData.id);
    // Navigate to case details
  }
</script>

<button 
  on:click={handleCaseClick}
  class="w-full p-4 bg-stone-800/50 border border-stone-600 hover:bg-stone-700/50 transition-colors text-left"
>
  <div class="flex justify-between items-start mb-3">
    <div class="flex-1">
      <h3 class="font-mono text-stone-100 font-bold tracking-wider mb-1">{caseData.title}</h3>
      <p class="text-sm text-stone-400 font-mono">{caseData.id} • {caseData.items} items • {caseData.timeAgo}</p>
    </div>
    <div class="flex gap-2 ml-4">
      <span class="px-2 py-1 text-xs font-mono rounded {priorityBadges[caseData.priority].class}">
        {priorityBadges[caseData.priority].label}
      </span>
      <span class="px-2 py-1 text-xs font-mono rounded {statusBadges[caseData.status].class}">
        {statusBadges[caseData.status].label}
      </span>
    </div>
  </div>
  
  <!-- Progress or additional info could go here -->
  <div class="text-xs text-stone-500 font-mono">
    {#if caseData.status === 'active'}
      → Click to view case details
    {:else if caseData.status === 'pending'}
      → Awaiting review
    {:else}
      → Case archived
    {/if}
  </div>
</button>

<style>
  button {
    font-family: 'Courier New', monospace;
  }
</style>