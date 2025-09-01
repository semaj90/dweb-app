<script lang="ts">
  interface Props {
    loading?: boolean;
    error?: string | null;
    empty?: boolean;
    emptyMessage?: string;
    loadingMessage?: string;
    skeleton?: 'card' | 'list' | 'table' | 'dashboard' | 'custom';
    children?: any;
    fallback?: any;
  }
  
  let { 
    loading = false,
    error = null,
    empty = false,
    emptyMessage = 'No data available',
    loadingMessage = 'Loading...',
    skeleton = 'card',
    children,
    fallback
  }: Props = $props();
</script>

<!-- Error State -->
{#if error}
  <div class="flex flex-col items-center justify-center p-golden-xl text-center">
    <div class="w-16 h-16 mb-golden-lg bg-red-500/20 rounded-full flex items-center justify-center">
      <svg class="w-8 h-8 text-red-400" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
      </svg>
    </div>
    <h3 class="text-lg font-bold text-red-400 uppercase tracking-wide mb-golden-sm">
      Error Loading Data
    </h3>
    <p class="text-nier-text-secondary max-w-md">
      {error}
    </p>
  </div>

<!-- Loading State -->
{:else if loading}
  <div class="space-y-golden-lg">
    <!-- Loading Header -->
    <div class="flex items-center justify-center p-golden-lg">
      <div class="flex items-center gap-golden-sm">
        <!-- YoRHa Loading Spinner -->
        <div class="relative">
          <div class="w-8 h-8 border-2 border-nier-accent-warm border-t-transparent rounded-full animate-spin"></div>
          <div class="absolute inset-1 border-2 border-nier-accent-cool border-b-transparent rounded-full animate-spin animation-delay-150"></div>
        </div>
        <span class="text-nier-text-secondary font-mono uppercase tracking-wide">
          {loadingMessage}
        </span>
      </div>
    </div>
    
    <!-- Skeleton Content -->
    {#if skeleton === 'dashboard'}
      <!-- Dashboard Skeleton -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-golden-lg">
        {#each Array(4) as _}
          <div class="bg-nier-bg-secondary border border-nier-border-muted rounded p-golden-lg animate-pulse">
            <div class="space-y-golden-sm">
              <div class="h-8 w-16 bg-nier-bg-tertiary rounded"></div>
              <div class="h-4 w-24 bg-nier-bg-tertiary rounded"></div>
            </div>
          </div>
        {/each}
      </div>
      
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-golden-xl">
        {#each Array(3) as _}
          <div class="bg-nier-bg-secondary border border-nier-border-muted rounded p-golden-lg animate-pulse">
            <div class="space-y-golden-sm">
              <div class="h-6 w-32 bg-nier-bg-tertiary rounded"></div>
              <div class="space-y-golden-xs">
                {#each Array(5) as _}
                  <div class="h-4 bg-nier-bg-tertiary rounded"></div>
                {/each}
              </div>
            </div>
          </div>
        {/each}
      </div>
      
    {:else if skeleton === 'list'}
      <!-- List Skeleton -->
      <div class="space-y-golden-sm">
        {#each Array(6) as _}
          <div class="bg-nier-bg-secondary border border-nier-border-muted rounded p-golden-md animate-pulse">
            <div class="flex items-center gap-golden-sm">
              <div class="w-10 h-10 bg-nier-bg-tertiary rounded-full"></div>
              <div class="flex-1 space-y-golden-xs">
                <div class="h-4 bg-nier-bg-tertiary rounded w-3/4"></div>
                <div class="h-3 bg-nier-bg-tertiary rounded w-1/2"></div>
              </div>
              <div class="w-16 h-6 bg-nier-bg-tertiary rounded"></div>
            </div>
          </div>
        {/each}
      </div>
      
    {:else if skeleton === 'table'}
      <!-- Table Skeleton -->
      <div class="bg-nier-bg-secondary border border-nier-border-muted rounded overflow-hidden">
        <div class="p-golden-md border-b border-nier-border-muted animate-pulse">
          <div class="grid grid-cols-4 gap-golden-sm">
            {#each Array(4) as _}
              <div class="h-4 bg-nier-bg-tertiary rounded"></div>
            {/each}
          </div>
        </div>
        {#each Array(8) as _}
          <div class="p-golden-md border-b border-nier-border-muted animate-pulse">
            <div class="grid grid-cols-4 gap-golden-sm">
              {#each Array(4) as _}
                <div class="h-4 bg-nier-bg-tertiary rounded"></div>
              {/each}
            </div>
          </div>
        {/each}
      </div>
      
    {:else if skeleton === 'card'}
      <!-- Card Skeleton -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-golden-lg">
        {#each Array(6) as _}
          <div class="bg-nier-bg-secondary border border-nier-border-muted rounded p-golden-lg animate-pulse">
            <div class="space-y-golden-sm">
              <div class="h-6 bg-nier-bg-tertiary rounded w-3/4"></div>
              <div class="space-y-golden-xs">
                <div class="h-4 bg-nier-bg-tertiary rounded"></div>
                <div class="h-4 bg-nier-bg-tertiary rounded w-5/6"></div>
                <div class="h-4 bg-nier-bg-tertiary rounded w-4/6"></div>
              </div>
              <div class="flex justify-between items-center pt-golden-sm">
                <div class="h-6 w-16 bg-nier-bg-tertiary rounded"></div>
                <div class="h-8 w-20 bg-nier-bg-tertiary rounded"></div>
              </div>
            </div>
          </div>
        {/each}
      </div>
      
    {:else if skeleton === 'custom' && fallback}
      {@render fallback()}
    {/if}
  </div>

<!-- Empty State -->
{:else if empty}
  <div class="flex flex-col items-center justify-center p-golden-xl text-center">
    <div class="w-16 h-16 mb-golden-lg bg-nier-bg-tertiary rounded-full flex items-center justify-center">
      <svg class="w-8 h-8 text-nier-text-muted" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm0 4a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1V8zm8 0a1 1 0 011-1h6a1 1 0 011 1v2a1 1 0 01-1 1h-6a1 1 0 01-1-1V8zm0 4a1 1 0 011-1h6a1 1 0 011 1v2a1 1 0 01-1 1h-6a1 1 0 01-1-1v-2z" clip-rule="evenodd" />
      </svg>
    </div>
    <h3 class="text-lg font-bold text-nier-text-primary uppercase tracking-wide mb-golden-sm">
      No Data Available
    </h3>
    <p class="text-nier-text-secondary max-w-md">
      {emptyMessage}
    </p>
  </div>

<!-- Content State -->
{:else}
  {@render children?.()}
{/if}

<style>
  .animation-delay-150 {
    animation-delay: 150ms;
  }
  
  /* Custom YoRHa pulse animation */
  @keyframes yorha-pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.3;
    }
  }
  
  .animate-pulse {
    animation: yorha-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  
  /* Staggered animation for skeleton items */
  .animate-pulse:nth-child(1) { animation-delay: 0ms; }
  .animate-pulse:nth-child(2) { animation-delay: 100ms; }
  .animate-pulse:nth-child(3) { animation-delay: 200ms; }
  .animate-pulse:nth-child(4) { animation-delay: 300ms; }
  .animate-pulse:nth-child(5) { animation-delay: 400ms; }
  .animate-pulse:nth-child(6) { animation-delay: 500ms; }
</style>