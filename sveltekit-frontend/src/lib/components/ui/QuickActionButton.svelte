<!-- Quick Action Button Component - YoRHa Theme -->
<script lang="ts">
  export let action: {
    label: string;
    icon: string;
    action: () => void;
  };

  // Icon mapping
  const iconMap: Record<string, string> = {
    folder: '📁',
    clock: '🕒',
    terminal: '💻',
    search: '🔍',
    settings: '⚙️',
    analysis: '📊'
  };

  const displayIcon = iconMap[action.icon] || action.icon;
</script>

<button 
  on:click={action.action}
  class="w-full p-3 bg-stone-800/30 border border-stone-600 hover:bg-stone-700/50 hover:border-stone-500 
         transition-all duration-200 font-mono text-left group"
>
  <div class="flex items-center gap-3">
    <span class="text-lg group-hover:scale-110 transition-transform duration-200">
      {displayIcon}
    </span>
    <span class="text-sm text-stone-300 group-hover:text-stone-100 tracking-wider">
      {action.label}
    </span>
    <span class="ml-auto text-stone-500 group-hover:text-stone-300 transition-colors">
      →
    </span>
  </div>
</button>

<style>
  button {
    font-family: 'Courier New', monospace;
  }
</style>