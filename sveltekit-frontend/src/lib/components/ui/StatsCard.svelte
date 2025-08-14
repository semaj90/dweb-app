<!-- Statistics Card Component - YoRHa Theme -->
<script lang="ts">
  export let title: string;
  export let value: number | string;
  export let icon: string = '';
  export let className: string = '';

  // Icon mapping
  const iconMap: Record<string, string> = {
    folder: '📁',
    box: '📦',
    users: '👥',
    activity: '📈',
    clock: '🕒',
    check: '✅',
    warning: '⚠️',
    error: '❌'
  };

  const displayIcon = iconMap[icon] || icon;
</script>

<div class="p-4 border border-stone-600 bg-stone-800/50 {className}">
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-3">
      {#if displayIcon}
        <span class="text-2xl opacity-70">{displayIcon}</span>
      {/if}
      <div>
        <p class="text-sm text-stone-400 font-mono uppercase tracking-wider">{title}</p>
        <p class="text-2xl font-mono font-bold text-stone-100">{value}</p>
      </div>
    </div>
  </div>
</div>

<style>
  div {
    font-family: 'Courier New', monospace;
  }
</style>