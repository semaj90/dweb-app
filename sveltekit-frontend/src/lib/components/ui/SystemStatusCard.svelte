<!-- System Status Card Component - YoRHa Theme -->
<script lang="ts">
  export let alert: {
    type: 'success' | 'warning' | 'error' | 'info';
    message: string;
    time: string;
  };

  const alertTypes = {
    success: { 
      icon: '✅', 
      bgClass: 'bg-green-800/20 border-green-600',
      textClass: 'text-green-400'
    },
    warning: { 
      icon: '⚠️', 
      bgClass: 'bg-yellow-800/20 border-yellow-600',
      textClass: 'text-yellow-400'
    },
    error: { 
      icon: '❌', 
      bgClass: 'bg-red-800/20 border-red-600',
      textClass: 'text-red-400'
    },
    info: { 
      icon: 'ℹ️', 
      bgClass: 'bg-blue-800/20 border-blue-600',
      textClass: 'text-blue-400'
    }
  };

  const alertConfig = alertTypes[alert.type];
</script>

<div class="p-3 border {alertConfig.bgClass} font-mono">
  <div class="flex items-start gap-3">
    <span class="text-sm">{alertConfig.icon}</span>
    <div class="flex-1">
      <p class="text-sm {alertConfig.textClass} leading-tight">{alert.message}</p>
      <p class="text-xs text-stone-500 mt-1">{alert.time}</p>
    </div>
  </div>
</div>

<style>
  div {
    font-family: 'Courier New', monospace;
  }
</style>