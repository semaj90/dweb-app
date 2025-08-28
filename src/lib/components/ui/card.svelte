<script>
  let { 
    variant = 'default',
    className = '',
    header,
    children,
    footer 
  } = $props();
  
  const variants = {
    default: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
    elevated: 'bg-white dark:bg-gray-800 shadow-lg',
    outlined: 'border-2 border-gray-300 dark:border-gray-600',
    filled: 'bg-gray-100 dark:bg-gray-900'
  };
  
  let variantClass = $derived(variants[variant] || variants.default);
</script>

<div class="rounded-lg p-6 {variantClass} {className}">
  {#if header}
    <div class="card-header mb-4">
      {@render header()}
    </div>
  {/if}
  
  <div class="card-content">
    {#if children}
      {@render children()}
    {/if}
  </div>
  
  {#if footer}
    <div class="card-footer mt-4">
      {@render footer()}
    </div>
  {/if}
</div>

<style>
  .card-header {
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    padding-bottom: 1rem;
  }
  
  .card-footer {
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    padding-top: 1rem;
  }
</style>