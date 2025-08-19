<script lang="ts">
  interface Props {
    class?: string;
  }
  
  let {
    class: className = '',
    children
  }: Props & { children?: any } = $props();

  const classes = $derived(className);
</script>

<div class="space-y-4">
  {#if children}
    {@render children()}
  {/if}
</div>
