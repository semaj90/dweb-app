<script lang="ts">
  interface Props {
    value: string;;
  }
  let {
    value
  }: Props = $props();



  import { getContext } from 'svelte';

    let className: string = '';
  export { className as class };

  const { activeTab } = getContext('tabs') as any;
</script>

{#if $activeTab === value}
  <div class="mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 {className}">
    <slot />
  </div>
{/if}