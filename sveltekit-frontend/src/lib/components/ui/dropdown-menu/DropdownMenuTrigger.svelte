<script lang="ts">
  // import { melt } from 'melt'; // Removed melt dependency

  interface Props {
    trigger: any;
    children?: import('svelte').Snippet;
    class?: string;
  }
  
  let { 
    children, 
    trigger,
    class: className = ''
  }: Props = $props();
</script>

<button 
  <!-- <!-- use:melt={trigger} -->
  class="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 {className}"
>
  {@render children?.()}
</button>
