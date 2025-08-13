<script lang="ts">
  interface Props {
    overlay: any;
;
    content: any;
;
    openState: any;
;
    size: 'sm' | 'md' | 'lg' | 'xl' | 'full' ;
  }
  let {
    overlay,
    content,
    openState,
    size = 'md'
  }: Props = $props();



  import { fade, fly } from 'svelte/transition';

        export let size: 'sm' | 'md' | 'lg' | 'xl' | 'full' = 'md';

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-xl',
    full: 'max-w-[95vw] max-h-[95vh]'
  };
</script>

{#if $openState}
  <!-- Overlay -->
  <div
    use:overlay
    class="space-y-4"
    transition:fade={{ duration: 200 }}
  ></div>

  <!-- Content -->
  <div
    use:content
    class="space-y-4"
    transition:fly={{ y: -20, duration: 200  }}
    {...$$restProps}
  >
    <slot />
  </div>
{/if}
