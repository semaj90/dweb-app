<script lang="ts">
  interface Props {
    asChild?: any;
    builder: any ;
  }
  let {
    asChild = false,
    builder = null
  }: Props = $props();



    
  export const className = "";
</script>

{#if asChild}
  <slot {builder} />
{:else}
  <button type="button" {...$$restProps} use:builder>
    <slot />
  </button>
{/if}
