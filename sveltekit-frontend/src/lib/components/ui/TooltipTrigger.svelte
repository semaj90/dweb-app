<script lang="ts">
  export let asChild = false;
  export let builder: any = null;

  export const className = "";
</script>

{#if asChild}
  <slot {builder} />
{:else}
  <button type="button" {...$$restProps} use:builder>
    <slot />
  </button>
{/if}
