<script lang="ts">
  let {
    class: className = '',
    children,
    ...restProps
  } = $props();
</script>

<div class="text-sm [&_p]:leading-relaxed {className}" {...restProps}>
  {#if children}
    {@render children()}
  {/if}
</div>
