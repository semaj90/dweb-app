<script lang="ts">
  export let trigger: any;
</script>

<div use:trigger {...$$restProps}>
  <slot />
</div>
