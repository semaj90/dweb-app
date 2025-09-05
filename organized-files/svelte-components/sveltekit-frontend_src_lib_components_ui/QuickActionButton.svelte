<script>
  export let label = 'Action';
  export let onClick = () => {};
</script>
<button type="button" class="quick-action" on:click={onClick}>{label}</button>
<style>
  .quick-action{padding:.5rem 1rem;border-radius:6px;border:1px solid #ccc;background:#f9f9f9}
</style>
