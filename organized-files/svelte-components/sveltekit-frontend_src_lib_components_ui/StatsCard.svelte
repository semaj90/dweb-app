<script>
  export let value = 0;
  export let label = '';
</script>
<div class="stats-card">
  <div class="value">{value}</div>
  <div class="label">{label}</div>
</div>
<style>
  .stats-card{padding:.5rem;border:1px solid #eee;border-radius:6px}
  .value{font-weight:700}
</style>
