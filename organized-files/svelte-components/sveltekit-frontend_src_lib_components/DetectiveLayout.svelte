<script>
  export let title = 'Detective Dashboard';
</script>

<div class="detective-layout">
  <header><h1>{title}</h1></header>
  <main>
    <slot />
  </main>
  <style>
    .detective-layout{padding:1rem}
    header{margin-bottom:1rem}
  </style>
</div>
