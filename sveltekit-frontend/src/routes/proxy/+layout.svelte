<script lang="ts">
  // Svelte 5 layout for /proxy subtree.
  export let children: any;
</script>

<section class="container mx-auto px-6 py-10">
  <div class="mb-6">
    <h2 class="text-2xl font-semibold tracking-wide">Proxy Area</h2>
    <p class="text-sm text-neutral-400">Lightweight wrapper ensuring Svelte 5 layout compliance.</p>
  </div>
  {@render children?.()}
</section>
