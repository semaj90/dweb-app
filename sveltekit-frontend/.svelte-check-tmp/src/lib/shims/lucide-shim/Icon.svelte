<script lang="ts">
  export let name: string = '';
  export let className = '';
</script>

<!-- Minimal shim for lucide-svelte Icon used during Svelte 5 migration. This avoids $$props usage. -->
<svg class={className} aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor">
  <!-- Placeholder: real icons not required for build; keeps markup valid -->
  <circle cx="12" cy="12" r="10" stroke-width="2"></circle>
  <text x="12" y="16" text-anchor="middle" font-size="8">{name}</text>
</svg>
