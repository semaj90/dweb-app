<script lang="ts">
  // Local shim: sveltekit-superforms' SuperDebug has duplicate top-level <script> in dist build
  // Provide a no-op component during migration to avoid build-time parse errors.
  // Convert to Svelte 5 runes pattern
  let { debug = false } = $props<{
    debug?: boolean;
  }>();
</script>

<!-- Minimal visual placeholder -->
{#if debug}
  <div class="superdebug-shim">SuperDebug shim active</div>
{/if}
