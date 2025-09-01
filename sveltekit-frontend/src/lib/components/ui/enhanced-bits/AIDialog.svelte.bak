<!-- AI Dialog: Svelte 5, Bits UI, UnoCSS, transitions, analytics logging -->
<script lang="ts">

  interface Props {
    class?: string;
    children?: import('svelte').Snippet;
  }
  import { fade, scale } from 'svelte/transition';
  let { open = $bindable() } = $props(); // false;
  let { title = $bindable() } = $props(); // '';
  let { onClose = $bindable() } = $props(); // () => void = () => {};
</script>

{#if open}
  <div class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" transition:fade>
    <div class="bg-white rounded-lg shadow-lg p-6 w-full max-w-md relative" transition:scale>
      <h2 class="font-bold text-lg mb-4">{title}</h2>
      <slot />
      <button class="absolute top-2 right-2 text-gray-400 hover:text-gray-700" click={onClose} aria-label="Close">✕</button>
    </div>
  </div>
{/if}
