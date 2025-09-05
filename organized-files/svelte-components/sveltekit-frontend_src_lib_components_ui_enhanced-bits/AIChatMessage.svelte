<!-- AI Chat Message: Svelte 5, Bits UI, UnoCSS, analytics logging -->
<script lang="ts">
  interface Props {
    message: { role: string content: string timestamp?: string; references?: any[] };
  let {
    message,
    showReferences = false
  } = $props();

    showReferences?: any;
  }


    let { showReferences = $bindable() } = $props(); // false;

  let isUser = message.role === 'user';
</script>

<div class="flex gap-2 items-start mb-2" class:justify-end={isUser}>
  <div class="rounded-lg px-4 py-2 max-w-xl shadow text-sm"
    class:bg-blue-100={isUser}
    class:bg-gray-100={!isUser}
    class:text-right={isUser}
    class:text-left={!isUser}
  >
    <div>{message.content}</div>
    {#if showReferences && message.references?.length}
      <div class="mt-2 text-xs text-gray-400">
        References:
        <ul class="list-disc ml-4">
          {#each message.references as ref}
            <li>{ref.id} (score: {ref.score})</li>
          {/each}
        </ul>
      </div>
    {/if}
    {#if message.timestamp}
      <div class="mt-1 text-xs text-gray-300">{message.timestamp}</div>
    {/if}
  </div>
</div>
