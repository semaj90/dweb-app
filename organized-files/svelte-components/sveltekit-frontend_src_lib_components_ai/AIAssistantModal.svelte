<script>
  export let open = false;
  export let onClose = () => {};
</script>
{#if open}
  <div class="ai-modal">
    <div class="content">
      <h3>AI Assistant</h3>
      <button type="button" onclick={onClose}>Close</button>
    </div>
  </div>
{/if}
<style>
  .ai-modal{position: fixedinset:0;background:rgba(0,0,0,0.4);display: flexalign-items: centerjustify-content:center}
  .content{background: whitepadding:1rem;border-radius:8px}
</style>
