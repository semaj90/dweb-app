<script lang="ts">
  interface Props {
    onFileSelected?: (files: File[]) => void;
    accept?: string;
    multiple?: boolean;
  }

  let { onFileSelected = () => {}, accept = "*", multiple = false } = $props();

  let dragActive = false;
  let fileInput: HTMLInputElement

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragActive = false;

    if (e.dataTransfer?.files) {
      const files = Array.from(e.dataTransfer.files);
      onFileSelected(files);
    }
  }

  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    if (target.files) {
      const files = Array.from(target.files);
      onFileSelected(files);
    }
  }
</script>

<div
  class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors"
  class:border-blue-500={dragActive}
  class:bg-blue-50={dragActive}
  on:drop={handleDrop}
  on:dragover|preventDefault={() => dragActive = true}
  on:dragleave={() => dragActive = false}
>
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    on:change={handleFileSelect}
    class="hidden"
  />

  <div class="space-y-4">
    <div class="text-4xl">📁</div>
    <div>
      <p class="text-lg font-medium">Drop files here or click to browse</p>
      <p class="text-sm text-gray-500">Supports all file types</p>
    </div>
    <button
      on:click={() => fileInput.click()}
      class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
    >
      Select Files
    </button>
  </div>
</div>