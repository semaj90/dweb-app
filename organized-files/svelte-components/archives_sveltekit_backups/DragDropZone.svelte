<script lang="ts">
  import { AlertCircle, FileText, Image, Upload } from 'lucide-svelte';
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher<{
    filesDropped: File[];
    fileHover: boolean;
  }>();

  export let accept = '*/*';
  export let multiple = true;
  export let maxSize = 10 * 1024 * 1024; // 10MB default
  export let disabled = false;
  export let dragActive = false;

  let fileInput: HTMLInputElement;
  let isDragOver = false;
  let errors: string[] = [];

  const allowedTypes = {
    'image/*': { icon: Image, label: 'Images' },
    'application/pdf': { icon: FileText, label: 'PDF Documents' },
    'text/*': { icon: FileText, label: 'Text Files' },
    '*/*': { icon: Upload, label: 'Any File' }
  };

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;

    isDragOver = true;
    dragActive = true;
    dispatch('fileHover', true);
}
  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;

    isDragOver = false;
    dragActive = false;
    dispatch('fileHover', false);
}
  function handleDrop(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;

    isDragOver = false;
    dragActive = false;
    dispatch('fileHover', false);

    const files = Array.from(e.dataTransfer?.files || []);
    processFiles(files);
}
  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    const files = Array.from(target.files || []);
    processFiles(files);
}
  function processFiles(files: File[]) {
    errors = [];
    const validFiles: File[] = [];

    for (const file of files) {
      // Check file size
      if (file.size > maxSize) {
        errors.push(`${file.name} is too large (max ${formatFileSize(maxSize)})`);
        continue;
}
      // Check file type if accept is specified and not wildcard
      if (accept !== '*/*' && !matchesAcceptType(file.type, accept)) {
        errors.push(`${file.name} is not an accepted file type`);
        continue;
}
      validFiles.push(file);
}
    if (validFiles.length > 0) {
      dispatch('filesDropped', validFiles);
}
    // Clear input so same file can be selected again
    if (fileInput) {
      fileInput.value = '';
}}
  function matchesAcceptType(fileType: string, acceptString: string): boolean {
    const acceptTypes = acceptString.split(',').map(s => s.trim());
    return acceptTypes.some(acceptType => {
      if (acceptType === '*/*') return true;
      if (acceptType.endsWith('/*')) {
        const baseType = acceptType.slice(0, -2);
        return fileType.startsWith(baseType);
}
      return fileType === acceptType;
    });
}
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
  function getAcceptedFileInfo() {
    const types = accept.split(',').map(s => s.trim());
    return types.map(type => allowedTypes[type] || allowedTypes['*/*']);
}
  function openFileDialog() {
    if (!disabled && fileInput) {
      fileInput.click();
}}
</script>

<div
  class="mx-auto px-4 max-w-7xl"
  on:dragover={handleDragOver}
  on:dragleave={handleDragLeave}
  on:drop={handleDrop}
  onclick={() => openFileDialog()}
  role="button"
  tabindex={0}
  onkeydown={(e) => e.key === 'Enter' && openFileDialog()}
>
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    {disabled}
    onchange={handleFileSelect}
    class="mx-auto px-4 max-w-7xl"
  />

  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      {#if isDragOver}
        <div class="mx-auto px-4 max-w-7xl">
          <Upload class="mx-auto px-4 max-w-7xl" />
        </div>
      {:else}
        <div class="mx-auto px-4 max-w-7xl">
          <Upload class="mx-auto px-4 max-w-7xl" />
        </div>
      {/if}
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      <p class="mx-auto px-4 max-w-7xl">
        {isDragOver ? 'Drop files here' : 'Drag and drop files here'}
      </p>

      <p class="mx-auto px-4 max-w-7xl">
        or <span class="mx-auto px-4 max-w-7xl">browse files</span>
      </p>

      {#if accept !== '*/*'}
        <div class="mx-auto px-4 max-w-7xl">
          {#each getAcceptedFileInfo() as { icon: Icon, label "
            <div class="mx-auto px-4 max-w-7xl">
              <Icon class="mx-auto px-4 max-w-7xl" />
              {label}
            </div>
          {/each}
        </div>
      {/if}

      <p class="mx-auto px-4 max-w-7xl">
        Max file size: {formatFileSize(maxSize)}
      </p>
    </div>
  </div>

  {#if errors.length > 0}
    <div class="mx-auto px-4 max-w-7xl">
      {#each errors as error}
        <div class="mx-auto px-4 max-w-7xl">
          <AlertCircle class="mx-auto px-4 max-w-7xl" />
          {error}
        </div>
      {/each}
    </div>
  {/if}
</div>
