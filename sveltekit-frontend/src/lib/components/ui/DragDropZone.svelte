<script lang="ts">

  import { AlertCircle, FileText, Image, Upload } from 'lucide-svelte';
  // runtime helpers ($props, $state, $derived, $effect, $bindable) are provided by the compiler in runes mode

  interface Props {
    accept?: string;
    multiple?: boolean;
    maxSize?: number;
    disabled?: boolean;
    dragActive?: boolean;
    onFilesDropped?: (files: File[]) => void;
    onFileHover?: (hovering: boolean) => void;
  }

  let {
    accept = '*/*',
    multiple = true,
    maxSize = 10 * 1024 * 1024, // 10MB default
    disabled = false,
    dragActive = $bindable(false),
    onFilesDropped,
    onFileHover
  }: Props = $props();

  let fileInput: HTMLInputElement;
  let isDragOver = $state(false);
  let errors = $state<string[]>([]);

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
    onFileHover?.(true);
}
  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;

    isDragOver = false;
    dragActive = false;
    onFileHover?.(false);
}
  function handleDrop(e: DragEvent) {
    e.preventDefault();
    if (disabled) return;

    isDragOver = false;
    dragActive = false;
    onFileHover?.(false);

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
      onFilesDropped?.(validFiles);
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
    return types.map(type => allowedTypes[type as keyof typeof allowedTypes] || allowedTypes['*/*']);
}
  function openFileDialog() {
    if (!disabled && fileInput) {
      fileInput.click();
}}
</script>

<div
  class="space-y-4"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
  onclick={openFileDialog}
  role="button"
  tabindex={0}
  onkeydown={(e: KeyboardEvent) => e.key === 'Enter' && openFileDialog()}
>
  <input
    bind:this={fileInput}
    type="file"
    {accept}
    {multiple}
    {disabled}
    onchange={handleFileSelect}
    class="space-y-4"
  />

  <div class="space-y-4">
    <div class="space-y-4">
      {#if isDragOver}
        <div class="space-y-4">
          <Upload class="space-y-4" />
        </div>
      {:else}
        <div class="space-y-4">
          <Upload class="space-y-4" />
        </div>
      {/if}
    </div>

    <div class="space-y-4">
      <p class="space-y-4">
        {isDragOver ? 'Drop files here' : 'Drag and drop files here'}
      </p>

      <p class="space-y-4">
        or <span class="space-y-4">browse files</span>
      </p>

      {#if accept !== '*/*'}
        <div class="space-y-4">
          {#each getAcceptedFileInfo() as { icon: Icon, label }}
            <div class="space-y-4">
              <Icon class="space-y-4" />
              {label}
            </div>
          {/each}
        </div>
      {/if}

      <p class="space-y-4">
        Max file size: {formatFileSize(maxSize)}
      </p>
    </div>
  </div>

  {#if errors.length > 0}
    <div class="space-y-4">
      {#each errors as error}
        <div class="space-y-4">
          <AlertCircle class="space-y-4" />
          {error}
        </div>
      {/each}
    </div>
  {/if}
</div>
