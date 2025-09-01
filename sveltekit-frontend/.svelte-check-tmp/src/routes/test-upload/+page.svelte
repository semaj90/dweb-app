<script lang="ts">
  import AdvancedFileUpload from "$lib/components/upload/AdvancedFileUpload.svelte";
  // $state runtime rune is provided globally

  interface UploadedFile {
    name: string;
    size: number;
    type: string;
    status?: string;
    url?: string;
  }

  let uploadedFiles: UploadedFile[] = $state([]);

  function handleFilesAdded(event: CustomEvent<{ files: File[] }>) {
    console.log("Files added:", event.detail.files);
  }
  function handleUploadComplete(event: CustomEvent<{ files: UploadedFile[] }>) {
    console.log("Upload complete:", event.detail.files);
    uploadedFiles = [...uploadedFiles, ...event.detail.files];
  }
  function handleFileRemoved(event: CustomEvent<{ fileId: string }>) {
    console.log("File removed:", event.detail.fileId);
  }
</script>

<svelte:head>
  <title>File Upload Test</title>
</svelte:head>

<div class="space-y-4">
  <h1>File Upload Test</h1>

  <div class="space-y-4">
    <AdvancedFileUpload
      multiple={true}
      accept="*/*"
      maxFileSize={50 * 1024 * 1024}
      maxFiles={5}
      enablePreview={true}
      enableDragDrop={true}
      enablePasteUpload={true}
      autoUpload={false}
      filesadded={handleFilesAdded}
      uploadcomplete={handleUploadComplete}
      fileremoved={handleFileRemoved}
    />
  </div>

  {#if uploadedFiles.length > 0}
    <div class="space-y-4">
      <h2>Upload Results</h2>
      <ul>
        {#each uploadedFiles as file}
          <li>
            <strong>{file.name}</strong> - {file.status}
            {#if file.url}
              <a href={file.url} target="_blank" rel="noopener noreferrer"
                >View</a
              >
            {/if}
          </li>
        {/each}
      </ul>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>
