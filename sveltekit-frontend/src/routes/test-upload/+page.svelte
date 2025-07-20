<script lang="ts">
  import AdvancedFileUpload from "$lib/components/upload/AdvancedFileUpload.svelte";

  let uploadedFiles: any[] = [];

  function handleFilesAdded(event) {
    console.log("Files added:", event.detail.files);
}
  function handleUploadComplete(event) {
    console.log("Upload complete:", event.detail.files);
    uploadedFiles = [...uploadedFiles, ...event.detail.files];
}
  function handleFileRemoved(event) {
    console.log("File removed:", event.detail.fileId);
}
</script>

<svelte:head>
  <title>File Upload Test</title>
</svelte:head>

<div class="container mx-auto px-4">
  <h1>File Upload Test</h1>

  <div class="container mx-auto px-4">
    <AdvancedFileUpload
      multiple={true}
      accept="*/*"
      maxFileSize={50 * 1024 * 1024}
      maxFiles={5}
      enablePreview={true}
      enableDragDrop={true}
      enablePasteUpload={true}
      autoUpload={false}
      on:filesAdded={handleFilesAdded}
      on:uploadComplete={handleUploadComplete}
      on:fileRemoved={handleFileRemoved}
    />
  </div>

  {#if uploadedFiles.length > 0}
    <div class="container mx-auto px-4">
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
