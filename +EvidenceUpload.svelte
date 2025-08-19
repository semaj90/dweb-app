<script lang="ts">
  import type { Evidence } from "$lib/data/types";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher<{
    upload: Evidence;
    error: string;
  }>();

  let dragActive = false;
  let files: FileList | null = null;
  let uploadProgress = 0;

  function handleDragEnter(e: DragEvent) {
    e.preventDefault();
    dragActive = true;
  }

  function handleDragLeave() {
    dragActive = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragActive = false;
    if (e.dataTransfer?.files) {
      files = e.dataTransfer.files;
      handleUpload();
    }
  }

  async function handleUpload() {
    if (!files?.length) return;

    const formData = new FormData();
    Array.from(files).forEach((file) => {
      formData.append("files", file);
    });

    try {
      const response = await fetch("/api/evidence", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const evidence = await response.json();
      dispatch("upload", evidence);
    } catch (error) {
      dispatch(
        "error",
        error instanceof Error ? error.message : "Upload failed"
      );
    }
  }
</script>

<div
  class="upload-zone"
  class:active={dragActive}
  role="button"
  tabindex="0"
  aria-label="Evidence upload area. Press Enter or Space to choose files, or drag and drop."
  on:dragenter={handleDragEnter}
  on:dragleave={handleDragLeave}
  on:dragover|preventDefault
  on:drop={handleDrop}
  onkeydown={(e) =>
    (e.key === "Enter" || e.key === " ") &&
    (e.preventDefault(),
    (
      e.currentTarget.querySelector('input[type="file"]') as HTMLInputElement
    )?.click())}
>
  <div class="upload-content">
    <svg
      xmlns="http://www.w3.org/2000/svg"
      class="upload-icon"
      viewBox="0 0 20 20"
      fill="currentColor"
    >
      <path
        fill-rule="evenodd"
        d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z"
        clip-rule="evenodd"
      />
    </svg>
    <p>Drag and drop evidence files here or click to select</p>
    <input
      type="file"
      multiple
      onchange={(e) => {
        files = e.currentTarget.files;
        handleUpload();
      }}
      style="display: none"
    />
  </div>

  {#if uploadProgress > 0}
    <div class="progress-bar">
      <div class="progress" style="width: {uploadProgress}%"></div>
    </div>
  {/if}
</div>

<style>
  .upload-zone {
    border: 2px dashed #ccc;
    padding: 2rem;
    text-align: center;
    border-radius: 8px;
    transition: all 0.3s ease;
  }

  .upload-zone.active {
    border-color: #007bff;
    background-color: rgba(0, 123, 255, 0.1);
  }

  .upload-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .upload-icon {
    width: 48px;
    height: 48px;
    color: #666;
  }

  .progress-bar {
    margin-top: 1rem;
    background-color: #eee;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress {
    height: 4px;
    background-color: #007bff;
    transition: width 0.3s ease;
  }
</style>
