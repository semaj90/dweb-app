<script lang="ts">

  import Button from "$lib/components/ui/Button.svelte";
  import { $derived } from 'svelte';
  import { uploadActions, uploadModal } from "$lib/stores/evidence-store";
  import { formatFileSize } from "$lib/utils/file-utils";
  import {
    AlertCircle,
    CheckCircle,
    File,
    Loader2,
    Upload,
    X,
  } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();

  let fileInput: HTMLInputElement;
let dragActive = $state(false);

  let isOpen = $derived($uploadModal.isOpen);
  let files = $derived($uploadModal.files || []);
  let activeUploads = $derived(files.filter(
    (f) => f?.status === "uploading" || f?.status === "processing"
  ));
  let completedUploads = $derived(files.filter((f) => f?.status === "completed"));

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      uploadActions.addFiles(Array.from(target.files));
    }
  }
  function handleDrop(event: DragEvent) {
    event.preventDefault();
    dragActive = false;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      uploadActions.addFiles(Array.from(event.dataTransfer.files));
    }
  }
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    dragActive = true;
  }
  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    dragActive = false;
  }
  function removeFile(fileId: string) {
    uploadActions.removeFile(fileId);
  }
  function closeModal() {
    uploadActions.closeModal();
    dispatch("close");
  }
</script>

{#if isOpen}
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <!-- Header -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <Upload class="container mx-auto px-4" />
          <h2 class="container mx-auto px-4">Upload Evidence</h2>
        </div>
        <Button variant="ghost" size="sm" on:on:click={() => closeModal()}>
          <X class="container mx-auto px-4" />
        </Button>
      </div>

      <!-- Body -->
      <div class="container mx-auto px-4">
        <!-- File Drop Zone -->
        <div
          role="button"
          tabindex={0}
          aria-label="Evidence file drop zone"
          aria-describedby="evidence-dropzone-instructions"
          class="container mx-auto px-4"
          ondrop={handleDrop}
          ondragover={handleDragOver}
          ondragleave={handleDragLeave}
          on:onclick={() => fileInput?.click()}
          keydown={(e) =>
            (e.key === "Enter" || e.key === " ") && fileInput?.click()}
        >
          <Upload class="container mx-auto px-4" />
          <h3 class="container mx-auto px-4">
            Drop files here or click to browse
          </h3>
          <p id="evidence-dropzone-instructions" class="container mx-auto px-4">
            Support for images, documents, audio, and video files
          </p>
          <Button variant="outline" on:on:click={() => fileInput?.click()}>
            Choose Files
          </Button>
          <input
            bind:this={fileInput}
            type="file"
            multiple
            class="container mx-auto px-4"
            accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.csv,.xlsx,.xls"
            change={handleFileSelect}
          />
        </div>

        <!-- File List -->
        {#if files.length > 0}
          <div class="container mx-auto px-4">
            <h3 class="container mx-auto px-4">
              Files ({files.length})
            </h3>

            <div class="container mx-auto px-4">
              {#each files as file (file.id)}
                {#if file?.file}
                  <div class="container mx-auto px-4">
                    <div class="container mx-auto px-4">
                      <div class="container mx-auto px-4">
                        {#if file.status === "completed"}
                          <CheckCircle class="container mx-auto px-4" />
                        {:else if file.status === "error"}
                          <AlertCircle class="container mx-auto px-4" />
                        {:else if file.status === "uploading" || file.status === "processing"}
                          <Loader2 class="container mx-auto px-4" />
                        {:else}
                          <File class="container mx-auto px-4" />
                        {/if}
                      </div>

                      <div class="container mx-auto px-4">
                        <p class="container mx-auto px-4">
                          {file.file?.name || "Unknown file"}
                        </p>
                        <p class="container mx-auto px-4">
                          {file.file?.size
                            ? formatFileSize(file.file.size)
                            : "Unknown size"}
                          {#if file.status === "uploading"}
                            • {Math.round(file.progress || 0)}% uploaded
                          {:else if file.status === "processing"}
                            • Processing...
                          {:else if file.status === "error"}
                            • Upload failed
                          {:else if file.status === "completed"}
                            • Upload complete
                          {/if}
                        </p>

                        {#if file.status === "uploading" && file.progress && file.progress > 0}
                          <div class="container mx-auto px-4">
                            <div
                              class="container mx-auto px-4"
                              style="width: {file.progress}%"
                            ></div>
                          </div>
                        {/if}

                        {#if file.error}
                          <p class="container mx-auto px-4">{file.error}</p>
                        {/if}
                      </div>
                    </div>

                    <div class="container mx-auto px-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        on:on:click={() => removeFile(file.id)}
                      >
                        <X class="container mx-auto px-4" />
                      </Button>
                    </div>
                  </div>
                {/if}
              {/each}
            </div>
          </div>
        {/if}
      </div>

      <!-- Footer -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          {#if activeUploads.length > 0}
            Processing {activeUploads.length} file{activeUploads.length !== 1
              ? "s"
              : ""}...
          {:else if completedUploads.length > 0}
            {completedUploads.length} file{completedUploads.length !== 1
              ? "s"
              : ""} uploaded successfully
          {:else}
            Ready to upload files
          {/if}
        </div>

        <div class="container mx-auto px-4">
          <Button variant="outline" on:on:click={() => closeModal()}>
            {activeUploads.length > 0 ? "Continue in Background" : "Close"}
          </Button>

          {#if completedUploads.length > 0}
            <Button on:on:click={() => dispatch("viewEvidence", completedUploads)}>
              View Evidence
            </Button>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}


