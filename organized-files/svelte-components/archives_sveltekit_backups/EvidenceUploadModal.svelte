<script lang="ts">
  import Button from "$lib/components/ui/Button.svelte";
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
  let dragActive = false;

  $: isOpen = $uploadModal.isOpen;
  $: files = $uploadModal.files || [];
  $: activeUploads = files.filter(
    (f) => f?.status === "uploading" || f?.status === "processing"
  );
  $: completedUploads = files.filter((f) => f?.status === "completed");

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
  <div
    class="mx-auto px-4 max-w-7xl"
  >
    <div
      class="mx-auto px-4 max-w-7xl"
    >
      <!-- Header -->
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Upload class="mx-auto px-4 max-w-7xl" />
          <h2 class="mx-auto px-4 max-w-7xl">Upload Evidence</h2>
        </div>
        <Button variant="ghost" size="sm" onclick={() => closeModal()}>
          <X class="mx-auto px-4 max-w-7xl" />
        </Button>
      </div>

      <!-- Body -->
      <div class="mx-auto px-4 max-w-7xl">
        <!-- File Drop Zone -->
        <div
          role="button"
          tabindex={0}
          class="mx-auto px-4 max-w-7xl"
          on:drop={handleDrop}
          on:dragover={handleDragOver}
          on:dragleave={handleDragLeave}
          onclick={() => fileInput?.click()}
          onkeydown={(e) => e.key === "Enter" && fileInput?.click()}
        >
          <Upload class="mx-auto px-4 max-w-7xl" />
          <h3 class="mx-auto px-4 max-w-7xl">
            Drop files here or click to browse
          </h3>
          <p class="mx-auto px-4 max-w-7xl">
            Support for images, documents, audio, and video files
          </p>
          <Button variant="outline" onclick={() => fileInput?.click()}>
            Choose Files
          </Button>
          <input
            bind:this={fileInput}
            type="file"
            multiple
            class="mx-auto px-4 max-w-7xl"
            accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.csv,.xlsx,.xls"
            onchange={handleFileSelect}
          />
        </div>

        <!-- File List -->
        {#if files.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            <h3 class="mx-auto px-4 max-w-7xl">
              Files ({files.length})
            </h3>

            <div class="mx-auto px-4 max-w-7xl">
              {#each files as file (file.id)}
                {#if file?.file}
                  <div
                    class="mx-auto px-4 max-w-7xl"
                  >
                    <div class="mx-auto px-4 max-w-7xl">
                      <div class="mx-auto px-4 max-w-7xl">
                        {#if file.status === "completed"}
                          <CheckCircle class="mx-auto px-4 max-w-7xl" />
                        {:else if file.status === "error"}
                          <AlertCircle class="mx-auto px-4 max-w-7xl" />
                        {:else if file.status === "uploading" || file.status === "processing"}
                          <Loader2 class="mx-auto px-4 max-w-7xl" />
                        {:else}
                          <File class="mx-auto px-4 max-w-7xl" />
                        {/if}
                      </div>

                      <div class="mx-auto px-4 max-w-7xl">
                        <p class="mx-auto px-4 max-w-7xl">
                          {file.file?.name || "Unknown file"}
                        </p>
                        <p class="mx-auto px-4 max-w-7xl">
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
                          <div
                            class="mx-auto px-4 max-w-7xl"
                          >
                            <div
                              class="mx-auto px-4 max-w-7xl"
                              style="width: {file.progress}%"
                            ></div>
                          </div>
                        {/if}

                        {#if file.error}
                          <p class="mx-auto px-4 max-w-7xl">{file.error}</p>
                        {/if}
                      </div>
                    </div>

                    <div class="mx-auto px-4 max-w-7xl">
                      <Button
                        variant="ghost"
                        size="sm"
                        onclick={() => removeFile(file.id)}
                      >
                        <X class="mx-auto px-4 max-w-7xl" />
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
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
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

        <div class="mx-auto px-4 max-w-7xl">
          <Button variant="outline" onclick={() => closeModal()}>
            {activeUploads.length > 0 ? "Continue in Background" : "Close"}
          </Button>

          {#if completedUploads.length > 0}
            <Button onclick={() => dispatch("viewEvidence", completedUploads)}>
              View Evidence
            </Button>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}
