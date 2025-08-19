<script lang="ts">
  import { page } from "$app/stores";
  import { Button } from "$lib/components/ui/button";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import { notifications } from "$lib/stores/notification";
  import {
    AlertCircle,
    Archive,
    CheckSquare,
    Download,
    Eye,
    File,
    FileText,
    Folder,
    Grid,
    Image,
    List,
    MoreHorizontal,
    Music,
    Plus,
    RefreshCw,
    Search,
    Square,
    Trash2,
    Upload,
    Video,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Props
  export let caseId: string = "";

  // State
  let evidenceFiles: any[] = [];
  let filteredFiles: any[] = [];
  let loading = false;
  let error: string | null = null;
  let uploadProgress = 0;
  let uploading = false;
  let selectedFiles = new Set<string>();
  let showBulkActions = false;

  // Filters and view options
  let searchQuery = "";
  let selectedCategory = "";
  let viewMode = "grid"; // 'grid' | 'list'
  let sortBy = "uploadedAt";
  let sortOrder = "desc";

  // Upload modal state
  let showUploadModal = false;
  let dragActive = false;
  let uploadFiles: FileList | null = null;
  let uploadDescription = "";
  let uploadTags = "";

  // File categories
  const categories = [
    { value: "", label: "All Files", icon: Folder },
    { value: "image", label: "Images", icon: Image },
    { value: "video", label: "Videos", icon: Video },
    { value: "document", label: "Documents", icon: FileText },
    { value: "audio", label: "Audio", icon: Music },
    { value: "archive", label: "Archives", icon: Archive },
  ];

  // Get caseId from URL if not provided as prop
  $: if (!caseId) {
    caseId = $page.url.searchParams.get("caseId") || $page.params.id || "";
  }

  onMount(() => {
    if (caseId) {
      loadEvidenceFiles();
    }
  });

  async function loadEvidenceFiles() {
    if (!caseId) {
      error = "Case ID is required";
      return;
    }

    loading = true;
    error = null;

    try {
      const params = new URLSearchParams({ caseId });
      if (selectedCategory) params.append("category", selectedCategory);

      const response = await fetch(`/api/evidence/upload?${params}`);
      const data = await response.json();

      if (data.success) {
        evidenceFiles = data.files || [];
        filterAndSortFiles();
      } else {
        error = data.error || "Failed to load evidence files";
      }
    } catch (err) {
      console.error("Error loading evidence:", err);
      error = "Failed to load evidence files";
      notifications.add({
        type: "error",
        title: "Error Loading Evidence",
        message: "Failed to load evidence files. Please try again.",
        duration: 5000,
      });
    } finally {
      loading = false;
    }
  }

  function filterAndSortFiles() {
    let filtered = [...evidenceFiles];

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (f) =>
          f.title?.toLowerCase().includes(query) ||
          f.fileName?.toLowerCase().includes(query) ||
          f.description?.toLowerCase().includes(query)
      );
    }

    // Apply category filter
    if (selectedCategory) {
      filtered = filtered.filter((f) => f.evidenceType === selectedCategory);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];

      if (sortBy === "uploadedAt" || sortBy === "updatedAt") {
        aValue = new Date(aValue || 0).getTime();
        bValue = new Date(bValue || 0).getTime();
      } else if (sortBy === "fileSize") {
        aValue = Number(aValue) || 0;
        bValue = Number(bValue) || 0;
      } else if (typeof aValue === "string") {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }

      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    filteredFiles = filtered;
  }

  // File upload handlers
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    dragActive = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    dragActive = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragActive = false;

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      uploadFiles = files;
      if (files.length === 1) {
        showUploadModal = true;
      } else {
        uploadMultipleFiles();
      }
    }
  }

  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    uploadFiles = input.files;
    if (uploadFiles && uploadFiles.length > 0) {
      if (uploadFiles.length === 1) {
        showUploadModal = true;
      } else {
        uploadMultipleFiles();
      }
    }
  }

  async function uploadSingleFile() {
    if (!uploadFiles || uploadFiles.length === 0 || !caseId) return;

    uploading = true;
    uploadProgress = 0;

    try {
      const file = uploadFiles[0];
      const formData = new FormData();
      formData.append("file", file);
      formData.append("caseId", caseId);
      formData.append("description", uploadDescription);
      formData.append("tags", uploadTags);

      const response = await fetch("/api/evidence/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        notifications.add({
          type: "success",
          title: "File Uploaded",
          message: `${file.name} uploaded successfully`,
        });

        showUploadModal = false;
        uploadDescription = "";
        uploadTags = "";
        uploadFiles = null;

        await loadEvidenceFiles();
      } else {
        throw new Error(result.error || "Upload failed");
      }
    } catch (err) {
      console.error("Upload error:", err);
      notifications.add({
        type: "error",
        title: "Upload Failed",
        message: err instanceof Error ? err.message : "File upload failed",
        duration: 5000,
      });
    } finally {
      uploading = false;
      uploadProgress = 0;
    }
  }

  async function uploadMultipleFiles() {
    if (!uploadFiles || uploadFiles.length === 0 || !caseId) return;

    uploading = true;
    uploadProgress = 0;

    try {
      const formData = new FormData();
      Array.from(uploadFiles).forEach((file) => {
        formData.append("files", file);
      });
      formData.append("caseId", caseId);

      const response = await fetch("/api/evidence/upload", {
        method: "PUT",
        body: formData,
      });

      const result = await response.json();

      if (result.success && result.successCount > 0) {
        notifications.add({
          type: "success",
          title: "Bulk Upload Complete",
          message: `${result.successCount} files uploaded successfully`,
        });

        if (result.failureCount > 0) {
          notifications.add({
            type: "warning",
            title: "Some Uploads Failed",
            message: `${result.failureCount} files failed to upload`,
            duration: 8000,
          });
        }

        uploadFiles = null;
        await loadEvidenceFiles();
      } else {
        throw new Error(result.error || "Bulk upload failed");
      }
    } catch (err) {
      console.error("Bulk upload error:", err);
      notifications.add({
        type: "error",
        title: "Bulk Upload Failed",
        message: err instanceof Error ? err.message : "Bulk upload failed",
        duration: 5000,
      });
    } finally {
      uploading = false;
      uploadProgress = 0;
    }
  }

  // Selection handlers
  function toggleFileSelection(fileId: string) {
    if (selectedFiles.has(fileId)) {
      selectedFiles.delete(fileId);
    } else {
      selectedFiles.add(fileId);
    }
    selectedFiles = selectedFiles;
    showBulkActions = selectedFiles.size > 0;
  }

  function selectAllFiles() {
    if (selectedFiles.size === filteredFiles.length) {
      selectedFiles.clear();
    } else {
      filteredFiles.forEach((f) => selectedFiles.add(f.id));
    }
    selectedFiles = selectedFiles;
    showBulkActions = selectedFiles.size > 0;
  }

  // Utility functions
  function formatFileSize(bytes: number): string {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function getFileIcon(evidenceType: string) {
    switch (evidenceType) {
      case "image":
        return Image;
      case "video":
        return Video;
      case "audio":
        return Music;
      case "document":
        return FileText;
      case "archive":
        return Archive;
      default:
        return File;
    }
  }

  function getFileUrl(file: any): string {
    return file.fileUrl || `/uploads/${caseId}/${file.fileName}`;
  }

  // Reactive statements
  $: if (searchQuery || selectedCategory || sortBy || sortOrder) {
    filterAndSortFiles();
  }
</script>

<svelte:head>
  <title>Evidence Files - Case {caseId}</title>
</svelte:head>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Header -->
  <div
    class="mx-auto px-4 max-w-7xl"
  >
    <div>
      <h1 class="mx-auto px-4 max-w-7xl">Evidence Files</h1>
      <p class="mx-auto px-4 max-w-7xl">
        Manage evidence files for Case {caseId}
      </p>
    </div>

    <div class="mx-auto px-4 max-w-7xl">
      <Tooltip content="Refresh files">
        <Button
          variant="outline"
          size="sm"
          on:click={() => loadEvidenceFiles()}
          disabled={loading}
          aria-label="Refresh evidence files"
        >
          <RefreshCw class={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </Tooltip>

      <Tooltip content="Upload files">
        <Button
          on:click={() => (showUploadModal = true)}
          class="mx-auto px-4 max-w-7xl"
          disabled={!caseId}
        >
          <Upload class="mx-auto px-4 max-w-7xl" />
          Upload Files
        </Button>
      </Tooltip>
    </div>
  </div>

  <!-- Search and Filters -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <!-- Search -->
      <div class="mx-auto px-4 max-w-7xl">
        <Search
          class="mx-auto px-4 max-w-7xl"
        />
        <input
          type="text"
          placeholder="Search files by name, description..."
          class="mx-auto px-4 max-w-7xl"
          bind:value={searchQuery}
          aria-label="Search evidence files"
        />
      </div>

      <!-- Category Filter -->
      <select
        class="mx-auto px-4 max-w-7xl"
        bind:value={selectedCategory}
        aria-label="Filter by category"
      >
        {#each categories as category}
          <option value={category.value}>{category.label}</option>
        {/each}
      </select>

      <!-- View Mode Toggle -->
      <Tooltip content="Toggle view mode">
        <Button
          variant="outline"
          size="sm"
          on:click={() => (viewMode = viewMode === "grid" ? "list" : "grid")}
          aria-label="Toggle view mode"
        >
          {#if viewMode === "grid"}
            <List class="mx-auto px-4 max-w-7xl" />
          {:else}
            <Grid class="mx-auto px-4 max-w-7xl" />
          {/if}
        </Button>
      </Tooltip>
    </div>

    <!-- Sort Options -->
    <div class="mx-auto px-4 max-w-7xl">
      <select
        class="mx-auto px-4 max-w-7xl"
        bind:value={sortBy}
        aria-label="Sort by"
      >
        <option value="uploadedAt">Upload Date</option>
        <option value="title">Name</option>
        <option value="fileSize">File Size</option>
        <option value="evidenceType">Type</option>
      </select>

      <select
        class="mx-auto px-4 max-w-7xl"
        bind:value={sortOrder}
        aria-label="Sort order"
      >
        <option value="desc">Descending</option>
        <option value="asc">Ascending</option>
      </select>
    </div>
  </div>

  <!-- Bulk Actions -->
  {#if showBulkActions}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <span class="mx-auto px-4 max-w-7xl">{selectedFiles.size} file(s) selected</span>
        <div class="mx-auto px-4 max-w-7xl">
          <Button variant="outline" size="sm" class="mx-auto px-4 max-w-7xl">
            <Download class="mx-auto px-4 max-w-7xl" />
            Download
          </Button>
          <Button variant="outline" size="sm" class="mx-auto px-4 max-w-7xl">
            <Trash2 class="mx-auto px-4 max-w-7xl" />
            Delete
          </Button>
          <Button
            variant="ghost"
            size="sm"
            on:click={() => {
              selectedFiles.clear();
              selectedFiles = selectedFiles;
              showBulkActions = false;
            "
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Content -->
  {#if loading}
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl"></div>
      <span class="mx-auto px-4 max-w-7xl">Loading evidence files...</span>
    </div>
  {:else if error}
    <div class="mx-auto px-4 max-w-7xl" role="alert">
      <AlertCircle class="mx-auto px-4 max-w-7xl" />
      <div>
        <h3 class="mx-auto px-4 max-w-7xl">Error Loading Files</h3>
        <div class="mx-auto px-4 max-w-7xl">{error}</div>
      </div>
      <Button variant="outline" size="sm" on:click={() => loadEvidenceFiles()}>
        <RefreshCw class="mx-auto px-4 max-w-7xl" />
        Retry
      </Button>
    </div>
  {:else if filteredFiles.length === 0}
    <!-- Drop Zone for Empty State -->
    <div
      class="mx-auto px-4 max-w-7xl"
      class:border-primary={dragActive}
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
      on:drop={handleDrop}
      role="button"
      tabindex={0}
      aria-label="Drop files here to upload"
    >
      <Upload class="mx-auto px-4 max-w-7xl" />
      <h3 class="mx-auto px-4 max-w-7xl">
        {searchQuery || selectedCategory
          ? "No matching files found"
          : "No evidence files yet"}
      </h3>
      <p class="mx-auto px-4 max-w-7xl">
        {searchQuery || selectedCategory
          ? "Try adjusting your search criteria"
          : "Drag and drop files here or click to upload"}
      </p>

      {#if !searchQuery && !selectedCategory}
        <input
          type="file"
          multiple
          class="mx-auto px-4 max-w-7xl"
          id="file-upload"
          on:change={handleFileSelect}
          accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.rar"
        />
        <label for="file-upload">
          <Button class="mx-auto px-4 max-w-7xl">
            <Plus class="mx-auto px-4 max-w-7xl" />
            Choose Files
          </Button>
        </label>
      {/if}
    </div>
  {:else}
    <!-- Files Header -->
    <div class="mx-auto px-4 max-w-7xl">
      <span class="mx-auto px-4 max-w-7xl">
        {filteredFiles.length} file{filteredFiles.length !== 1 ? "s" : ""} found
      </span>

      <Button
        variant="ghost"
        size="sm"
        on:click={() => selectAllFiles()}
        class="mx-auto px-4 max-w-7xl"
      >
        {#if selectedFiles.size === filteredFiles.length}
          <CheckSquare class="mx-auto px-4 max-w-7xl" />
        {:else}
          <Square class="mx-auto px-4 max-w-7xl" />
        {/if}
        Select All
      </Button>
    </div>

    <!-- Files Grid/List -->
    {#if viewMode === "grid"}
      <div class="mx-auto px-4 max-w-7xl">
        {#each filteredFiles as file}
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <div class="mx-auto px-4 max-w-7xl">
              <!-- Selection and Actions -->
              <div class="mx-auto px-4 max-w-7xl">
                <input
                  type="checkbox"
                  class="mx-auto px-4 max-w-7xl"
                  checked={selectedFiles.has(file.id)}
                  on:change={() => toggleFileSelection(file.id)}
                  aria-label="Select file {file.title || file.fileName}"
                />

                <div class="mx-auto px-4 max-w-7xl">
                  <Button variant="ghost" size="sm" tabindex={0} role="button">
                    <MoreHorizontal class="mx-auto px-4 max-w-7xl" />
                  </Button>
                  <ul
                    tabindex={0}
                    role="menu"
                    class="mx-auto px-4 max-w-7xl"
                  >
                    <li>
                      <a href={getFileUrl(file)} target="_blank" class="mx-auto px-4 max-w-7xl">
                        <Eye class="mx-auto px-4 max-w-7xl" />
                        View
                      </a>
                    </li>
                    <li>
                      <a href={getFileUrl(file)} download class="mx-auto px-4 max-w-7xl">
                        <Download class="mx-auto px-4 max-w-7xl" />
                        Download
                      </a>
                    </li>
                    <li>
                      <button class="mx-auto px-4 max-w-7xl">
                        <Trash2 class="mx-auto px-4 max-w-7xl" />
                        Delete
                      </button>
                    </li>
                  </ul>
                </div>
              </div>

              <!-- File Preview/Icon -->
              <div class="mx-auto px-4 max-w-7xl">
                {#if file.evidenceType === "image"}
                  <img
                    src={getFileUrl(file)}
                    alt={file.title || file.fileName}
                    class="mx-auto px-4 max-w-7xl"
                    loading="lazy"
                  />
                {:else}
                  {@const IconComponent = getFileIcon(file.evidenceType)}
                  <IconComponent class="mx-auto px-4 max-w-7xl" />
                {/if}
              </div>

              <!-- File Info -->
              <div class="mx-auto px-4 max-w-7xl">
                <h3
                  class="mx-auto px-4 max-w-7xl"
                  title={file.title || file.fileName}
                >
                  {file.title || file.fileName}
                </h3>

                <div class="mx-auto px-4 max-w-7xl">
                  <div>Size: {formatFileSize(file.fileSize || 0)}</div>
                  <div>
                    Uploaded: {new Date(file.uploadedAt).toLocaleDateString()}
                  </div>
                  {#if file.description}
                    <div class="mx-auto px-4 max-w-7xl" title={file.description}>
                      {file.description}
                    </div>
                  {/if}
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {:else}
      <!-- List View -->
      <div class="mx-auto px-4 max-w-7xl">
        {#each filteredFiles as file}
          {@const IconComponent = getFileIcon(file.evidenceType)}
          <div
            class="mx-auto px-4 max-w-7xl"
          >
            <div class="mx-auto px-4 max-w-7xl">
              <input
                type="checkbox"
                class="mx-auto px-4 max-w-7xl"
                checked={selectedFiles.has(file.id)}
                on:change={() => toggleFileSelection(file.id)}
                aria-label="Select file {file.title || file.fileName}"
              />

              <IconComponent
                class="mx-auto px-4 max-w-7xl"
              />

              <div class="mx-auto px-4 max-w-7xl">
                <h3 class="mx-auto px-4 max-w-7xl">
                  {file.title || file.fileName}
                </h3>
                <div class="mx-auto px-4 max-w-7xl">
                  <span>{formatFileSize(file.fileSize || 0)}</span>
                  <span>{new Date(file.uploadedAt).toLocaleDateString()}</span>
                  {#if file.description}
                    <span class="mx-auto px-4 max-w-7xl">{file.description}</span>
                  {/if}
                </div>
              </div>

              <div class="mx-auto px-4 max-w-7xl">
                <Tooltip content="View file">
                  <a href={getFileUrl(file)} target="_blank">
                    <Button variant="outline" size="sm">
                      <Eye class="mx-auto px-4 max-w-7xl" />
                    </Button>
                  </a>
                </Tooltip>

                <Tooltip content="Download file">
                  <a href={getFileUrl(file)} download>
                    <Button variant="outline" size="sm">
                      <Download class="mx-auto px-4 max-w-7xl" />
                    </Button>
                  </a>
                </Tooltip>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<!-- Upload Modal -->
{#if showUploadModal}
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <h3 class="mx-auto px-4 max-w-7xl">Upload Evidence File</h3>

      {#if uploadFiles && uploadFiles.length > 0}
        <div class="mx-auto px-4 max-w-7xl">
          <!-- File Info -->
          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              <File class="mx-auto px-4 max-w-7xl" />
              <div>
                <div class="mx-auto px-4 max-w-7xl">{uploadFiles[0].name}</div>
                <div class="mx-auto px-4 max-w-7xl">
                  {formatFileSize(uploadFiles[0].size)}
                </div>
              </div>
            </div>
          </div>

          <!-- Description -->
          <div class="mx-auto px-4 max-w-7xl">
            <label class="mx-auto px-4 max-w-7xl" for="upload-description">
              <span class="mx-auto px-4 max-w-7xl">Description</span>
            </label>
            <textarea
              id="upload-description"
              class="mx-auto px-4 max-w-7xl"
              placeholder="Describe this evidence file..."
              bind:value={uploadDescription}
              rows={${1"
            ></textarea>
          </div>

          <!-- Tags -->
          <div class="mx-auto px-4 max-w-7xl">
            <label class="mx-auto px-4 max-w-7xl" for="upload-tags">
              <span class="mx-auto px-4 max-w-7xl">Tags</span>
            </label>
            <input
              id="upload-tags"
              type="text"
              class="mx-auto px-4 max-w-7xl"
              placeholder="crime-scene, photograph, evidence (comma-separated)"
              bind:value={uploadTags}
            />
          </div>

          <!-- Upload Progress -->
          {#if uploading}
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <progress
                class="mx-auto px-4 max-w-7xl"
                value={uploadProgress}
                max="100"
              ></progress>
            </div>
          {/if}
        </div>
      {/if}

      <div class="mx-auto px-4 max-w-7xl">
        <Button
          variant="outline"
          on:click={() => {
            showUploadModal = false;
            uploadFiles = null;
            uploadDescription = "";
            uploadTags = "";
          "
          disabled={uploading}
        >
          Cancel
        </Button>
        <Button
          on:click={() => uploadSingleFile()}
          disabled={uploading || !uploadFiles}
          class="mx-auto px-4 max-w-7xl"
        >
          {#if uploading}
            <div class="mx-auto px-4 max-w-7xl"></div>
            Uploading...
          {:else}
            <Upload class="mx-auto px-4 max-w-7xl" />
            Upload File
          {/if}
        </Button>
      </div>
    </div>
  </div>
{/if}

<!-- Hidden file input for drag and drop -->
<input
  type="file"
  multiple
  class="mx-auto px-4 max-w-7xl"
  id="bulk-upload"
  on:change={handleFileSelect}
  accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.rar"
/>
