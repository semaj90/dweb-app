<script lang="ts">
  interface Props {
    caseId: string
  }
  let {
    caseId = ""
  } = $props();



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
  $effect(() => { if (!caseId) {
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
}}
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
}}}
  function handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    uploadFiles = input.files;
    if (uploadFiles && uploadFiles.length > 0) {
      if (uploadFiles.length === 1) {
        showUploadModal = true;
      } else {
        uploadMultipleFiles();
}}}
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
}}
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
}}
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
}}
  function getFileUrl(file: any): string {
    return file.fileUrl || `/uploads/${caseId}/${file.fileName}`;
}
  // Reactive statements
  $effect(() => { if (searchQuery || selectedCategory || sortBy || sortOrder) {
    filterAndSortFiles();
}
</script>

<svelte:head>
  <title>Evidence Files - Case {caseId}</title>
</svelte:head>

<div class="space-y-4">
  <!-- Header -->
  <div
    class="space-y-4"
  >
    <div>
      <h1 class="space-y-4">Evidence Files</h1>
      <p class="space-y-4">
        Manage evidence files for Case {caseId}
      </p>
    </div>

    <div class="space-y-4">
      <Tooltip content="Refresh files">
        <Button
          variant="outline"
          size="sm"
          onclick={() => loadEvidenceFiles()}
          disabled={loading}
          aria-label="Refresh evidence files"
        >
          <RefreshCw class={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </Tooltip>

      <Tooltip content="Upload files">
        <Button
          onclick={() => (showUploadModal = true)}
          class="space-y-4"
          disabled={!caseId}
        >
          <Upload class="space-y-4" />
          Upload Files
        </Button>
      </Tooltip>
    </div>
  </div>

  <!-- Search and Filters -->
  <div class="space-y-4">
    <div class="space-y-4">
      <!-- Search -->
      <div class="space-y-4">
        <Search
          class="space-y-4"
        />
        <input
          type="text"
          placeholder="Search files by name, description..."
          class="space-y-4"
          bind:value={searchQuery}
          aria-label="Search evidence files"
        />
      </div>

      <!-- Category Filter -->
      <select
        class="space-y-4"
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
          onclick={() => (viewMode = viewMode === "grid" ? "list" : "grid")}
          aria-label="Toggle view mode"
        >
          {#if viewMode === "grid"}
            <List class="space-y-4" />
          {:else}
            <Grid class="space-y-4" />
          {/if}
        </Button>
      </Tooltip>
    </div>

    <!-- Sort Options -->
    <div class="space-y-4">
      <select
        class="space-y-4"
        bind:value={sortBy}
        aria-label="Sort by"
      >
        <option value="uploadedAt">Upload Date</option>
        <option value="title">Name</option>
        <option value="fileSize">File Size</option>
        <option value="evidenceType">Type</option>
      </select>

      <select
        class="space-y-4"
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
    <div class="space-y-4">
      <div class="space-y-4">
        <span class="space-y-4">{selectedFiles.size} file(s) selected</span>
        <div class="space-y-4">
          <Button variant="outline" size="sm" class="space-y-4">
            <Download class="space-y-4" />
            Download
          </Button>
          <Button variant="outline" size="sm" class="space-y-4">
            <Trash2 class="space-y-4" />
            Delete
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onclick={() => {
              selectedFiles.clear();
              selectedFiles = selectedFiles;
              showBulkActions = false;
            }}
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  {/if}

  <!-- Content -->
  {#if loading}
    <div class="space-y-4">
      <div class="space-y-4"></div>
      <span class="space-y-4">Loading evidence files...</span>
    </div>
  {:else if error}
    <div class="space-y-4" role="alert">
      <AlertCircle class="space-y-4" />
      <div>
        <h3 class="space-y-4">Error Loading Files</h3>
        <div class="space-y-4">{error}</div>
      </div>
      <Button variant="outline" size="sm" onclick={() => loadEvidenceFiles()}>
        <RefreshCw class="space-y-4" />
        Retry
      </Button>
    </div>
  {:else if filteredFiles.length === 0}
    <!-- Drop Zone for Empty State -->
    <div
      class="space-y-4"
      class:border-primary={dragActive}
      ondragover={handleDragOver}
      ondragleave={handleDragLeave}
      ondrop={handleDrop}
      role="button"
      tabindex={0}
      aria-label="Drop files here to upload"
    >
      <Upload class="space-y-4" />
      <h3 class="space-y-4">
        {searchQuery || selectedCategory
          ? "No matching files found"
          : "No evidence files yet"}
      </h3>
      <p class="space-y-4">
        {searchQuery || selectedCategory
          ? "Try adjusting your search criteria"
          : "Drag and drop files here or click to upload"}
      </p>

      {#if !searchQuery && !selectedCategory}
        <input
          type="file"
          multiple
          class="space-y-4"
          id="file-upload"
          onchange={handleFileSelect}
          accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.rar"
        />
        <label for="file-upload">
          <Button class="space-y-4">
            <Plus class="space-y-4" />
            Choose Files
          </Button>
        </label>
      {/if}
    </div>
  {:else}
    <!-- Files Header -->
    <div class="space-y-4">
      <span class="space-y-4">
        {filteredFiles.length} file{filteredFiles.length !== 1 ? "s" : ""} found
      </span>

      <Button
        variant="ghost"
        size="sm"
        onclick={() => selectAllFiles()}
        class="space-y-4"
      >
        {#if selectedFiles.size === filteredFiles.length}
          <CheckSquare class="space-y-4" />
        {:else}
          <Square class="space-y-4" />
        {/if}
        Select All
      </Button>
    </div>

    <!-- Files Grid/List -->
    {#if viewMode === "grid"}
      <div class="space-y-4">
        {#each filteredFiles as file}
          <div
            class="space-y-4"
          >
            <div class="space-y-4">
              <!-- Selection and Actions -->
              <div class="space-y-4">
                <input
                  type="checkbox"
                  class="space-y-4"
                  checked={selectedFiles.has(file.id)}
                  onchange={() => toggleFileSelection(file.id)}
                  aria-label="Select file {file.title || file.fileName}"
                />

                <div class="space-y-4">
                  <Button variant="ghost" size="sm" tabindex={0} role="button">
                    <MoreHorizontal class="space-y-4" />
                  </Button>
                  <ul
                    tabindex={0}
                    role="menu"
                    class="space-y-4"
                  >
                    <li>
                      <a href={getFileUrl(file)} target="_blank" class="space-y-4">
                        <Eye class="space-y-4" />
                        View
                      </a>
                    </li>
                    <li>
                      <a href={getFileUrl(file)} download class="space-y-4">
                        <Download class="space-y-4" />
                        Download
                      </a>
                    </li>
                    <li>
                      <button class="space-y-4">
                        <Trash2 class="space-y-4" />
                        Delete
                      </button>
                    </li>
                  </ul>
                </div>
              </div>

              <!-- File Preview/Icon -->
              <div class="space-y-4">
                {#if file.evidenceType === "image"}
                  <img
                    src={getFileUrl(file)}
                    alt={file.title || file.fileName}
                    class="space-y-4"
                    loading="lazy"
                  />
                {:else}
                  {@const IconComponent = getFileIcon(file.evidenceType)}
                  <IconComponent class="space-y-4" />
                {/if}
              </div>

              <!-- File Info -->
              <div class="space-y-4">
                <h3
                  class="space-y-4"
                  title={file.title || file.fileName}
                >
                  {file.title || file.fileName}
                </h3>

                <div class="space-y-4">
                  <div>Size: {formatFileSize(file.fileSize || 0)}</div>
                  <div>
                    Uploaded: {new Date(file.uploadedAt).toLocaleDateString()}
                  </div>
                  {#if file.description}
                    <div class="space-y-4" title={file.description}>
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
      <div class="space-y-4">
        {#each filteredFiles as file}
          {@const IconComponent = getFileIcon(file.evidenceType)}
          <div
            class="space-y-4"
          >
            <div class="space-y-4">
              <input
                type="checkbox"
                class="space-y-4"
                checked={selectedFiles.has(file.id)}
                onchange={() => toggleFileSelection(file.id)}
                aria-label="Select file {file.title || file.fileName}"
              />

              <IconComponent
                class="space-y-4"
              />

              <div class="space-y-4">
                <h3 class="space-y-4">
                  {file.title || file.fileName}
                </h3>
                <div class="space-y-4">
                  <span>{formatFileSize(file.fileSize || 0)}</span>
                  <span>{new Date(file.uploadedAt).toLocaleDateString()}</span>
                  {#if file.description}
                    <span class="space-y-4">{file.description}</span>
                  {/if}
                </div>
              </div>

              <div class="space-y-4">
                <Tooltip content="View file">
                  <a href={getFileUrl(file)} target="_blank">
                    <Button variant="outline" size="sm">
                      <Eye class="space-y-4" />
                    </Button>
                  </a>
                </Tooltip>

                <Tooltip content="Download file">
                  <a href={getFileUrl(file)} download>
                    <Button variant="outline" size="sm">
                      <Download class="space-y-4" />
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
  <div class="space-y-4">
    <div class="space-y-4">
      <h3 class="space-y-4">Upload Evidence File</h3>

      {#if uploadFiles && uploadFiles.length > 0}
        <div class="space-y-4">
          <!-- File Info -->
          <div class="space-y-4">
            <div class="space-y-4">
              <File class="space-y-4" />
              <div>
                <div class="space-y-4">{uploadFiles[0].name}</div>
                <div class="space-y-4">
                  {formatFileSize(uploadFiles[0].size)}
                </div>
              </div>
            </div>
          </div>

          <!-- Description -->
          <div class="space-y-4">
            <label class="space-y-4" for="upload-description">
              <span class="space-y-4">Description</span>
            </label>
            <textarea
              id="upload-description"
              class="space-y-4"
              placeholder="Describe this evidence file..."
              bind:value={uploadDescription}
              rows={4}
            ></textarea>
          </div>

          <!-- Tags -->
          <div class="space-y-4">
            <label class="space-y-4" for="upload-tags">
              <span class="space-y-4">Tags</span>
            </label>
            <input
              id="upload-tags"
              type="text"
              class="space-y-4"
              placeholder="crime-scene, photograph, evidence (comma-separated)"
              bind:value={uploadTags}
            />
          </div>

          <!-- Upload Progress -->
          {#if uploading}
            <div class="space-y-4">
              <div class="space-y-4">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <progress
                class="space-y-4"
                value={uploadProgress}
                max="100"
              ></progress>
            </div>
          {/if}
        </div>
      {/if}

      <div class="space-y-4">
        <Button
          variant="outline"
          onclick={() => {
            showUploadModal = false;
            uploadFiles = null;
            uploadDescription = "";
            uploadTags = "";
          }}
          disabled={uploading}
        >
          Cancel
        </Button>
        <Button
          onclick={() => uploadSingleFile()}
          disabled={uploading || !uploadFiles}
          class="space-y-4"
        >
          {#if uploading}
            <div class="space-y-4"></div>
            Uploading...
          {:else}
            <Upload class="space-y-4" />
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
  class="space-y-4"
  id="bulk-upload"
  onchange={handleFileSelect}
  accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.rar"
/>
