<!-- Enhanced Document Uploader with Bits UI v2, AI Processing, and Real-time Status -->
<script lang="ts">
  import {
    Badge,
    Button,
    Card,
    CardContent,
    CardHeader,
    CardTitle,
    Checkbox,
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    Input,
    Label,
    Progress,
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
    Textarea,
  } from "bits-ui";
  import {
    AlertTriangle,
    CheckCircle,
    File,
    FileImage,
    FileText,
    Loader2,
    Upload,
    X,
  } from "lucide-svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { derived, writable } from "svelte/store";

  // Props with Svelte 5 syntax
  let {
    acceptedTypes = ".pdf,.docx,.txt,.jpg,.jpeg,.png,.gif,.webp",
    maxFileSize = 50 * 1024 * 1024, // 50MB
    maxFiles = 10,
    caseId = "",
    userId = "",
    autoProcess = true,
    showMetadataForm = true,
    class: className = "",
  } = $props();

  // Event dispatcher
  const dispatch = createEventDispatcher<{
    upload: { files: ProcessedFile[] };
    processing: { fileId: string progress: number };
    complete: { fileId: string result: ProcessingResult };
    error: { fileId: string error: string };
  }>();

  // Types
  interface UploadFile {
    id: string
    file: File
    preview?: string;
    status: "pending" | "uploading" | "processing" | "completed" | "error";
    progress: number
    error?: string;
    metadata: {
      title?: string;
      description?: string;
      documentType?: string;
      jurisdiction?: string;
      tags?: string[];
      autoSummarize?: boolean;
      extractEntities?: boolean;
    };
  }

  interface ProcessedFile {
    id: string
    documentId: string
    filename: string
    size: number
    type: string
    url?: string;
    thumbnail?: string;
  }

  interface ProcessingResult {
    summary?: string;
    entities?: Array<{ text: string type: string confidence: number }>;
    chunks?: number;
    embeddings?: boolean;
  }

  // State management
  const files = writable<UploadFile[]>([]);
  const isDragging = writable(false);
  const isProcessing = writable(false);
  const showMetadata = writable(false);
  const selectedFile = writable<UploadFile | null>(null);

  // Derived state
  const totalProgress = derived(files, ($files) => {
    if ($files.length === 0) return 0;
    return $files.reduce((acc, file) => acc + file.progress, 0) / $files.length;
  });

  const completedFiles = derived(files, ($files) =>
    $files.filter((file) => file.status === "completed")
  );

  const hasErrors = derived(files, ($files) =>
    $files.some((file) => file.status === "error")
  );

  // File input reference
  let fileInput: HTMLInputElement
  let dropZone: HTMLDivElement

  // Document types for legal AI
  const documentTypes = [
    { value: "contract", label: "Contract" },
    { value: "motion", label: "Motion" },
    { value: "brief", label: "Brief" },
    { value: "evidence", label: "Evidence" },
    { value: "correspondence", label: "Correspondence" },
    { value: "statute", label: "Statute" },
    { value: "regulation", label: "Regulation" },
    { value: "case_law", label: "Case Law" },
    { value: "other", label: "Other" },
  ];

  const jurisdictions = [
    { value: "federal", label: "Federal" },
    { value: "state", label: "State" },
    { value: "local", label: "Local" },
    { value: "international", label: "International" },
  ];

  // ============================================================================
  // DRAG & DROP HANDLERS
  // ============================================================================

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragging.set(true);
  }

  function handleDragLeave(event: DragEvent) {
    if (
      !event.relatedTarget ||
      !dropZone?.contains(event.relatedTarget as Node)
    ) {
      isDragging.set(false);
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragging.set(false);

    const droppedFiles = Array.from(event.dataTransfer?.files || []);
    processSelectedFiles(droppedFiles);
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const selectedFiles = Array.from(target.files || []);
    processSelectedFiles(selectedFiles);
    target.value = ""; // Reset input
  }

  // ============================================================================
  // FILE PROCESSING
  // ============================================================================

  function processSelectedFiles(selectedFiles: File[]) {
    const validFiles = selectedFiles.filter((file) => {
      // Check file type
      const extension = "." + file.name.split(".").pop()?.toLowerCase();
      if (!acceptedTypes.includes(extension)) {
        console.warn(`File type ${extension} not accepted`);
        return false;
      }

      // Check file size
      if (file.size > maxFileSize) {
        console.warn(`File ${file.name} exceeds maximum size`);
        return false;
      }

      return true;
    });

    // Check total file count
    files.update((currentFiles) => {
      if (currentFiles.length + validFiles.length > maxFiles) {
        console.warn(`Maximum ${maxFiles} files allowed`);
        return currentFiles;
      }

      const newFiles: UploadFile[] = validFiles.map((file) => ({
        id: crypto.randomUUID(),
        file,
        status: "pending",
        progress: 0,
        metadata: {
          title: file.name.replace(/\.[^/.]+$/, ""),
          documentType: "other",
          autoSummarize: true,
          extractEntities: true,
          tags: [],
        },
      }));

      // Generate previews for images
      newFiles.forEach((uploadFile) => {
        if (uploadFile.file.type.startsWith("image/")) {
          const reader = new FileReader();
          reader.onload = (e) => {
            uploadFile.preview = e.target?.result as string;
            files.update((f) => [...f]); // Trigger reactivity
          };
          reader.readAsDataURL(uploadFile.file);
        }
      });

      return [...currentFiles, ...newFiles];
    });

    // Auto-upload if enabled
    if (autoProcess) {
      uploadFiles();
    }
  }

  // ============================================================================
  // UPLOAD & AI PROCESSING
  // ============================================================================

  async function uploadFiles() {
    isProcessing.set(true);

    const currentFiles = $files.filter((file) => file.status === "pending");

    for (const uploadFile of currentFiles) {
      try {
        await uploadSingleFile(uploadFile);
      } catch (error) {
        console.error("Upload error:", error);
        updateFileStatus(uploadFile.id, "error", 0, String(error));
      }
    }

    isProcessing.set(false);
  }

  async function uploadSingleFile(uploadFile: UploadFile) {
    updateFileStatus(uploadFile.id, "uploading", 10);

    // Create FormData
    const formData = new FormData();
    formData.append("file", uploadFile.file);
    formData.append("caseId", caseId);
    formData.append("userId", userId);
    formData.append("metadata", JSON.stringify(uploadFile.metadata));

    try {
      // Upload file
      const uploadResponse = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`);
      }

      const uploadResult = await uploadResponse.json();
      updateFileStatus(uploadFile.id, "processing", 50);

      // Start AI processing
      if (
        uploadFile.metadata.autoSummarize ||
        uploadFile.metadata.extractEntities
      ) {
        const processingResponse = await fetch("/api/ai/process-document", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            documentId: uploadResult.documentId,
            extractEntities: uploadFile.metadata.extractEntities,
            generateSummary: uploadFile.metadata.autoSummarize,
            riskAssessment: true,
          }),
        });

        if (!processingResponse.ok) {
          throw new Error(
            `AI processing failed: ${processingResponse.statusText}`
          );
        }

        const processingResult = await processingResponse.json();
        updateFileStatus(uploadFile.id, "completed", 100);

        // Emit events
        dispatch("complete", {
          fileId: uploadFile.id,
          result: processingResult,
        });

        dispatch("upload", {
          files: [
            {
              id: uploadFile.id,
              documentId: uploadResult.documentId,
              filename: uploadFile.file.name,
              size: uploadFile.file.size,
              type: uploadFile.file.type,
              url: uploadResult.url,
              thumbnail: uploadFile.preview,
            },
          ],
        });
      } else {
        updateFileStatus(uploadFile.id, "completed", 100);
      }
    } catch (error) {
      updateFileStatus(uploadFile.id, "error", 0, String(error));
      dispatch("error", {
        fileId: uploadFile.id,
        error: String(error),
      });
    }
  }

  function updateFileStatus(
    fileId: string,
    status: UploadFile["status"],
    progress: number,
    error?: string
  ) {
    files.update((currentFiles) =>
      currentFiles.map((file) =>
        file.id === fileId
          ? { ...file, status, progress, ...(error && { error }) }
          : file
      )
    );

    if (status === "processing") {
      dispatch("processing", { fileId, progress });
    }
  }

  // ============================================================================
  // FILE MANAGEMENT
  // ============================================================================

  function removeFile(fileId: string) {
    files.update((currentFiles) =>
      currentFiles.filter((file) => file.id !== fileId)
    );
  }

  function openMetadataDialog(file: UploadFile) {
    selectedFile.set(file);
    showMetadata.set(true);
  }

  function updateFileMetadata(
    fileId: string,
    metadata: Partial<UploadFile["metadata"]>
  ) {
    files.update((currentFiles) =>
      currentFiles.map((file) =>
        file.id === fileId
          ? { ...file, metadata: { ...file.metadata, ...metadata } }
          : file
      )
    );
  }

  function getFileIcon(file: File) {
    if (file.type.startsWith("image/")) return FileImage;
    if (file.type.includes("pdf")) return FileText;
    return File;
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function getStatusColor(status: UploadFile["status"]): string {
    switch (status) {
      case "completed":
        return "green";
      case "error":
        return "red";
      case "processing":
        return "blue";
      case "uploading":
        return "yellow";
      default:
        return "gray";
    }
  }

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(() => {
    // Set up global drag and drop prevention
    const preventDefaults = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
    };

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      document.addEventListener(eventName, preventDefaults, false);
    });

    return () => {
      ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        document.removeEventListener(eventName, preventDefaults, false);
      });
    };
  });
</script>

<!-- Main Upload Interface -->
<div class="enhanced-document-uploader {className}">
  <!-- Drop Zone -->
  <div
    bind:this={dropZone}
    class="drop-zone"
    class:dragging={$isDragging}
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
    role="button"
    tabindex="0"
    on:click={() => fileInput?.click()}
    on:keydown={(e) => e.key === "Enter" && fileInput?.click()}
  >
    <div class="drop-zone-content">
      <Upload class="drop-zone-icon" size={48} />
      <h3 class="drop-zone-title">
        {$isDragging ? "Drop files here" : "Upload Legal Documents"}
      </h3>
      <p class="drop-zone-description">
        Drag and drop files here, or click to select
      </p>
      <p class="drop-zone-specs">
        Supports: PDF, DOCX, TXT, Images • Max {formatFileSize(maxFileSize)} • Up
        to {maxFiles} files
      </p>

      <Button variant="outline" class="mt-4" disabled={$isProcessing}>
        <Upload class="mr-2" size={16} />
        Choose Files
      </Button>
    </div>
  </div>

  <!-- Hidden File Input -->
  <input
    bind:this={fileInput}
    type="file"
    multiple
    accept={acceptedTypes}
    on:change={handleFileSelect}
    class="sr-only"
  />

  <!-- Progress Overview -->
  {#if $files.length > 0}
    <Card class="mt-6">
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          <span>Upload Progress</span>
          <Badge variant={$hasErrors ? "destructive" : "default"}>
            {$completedFiles.length} / {$files.length} completed
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-2">
          <Progress value={$totalProgress} class="w-full" />
          <p class="text-sm text-muted-foreground">
            Overall Progress: {Math.round($totalProgress)}%
          </p>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- File List -->
  {#if $files.length > 0}
    <div class="file-list mt-6">
      {#each $files as file (file.id)}
        <Card class="file-item">
          <CardContent class="p-4">
            <div class="file-info">
              <!-- File Icon/Preview -->
              <div class="file-preview">
                {#if file.preview}
                  <img src={file.preview} alt="Preview" class="preview-image" />
                {:else}
                  <svelte:component this={getFileIcon(file.file)} size={24} />
                {/if}
              </div>

              <!-- File Details -->
              <div class="file-details">
                <h4 class="file-name">
                  {file.metadata.title || file.file.name}
                </h4>
                <p class="file-meta">
                  {formatFileSize(file.file.size)} • {file.file.type}
                  {#if file.metadata.documentType !== "other"}
                    • {documentTypes.find(
                      (t) => t.value === file.metadata.documentType
                    )?.label}
                  {/if}
                </p>

                <!-- Progress Bar -->
                {#if file.status !== "pending" && file.status !== "completed"}
                  <Progress value={file.progress} class="file-progress" />
                {/if}

                <!-- Error Message -->
                {#if file.error}
                  <p class="error-message">
                    <AlertTriangle size={16} />
                    {file.error}
                  </p>
                {/if}
              </div>

              <!-- Status & Actions -->
              <div class="file-actions">
                <Badge variant={getStatusColor(file.status) as any}>
                  {#if file.status === "processing"}
                    <Loader2 class="mr-1 animate-spin" size={12} />
                  {:else if file.status === "completed"}
                    <CheckCircle class="mr-1" size={12} />
                  {:else if file.status === "error"}
                    <AlertTriangle class="mr-1" size={12} />
                  {/if}
                  {file.status}
                </Badge>

                <div class="action-buttons">
                  {#if showMetadataForm && file.status === "pending"}
                    <Button
                      variant="ghost"
                      size="sm"
                      on:click={() => openMetadataDialog(file)}
                    >
                      Edit
                    </Button>
                  {/if}

                  <Button
                    variant="ghost"
                    size="sm"
                    on:click={() => removeFile(file.id)}
                    disabled={file.status === "uploading" ||
                      file.status === "processing"}
                  >
                    <X size={16} />
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      {/each}
    </div>

    <!-- Upload Actions -->
    <div class="upload-actions mt-6">
      <Button
        on:click={uploadFiles}
        disabled={$isProcessing || $files.every((f) => f.status !== "pending")}
        class="mr-4"
      >
        {#if $isProcessing}
          <Loader2 class="mr-2 animate-spin" size={16} />
          Processing...
        {:else}
          <Upload class="mr-2" size={16} />
          Upload & Process ({$files.filter((f) => f.status === "pending")
            .length} files)
        {/if}
      </Button>

      <Button
        variant="outline"
        on:click={() => files.set([])}
        disabled={$isProcessing}
      >
        Clear All
      </Button>
    </div>
  {/if}

  <!-- Metadata Dialog -->
  <Dialog bind:open={$showMetadata}>
    <DialogContent class="max-w-md">
      <DialogHeader>
        <DialogTitle>Document Metadata</DialogTitle>
      </DialogHeader>

      {#if $selectedFile}
        <div class="metadata-form space-y-4">
          <div>
            <Label for="title">Title</Label>
            <Input
              id="title"
              bind:value={$selectedFile.metadata.title}
              placeholder="Document title"
            />
          </div>

          <div>
            <Label for="description">Description</Label>
            <Textarea
              id="description"
              bind:value={$selectedFile.metadata.description}
              placeholder="Brief description"
              rows={3}
            />
          </div>

          <div>
            <Label for="document-type">Document Type</Label>
            <Select bind:value={$selectedFile.metadata.documentType}>
              <SelectTrigger>
                <SelectValue placeholder="Select type" />
              </SelectTrigger>
              <SelectContent>
                {#each documentTypes as type}
                  <SelectItem value={type.value}>{type.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label for="jurisdiction">Jurisdiction</Label>
            <Select bind:value={$selectedFile.metadata.jurisdiction}>
              <SelectTrigger>
                <SelectValue placeholder="Select jurisdiction" />
              </SelectTrigger>
              <SelectContent>
                {#each jurisdictions as jurisdiction}
                  <SelectItem value={jurisdiction.value}
                    >{jurisdiction.label}</SelectItem
                  >
                {/each}
              </SelectContent>
            </Select>
          </div>

          <div class="ai-options">
            <Label>AI Processing Options</Label>
            <div class="checkbox-group">
              <Checkbox bind:checked={$selectedFile.metadata.autoSummarize}>
                Auto-generate summary
              </Checkbox>
              <Checkbox bind:checked={$selectedFile.metadata.extractEntities}>
                Extract entities (names, dates, amounts)
              </Checkbox>
            </div>
          </div>

          <div class="dialog-actions">
            <Button variant="outline" on:click={() => showMetadata.set(false)}>
              Cancel
            </Button>
            <Button
              on:click={() => {
                if ($selectedFile) {
                  updateFileMetadata($selectedFile.id, $selectedFile.metadata);
                }
                showMetadata.set(false);
              }}
            >
              Save
            </Button>
          </div>
        </div>
      {/if}
    </DialogContent>
  </Dialog>
</div>

<style>
  .enhanced-document-uploader {
    @apply w-full;
  }

  .drop-zone {
    @apply border-2 border-dashed border-muted-foreground border-opacity-25 rounded-lg p-8 text-center cursor-pointer transition-colors hover:border-primary hover:border-opacity-50 hover:bg-muted hover:bg-opacity-50;
  }

  .drop-zone.dragging {
    @apply border-primary bg-primary bg-opacity-5;
  }

  .drop-zone-content {
    @apply space-y-2;
  }

  .drop-zone-icon {
    @apply mx-auto text-muted-foreground;
  }

  .drop-zone-title {
    @apply text-lg font-semibold;
  }

  .drop-zone-description {
    @apply text-sm text-muted-foreground;
  }

  .drop-zone-specs {
    @apply text-xs text-muted-foreground;
  }

  .file-list {
    @apply space-y-3;
  }

  .file-item {
    @apply transition-shadow hover:shadow-md;
  }

  .file-info {
    @apply flex items-center space-x-4;
  }

  .file-preview {
    @apply flex-shrink-0 w-12 h-12 rounded-lg bg-muted flex items-center justify-center overflow-hidden;
  }

  .preview-image {
    @apply w-full h-full object-cover;
  }

  .file-details {
    @apply flex-1 min-w-0;
  }

  .file-name {
    @apply font-medium truncate;
  }

  .file-meta {
    @apply text-sm text-muted-foreground;
  }

  .file-progress {
    @apply mt-2;
  }

  .error-message {
    @apply text-sm text-destructive flex items-center mt-2;
  }

  .file-actions {
    @apply flex flex-col items-end space-y-2;
  }

  .action-buttons {
    @apply flex space-x-2;
  }

  .upload-actions {
    @apply flex items-center justify-center;
  }

  .metadata-form {
    @apply p-1;
  }

  .checkbox-group {
    @apply space-y-2 mt-2;
  }

  .dialog-actions {
    @apply flex justify-end space-x-2 mt-6;
  }
</style>
