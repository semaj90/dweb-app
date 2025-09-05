<!--
  Enhanced File Upload Component with Superforms and Zod Validation
  Features: Drag & drop, progress tracking, AI processing, validation, preview
-->
<script lang="ts">
  interface Props {
    caseId: string | undefined ;
    multiple?: any;
    compact?: any;
    initialData: Partial<FileUpload> | undefined ;
    disabled?: any;
    maxFiles?: any;
    maxSizeMB?: any;
    acceptedTypes: string[] ;
  }
  let {
    caseId = undefined,
    multiple = false,
    compact = false,
    initialData = undefined,
    disabled = false,
    maxFiles = multiple ? 10 : 1,
    maxSizeMB = 100,
    acceptedTypes = [
  } = $props();



  import { Alert, AlertDescription } from "$lib/components/ui/alert";
  import { Badge } from "$lib/components/ui/badge";
  import { Button } from "$lib/components/ui/button";
  import {
    Card,
    CardContent,
    CardHeader,
    CardTitle,
  } from "$lib/components/ui/Card";
  import { Checkbox } from "$lib/components/ui/checkbox";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import { Progress } from "$lib/components/ui/progress";
  import Select from "$lib/components/ui/select/Select.svelte";
  import SelectValue from "$lib/components/ui/select/SelectValue.svelte";
  import { SelectContent, SelectItem, SelectTrigger } from "$lib/components/ui/select";
  import { Textarea } from "$lib/components/ui/textarea";
  import {
    defaultFileUploadValues,
    fileUploadSchema,
    formatFileSize,
    getFileCategory,
    type EvidenceType,
    type FileUpload,
  } from "$lib/schemas/file-upload";
  import {
    AlertCircle,
    File as FileIcon,
    FileText,
    Image,
    Loader2,
    Music,
    Upload,
    Video,
    X,
  } from "lucide-svelte";
  import { createEventDispatcher, onMount } from "svelte";
  import { superForm } from "sveltekit-superforms";
  import { zodClient } from "sveltekit-superforms/adapters";

  const dispatch = createEventDispatcher<{
    upload: { files: File[]; formData: FileUpload[] };
    cancel: void
    progress: { progress: number file: string };
  }>();

  // Props
   // string | undefined = undefined;
  let { multiple = $bindable() } = $props(); // false;
  let { compact = $bindable() } = $props(); // false;
  let { initialData = $bindable() } = $props(); // Partial<FileUpload> | undefined = undefined;
  let { disabled = $bindable() } = $props(); // false;
  let { maxFiles = $bindable() } = $props(); // multiple ? 10 : 1;
  let { maxSizeMB = $bindable() } = $props(); // 100;
      "image/*",
    "video/*",
    "audio/*",
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".csv",
    ".zip",
    ".rar",
  ];

  // State
  let fileInput: HTMLInputElement
  let isDragOver = false;
  let selectedFiles: File[] = [];
  let uploadProgress: Record<string, number> = {};
  let previews: Record<string, string> = {};
  let isUploading = false;
  let currentUploadFile = "";

  // Form setup with Superforms and Zod
  const { form, errors, enhance, submit, formId } = superForm(
    { ...defaultFileUploadValues, ...initialData, caseId },
    {
      SPA: true,
  validators: zodClient(fileUploadSchema) as any,
      resetForm: false,
      invalidateAll: false,
      onUpdate: ({ form }) => {
        if (form.valid && selectedFiles.length > 0) {
          handleFormSubmit();
        }
      },
      onError: ({ result }) => {
        console.error("File upload form error:", result);
      },
    }
  );

  // Evidence type options
  const evidenceTypes: { value: EvidenceType label: string }[] = [
    { value: "documents", label: "Documents" },
    { value: "physical_evidence", label: "Physical Evidence" },
    { value: "digital_evidence", label: "Digital Evidence" },
    { value: "photographs", label: "Photographs" },
    { value: "video_recording", label: "Video Recording" },
    { value: "audio_recording", label: "Audio Recording" },
    { value: "witness_testimony", label: "Witness Testimony" },
    { value: "expert_opinion", label: "Expert Opinion" },
    { value: "forensic_analysis", label: "Forensic Analysis" },
    { value: "chain_of_custody", label: "Chain of Custody" },
  ];

  const confidentialityLevels = [
    { value: "public", label: "Public" },
    { value: "standard", label: "Standard" },
    { value: "confidential", label: "Confidential" },
    { value: "classified", label: "Classified" },
    { value: "restricted", label: "Restricted" },
  ];

  // File handling
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    if (!disabled) isDragOver = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragOver = false;

    if (disabled) return;

    const files = Array.from(event.dataTransfer?.files || []);
    addFiles(files);
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files) {
      const files = Array.from(target.files);
      addFiles(files);
    }
  }

  function addFiles(files: File[]) {
    const validFiles = files.filter((file) => validateFile(file));

    if (multiple) {
      const totalFiles = selectedFiles.length + validFiles.length;
      if (totalFiles > maxFiles) {
        validFiles.splice(maxFiles - selectedFiles.length);
      }
      selectedFiles = [...selectedFiles, ...validFiles];
    } else {
      selectedFiles = validFiles.slice(0, 1);
    }

    // Generate previews for images
    validFiles.forEach((file) => {
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => {
          previews[file.name] = e.target?.result as string;
        };
        reader.readAsDataURL(file);
      }
    });

    // Auto-populate form fields based on first file
    if (validFiles.length > 0 && !$form.title) {
      const firstFile = validFiles[0];
      $form.title = firstFile.name.replace(/\.[^/.]+$/, ""); // Remove extension
      $form.fileType = getFileCategory(firstFile.type);

      // Auto-select evidence type based on file type
      if (firstFile.type.startsWith("image/")) {
        $form.evidenceType = "photographs";
      } else if (firstFile.type.startsWith("video/")) {
        $form.evidenceType = "video_recording";
      } else if (firstFile.type.startsWith("audio/")) {
        $form.evidenceType = "audio_recording";
      } else if (
        firstFile.type.includes("pdf") ||
        firstFile.type.includes("document")
      ) {
        $form.evidenceType = "documents";
      }
    }
  }

  function validateFile(file: File): boolean {
    // Check file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      alert(
        `File "${file.name}" is too large. Maximum size is ${maxSizeMB}MB.`
      );
      return false;
    }

    // Check file type
    const isValidType = acceptedTypes.some((type) => {
      if (type.startsWith(".")) {
        return file.name.toLowerCase().endsWith(type.toLowerCase());
      } else if (type.includes("*")) {
        const category = type.split("/")[0];
        return file.type.startsWith(category);
      } else {
        return file.type === type;
      }
    });

    if (!isValidType) {
      alert(`File type "${file.type}" is not supported.`);
      return false;
    }

    return true;
  }

  function removeFile(fileName: string) {
    selectedFiles = selectedFiles.filter((f) => f.name !== fileName);
    delete previews[fileName];
    delete uploadProgress[fileName];
  }

  function getFileIcon(file: File) {
    if (file.type.startsWith("image/")) return Image;
    if (file.type.startsWith("video/")) return Video;
    if (file.type.startsWith("audio/")) return Music;
    if (file.type.includes("pdf") || file.type.includes("document"))
      return FileText;
    return FileIcon;
  }

  async function handleFormSubmit() {
    if (selectedFiles.length === 0) return;

    isUploading = true;
    const formDataArray: FileUpload[] = [];

    try {
      for (const file of selectedFiles) {
        currentUploadFile = file.name;
        uploadProgress[file.name] = 0;

        // Create form data for this file
        const fileData: FileUpload = {
          ...$form,
          file,
        };

        // Simulate upload progress (replace with actual upload logic)
        await simulateUpload(file.name);

        formDataArray.push(fileData);
      }

      dispatch("upload", { files: selectedFiles, formData: formDataArray });

      // Reset form
      selectedFiles = [];
      previews = {};
      uploadProgress = {};

      if (fileInput) {
        fileInput.value = "";
      }
    } catch (error) {
      console.error("Upload failed:", error);
    } finally {
      isUploading = false;
      currentUploadFile = "";
    }
  }

  async function simulateUpload(fileName: string): Promise<void> {
    return new Promise((resolve) => {
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          resolve();
        }
        uploadProgress[fileName] = progress;
        dispatch("progress", { progress, file: fileName });
      }, 100);
    });
  }

  function openFileDialog() {
    if (!disabled) {
      fileInput?.click();
    }
  }

  // Tags management
  let tagInput = "";

  function addTag() {
    if (tagInput.trim() && !$form.tags.includes(tagInput.trim())) {
      $form.tags = [...$form.tags, tagInput.trim()];
      tagInput = "";
    }
  }

  function removeTag(tag: string) {
    $form.tags = $form.tags.filter((t) => t !== tag);
  }

  onMount(() => {
    if (caseId) {
      $form.caseId = caseId;
    }
  });
</script>

<!-- Hidden file input -->
<input
  bind:this={fileInput}
  type="file"
  {multiple}
  accept={acceptedTypes.join(",")}
  on:change={handleFileSelect}
  class="hidden"
/>

<Card class="w-full">
  <CardHeader class="pb-4">
    <CardTitle class="flex items-center gap-2">
      <Upload class="h-5 w-5" />
      {compact ? "Upload Files" : "Evidence Upload"}
    </CardTitle>
  </CardHeader>

  <CardContent class="space-y-6">
    <!-- Drop Zone -->
    <div
      class="border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
             {isDragOver
        ? 'border-primary bg-primary bg-opacity-5'
        : 'border-muted-foreground border-opacity-25 hover:border-primary hover:border-opacity-50'}
             {disabled ? 'opacity-50 cursor-not-allowed' : ''}"
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
      on:drop={handleDrop}
      on:click={openFileDialog}
  on:keydown={(e: KeyboardEvent) => e.key === "Enter" && openFileDialog()}
      role="button"
      tabindex="0"
      aria-label="File upload area"
    >
      {#if isUploading}
        <div class="flex flex-col items-center gap-4">
          <Loader2 class="h-8 w-8 animate-spin text-primary" />
          <div class="space-y-2">
            <p class="text-sm font-medium">Uploading {currentUploadFile}...</p>
            {#if uploadProgress[currentUploadFile] !== undefined}
              <Progress
                value={uploadProgress[currentUploadFile]}
                class="w-64"
              />
              <p class="text-xs text-muted-foreground">
                {Math.round(uploadProgress[currentUploadFile])}% complete
              </p>
            {/if}
          </div>
        </div>
      {:else}
        <div class="space-y-4">
          <Upload class="h-12 w-12 mx-auto text-muted-foreground" />
          <div>
            <p class="text-lg font-medium">
              {selectedFiles.length > 0
                ? "Add more files"
                : "Drop files here or click to browse"}
            </p>
            <p class="text-sm text-muted-foreground mt-1">
              Supports: Images, Videos, Audio, Documents (Max {maxSizeMB}MB
              each)
            </p>
          </div>
        </div>
      {/if}
    </div>

    <!-- Selected Files Preview -->
    {#if selectedFiles.length > 0}
      <div class="space-y-3">
        <h4 class="font-medium">Selected Files ({selectedFiles.length})</h4>
        <div class="grid gap-3">
          {#each selectedFiles as file (file.name)}
            <div
              class="flex items-center gap-3 p-3 border rounded-lg bg-muted bg-opacity-50"
            >
              <div class="flex-shrink-0">
                {#if previews[file.name]}
                  <img
                    src={previews[file.name]}
                    alt={file.name}
                    class="h-12 w-12 object-cover rounded"
                  />
                {:else}
                  <svelte:component
                    this={getFileIcon(file)}
                    class="h-12 w-12 text-muted-foreground"
                  />
                {/if}
              </div>

              <div class="flex-1 min-w-0">
                <p class="font-medium truncate">{file.name}</p>
                <p class="text-sm text-muted-foreground">
                  {formatFileSize(file.size)} • {file.type}
                </p>
                {#if uploadProgress[file.name] !== undefined}
                  <Progress value={uploadProgress[file.name]} class="mt-2" />
                {/if}
              </div>

              <Button
                variant="ghost"
                size="sm"
                on:click={() => removeFile(file.name)}
                disabled={isUploading}
                class="flex-shrink-0"
              >
                <X class="h-4 w-4" />
              </Button>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    {#if !compact && selectedFiles.length > 0}
      <!-- Form Fields -->
  <form use:enhance class="space-y-6" id={formId as any}>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Title -->
          <div class="space-y-2">
            <Label for="title">Title *</Label>
            <Input
              id="title"
              bind:value={$form.title}
              placeholder="Enter evidence title"
              disabled={isUploading}
              class:border-destructive={$errors.title}
            />
            {#if $errors.title}
              <p class="text-sm text-destructive">{$errors.title[0]}</p>
            {/if}
          </div>

          <!-- Evidence Type -->
          <div class="space-y-2">
            <Label for="evidenceType">Evidence Type *</Label>
            <Select bind:value={$form.evidenceType} disabled={isUploading}>
              <SelectTrigger class:border-destructive={$errors.evidenceType}>
                <SelectValue placeholder="Select evidence type" />
              </SelectTrigger>
              <SelectContent>
                {#each evidenceTypes as type}
                  <SelectItem value={type.value}>{type.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
            {#if $errors.evidenceType}
              <p class="text-sm text-destructive">{$errors.evidenceType[0]}</p>
            {/if}
          </div>

          <!-- Confidentiality Level -->
          <div class="space-y-2">
            <Label for="confidentialityLevel">Confidentiality Level</Label>
            <Select
              bind:value={$form.confidentialityLevel}
              disabled={isUploading}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {#each confidentialityLevels as level}
                  <SelectItem value={level.value}>{level.label}</SelectItem>
                {/each}
              </SelectContent>
            </Select>
          </div>

          <!-- Collected By -->
          <div class="space-y-2">
            <Label for="collectedBy">Collected By</Label>
            <Input
              id="collectedBy"
              bind:value={$form.collectedBy}
              placeholder="Officer/investigator name"
              disabled={isUploading}
            />
          </div>
        </div>

        <!-- Description -->
        <div class="space-y-2">
          <Label for="description">Description</Label>
          <Textarea
            id="description"
            bind:value={$form.description}
            placeholder="Describe the evidence and its relevance to the case"
            rows="3"
            disabled={isUploading}
            class:border-destructive={$errors.description}
          />
          {#if $errors.description}
            <p class="text-sm text-destructive">{$errors.description[0]}</p>
          {/if}
        </div>

        <!-- Location -->
        <div class="space-y-2">
          <Label for="location">Collection Location</Label>
          <Input
            id="location"
            bind:value={$form.location}
            placeholder="Where was this evidence collected?"
            disabled={isUploading}
          />
        </div>

        <!-- Tags -->
        <div class="space-y-2">
          <Label for="tags">Tags</Label>
          <div class="flex gap-2">
            <Input
              bind:value={tagInput}
              placeholder="Add a tag"
              disabled={isUploading}
              on:keydown={(e) =>
                e.key === "Enter" && (e.preventDefault(), addTag())}
            />
            <Button
              type="button"
              variant="outline"
              on:click={addTag}
              disabled={isUploading}
            >
              Add
            </Button>
          </div>
          {#if $form.tags.length > 0}
            <div class="flex flex-wrap gap-2 mt-2">
              {#each $form.tags as tag}
                <Badge variant="secondary" class="gap-1">
                  {tag}
                  <Button
                    variant="ghost"
                    size="sm"
                    class="h-auto p-0 hover:bg-transparent"
                    on:click={() => removeTag(tag)}
                    disabled={isUploading}
                  >
                    <X class="h-3 w-3" />
                  </Button>
                </Badge>
              {/each}
            </div>
          {/if}
        </div>

        <!-- AI Processing Options -->
        <div class="space-y-3">
          <Label>AI Processing Options</Label>
          <div class="grid grid-cols-2 gap-3">
            <div class="flex items-center space-x-2">
              <Checkbox
                bind:checked={$form.enableAiAnalysis}
                id="enableAiAnalysis"
                disabled={isUploading}
              />
              <Label for="enableAiAnalysis" class="text-sm"
                >Enable AI Analysis</Label
              >
            </div>
            <div class="flex items-center space-x-2">
              <Checkbox
                bind:checked={$form.enableOcr}
                id="enableOcr"
                disabled={isUploading}
              />
              <Label for="enableOcr" class="text-sm">OCR Text Extraction</Label>
            </div>
            <div class="flex items-center space-x-2">
              <Checkbox
                bind:checked={$form.enableEmbeddings}
                id="enableEmbeddings"
                disabled={isUploading}
              />
              <Label for="enableEmbeddings" class="text-sm"
                >Generate Embeddings</Label
              >
            </div>
            <div class="flex items-center space-x-2">
              <Checkbox
                bind:checked={$form.enableSummarization}
                id="enableSummarization"
                disabled={isUploading}
              />
              <Label for="enableSummarization" class="text-sm"
                >AI Summarization</Label
              >
            </div>
          </div>
        </div>

        <!-- Admissibility -->
        <div class="flex items-center space-x-2">
          <Checkbox
            bind:checked={$form.isAdmissible}
            id="isAdmissible"
            disabled={isUploading}
          />
          <Label for="isAdmissible">Mark as admissible evidence</Label>
        </div>
      </form>
    {/if}

    <!-- Action Buttons -->
    <div class="flex justify-between items-center pt-4 border-t">
      <p class="text-sm text-muted-foreground">
        {selectedFiles.length} file{selectedFiles.length !== 1 ? "s" : ""} selected
        {#if multiple}(max {maxFiles}){/if}
      </p>

      <div class="flex gap-2">
        <Button
          variant="outline"
          on:click={() => dispatch("cancel")}
          disabled={isUploading}
        >
          Cancel
        </Button>

        <Button
          on:click={handleFormSubmit}
          disabled={selectedFiles.length === 0 ||
            isUploading ||
            Object.keys($errors).length > 0}
          class="min-w-24"
        >
          {#if isUploading}
            <Loader2 class="h-4 w-4 animate-spin mr-2" />
            Uploading...
          {:else}
            <Upload class="h-4 w-4 mr-2" />
            Upload {selectedFiles.length} file{selectedFiles.length !== 1
              ? "s"
              : ""}
          {/if}
        </Button>
      </div>
    </div>

    <!-- Validation Errors -->
    {#if Object.keys($errors).length > 0 && selectedFiles.length > 0}
      <Alert variant="destructive">
        <AlertCircle class="h-4 w-4" />
        <AlertDescription>
          Please fix the following errors before uploading:
          <ul class="mt-2 list-disc list-inside">
            {#each Object.entries($errors) as [field, messages]}
              <li>{field}: {messages[0]}</li>
            {/each}
          </ul>
        </AlertDescription>
      </Alert>
    {/if}
  </CardContent>
</Card>

<style>
  .hidden {
    display: none
  }
</style>
