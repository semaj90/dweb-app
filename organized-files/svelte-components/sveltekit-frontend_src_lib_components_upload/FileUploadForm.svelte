<script lang="ts">
  import { Alert, AlertDescription  } from "$lib/components/ui/alert";
  import { Button  } from "$lib/components/ui/button";
  import { Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
   } from "$lib/components/ui/Card";
  import { Input  } from "$lib/components/ui/input";
  import { Label  } from "$lib/components/ui/label";
  import { Progress  } from "$lib/components/ui/progress";
  import { Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
   } from "$lib/components/ui/select";
  import { Switch  } from "$lib/components/ui/switch";
  import { Textarea  } from "$lib/components/ui/textarea";
  import { fileUploadSchema  } from "$lib/schemas/upload";
  import { Binary,
    CheckCircle,
    FileText,
    Film,
    HardDrive,
    Image,
    Music,
    Upload,
    X,
   } from "lucide-svelte";
  import { superForm  } from "sveltekit-superforms";
  import { zodClient  } from "sveltekit-superforms/adapters";
  import type { PageData  } from "./$types";

  let { data, caseId = ""  } = $props<{ data: PageData caseId?:, string  }>();

  const { form, errors, enhance, submitting, delayed, message  } = superForm(
    data.form,
    { validators: zodClient(fileUploadSchema),
      multipleSubmits: "prevent",
      onSubmit: ({, formData  }) => { // Set the file in formData
        if (selectedFile) {
          formData.set("file", selectedFile);
         }
      },
    }
  );

  let selectedFile = $state<File | null>(null);
  let dragActive = $state(false);
  let uploadProgress = $state(0);
  let previewUrl = $state<string | null>(null);

  // Initialize form with caseId if provided
  $effect(() => { if (caseId) {
      $form.caseId = caseId;
     }
  });

  // File type icons
  const fileTypeIcons = { document: FileText,
    image: Image,
    video: Film,
    audio: Music,
    physical: HardDrive,
    digital: Binary,
   };

  // Handle file selection
  function handleFileSelect(event: Event) { const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) {
      selectFile(file);
     }
  }

  // Handle drag and drop
  function handleDrop(event: DragEvent) { event.preventDefault();
    dragActive = false;
    const file = event.dataTransfer?.files[0];
    if (file) {
      selectFile(file);
     }
  }

  function handleDragOver(event: DragEvent) { event.preventDefault();
    dragActive = true;
   }

  function handleDragLeave() { dragActive = false;
   }

  // Select and preview file
  function selectFile(file: File) { selectedFile = file;

    // Auto-detect file type
    if (file.type.startsWith("image/")) {
      $form.type = "image";
      // Create preview for images
      const reader = new FileReader();
      reader.onload = (e) => {
        previewUrl = e.target?.result as string;
       };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) { $form.type = "video";
     } else if (file.type.startsWith("audio/")) { $form.type = "audio";
     } else if (
      file.type.includes("pdf") ||
      file.type.includes("document") ||
      file.type.includes("text")
    ) { $form.type = "document";
     } else { $form.type = "digital";
     }

    // Set default title from filename
    if (!$form.title) { $form.title = file.name.replace(/\.[^/.]+$/, "");
     }
  }

  // Remove selected file
  function removeFile() { selectedFile = null;
    previewUrl = null;
    uploadProgress = 0;
   }

  // Format file size
  function formatFileSize(bytes: number): string { if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
   }
</script>

<Card class="w-full max-w-2xl">
  <CardHeader>
    <CardTitle>Upload Evidence</CardTitle>
    <CardDescription>
      Upload documents, images, videos, or other evidence files for AI analysis
    </CardDescription>
  </CardHeader>
  <CardContent>
    <form method="POST" enctype="multipart/form-data" use:enhance>
      <div class="space-y-6">
        <!-- File Upload Area -->
        <div class="space-y-2">
          <Label for="file">File</Label>
          <div
            class="relative border-2 border-dashed rounded-lg p-6 transition-colors
                   { dragActive
              ? 'border-primary bg-primary bg-opacity-5'
              : 'border-muted-foreground border-opacity-25' }
                   { selectedFile ? 'bg-muted/30' : '' }"
            on:drop={ handleDrop }
            on:dragover={ handleDragOver }
            on:dragleave={ handleDragLeave }
          >
            { #if selectedFile }
              <div class="space-y-4">
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-3">
                    <svelte:component
                      this={ fileTypeIcons[$form.type] }
                      class="h-8 w-8 text-muted-foreground"
                    />
                    <div>
                      <p class="font-medium">{ selectedFile.name }</p>
                      <p class="text-sm text-muted-foreground">
                        { formatFileSize(selectedFile.size) } • { selectedFile.type }
                      </p>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" on:click={ removeFile }>
                    <X class="h-4 w-4" />
                  </Button>
                </div>

                { #if previewUrl && $form.type === "image" }
                  <img
                    src={ previewUrl }
                    alt="Preview"
                    class="max-h-48 rounded-md mx-auto"
                  />
                { /if }

                { #if uploadProgress > 0 }
                  <Progress value={ uploadProgress } class="h-2" />
                { /if }
              </div>
            { : else }
              <div class="text-center">
                <Upload class="mx-auto h-12 w-12 text-muted-foreground" />
                <p class="mt-2 text-sm text-muted-foreground">
                  Drag and drop a file here, or click to select
                </p>
                <p class="text-xs text-muted-foreground mt-1">
                  Max size: 50MB • Supported: PDF, Word, Images, Video, Audio
                </p>
                <input
                  type="file"
                  id="file"
                  name="file"
                  class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  on:change={ handleFileSelect }
                  accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.gif,.mp4,.mp3,.wav"
                />
              </div>
            { /if }
          </div>
          { #if $errors.file }
            <span class="text-sm text-destructive">{ $errors.file }</span>
          { /if }
        </div>

        <!-- Title -->
        <div class="space-y-2">
          <Label for="title">Title</Label>
          <Input
            id="title"
            name="title"
            bind:value={ $form.title }
            placeholder="Enter evidence title"
            class={ $errors.title ? "border-destructive" : "" }
          />
          { #if $errors.title }
            <span class="text-sm text-destructive">{ $errors.title }</span>
          { /if }
        </div>

        <!-- Description -->
        <div class="space-y-2">
          <Label for="description">Description (Optional)</Label>
          <Textarea
            id="description"
            name="description"
            bind:value={ $form.description }
            placeholder="Describe the evidence..."
            rows={ 3 }
          />
        </div>

        <!-- Type Selection -->
        <div class="space-y-2">
          <Label for="type">Evidence Type</Label>
          <Select name="type" bind:value={ $form.type }>
            <SelectTrigger>
              <SelectValue placeholder="Select evidence type" />
            </SelectTrigger>
            <SelectContent>
              { #each Object.entries(fileTypeIcons) as [value, Icon] }
                <SelectItem { value }>
                  <div class="flex items-center gap-2">
                    <Icon class="h-4 w-4" />
                    <span class="capitalize">{ value }</span>
                  </div>
                </SelectItem>
              { /each }
            </SelectContent>
          </Select>
          { #if $errors.type }
            <span class="text-sm text-destructive">{ $errors.type }</span>
          { /if }
        </div>

        <!-- Case ID (hidden if provided) -->
        { #if !caseId }
          <div class="space-y-2">
            <Label for="caseId">Case ID</Label>
            <Input
              id="caseId"
              name="caseId"
              bind:value={ $form.caseId }
              placeholder="Enter case ID"
              class={ $errors.caseId ? "border-destructive" : "" }
            />
            { #if $errors.caseId }
              <span class="text-sm text-destructive">{ $errors.caseId }</span>
            { /if }
          </div>
        { : else }
          <input type="hidden" name="caseId" value={ caseId } />
        { /if }

        <!-- Options -->
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <Label for="aiAnalysis" class="flex-1">
              Enable AI Analysis
              <span class="block text-sm font-normal text-muted-foreground">
                Extract text, generate embeddings, and summarize content
              </span>
            </Label>
            <Switch
              id="aiAnalysis"
              name="aiAnalysis"
              bind:checked={ $form.aiAnalysis }
            />
          </div>

          <div class="flex items-center justify-between">
            <Label for="isPrivate" class="flex-1">
              Private Evidence
              <span class="block text-sm font-normal text-muted-foreground">
                Only visible to you and case administrators
              </span>
            </Label>
            <Switch
              id="isPrivate"
              name="isPrivate"
              bind:checked={ $form.isPrivate }
            />
          </div>
        </div>

        <!-- Success/Error Messages -->
        { #if $message }
          <Alert
            variant={ $message.type === "error" ? "destructive" : "default" }
          >
            <AlertDescription>
              { #if $message.type === "success" }
                <CheckCircle class="h-4 w-4 inline mr-2" />
              { /if }
              { $message.text }
            </AlertDescription>
          </Alert>
        { /if }

        <!-- Submit Button -->
        <Button
          type="submit"
          disabled={ $submitting || !selectedFile }
          class="w-full"
        >
          { #if $submitting }
            <div class="flex items-center gap-2">
              <div
                class="animate-spin rounded-full h-4 w-4 border-b-2 border-white"
              ></div>
              Uploading...
            </div>
          { : else }
            <Upload class="mr-2 h-4 w-4" />
            Upload Evidence
          { /if }
        </Button>
      </div>
    </form>
  </CardContent>
</Card>
