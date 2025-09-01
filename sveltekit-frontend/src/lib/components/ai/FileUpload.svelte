<script lang="ts">
  import Button from "$lib/components/ui/button/Button.svelte";
  import Input from "$lib/components/ui/Input.svelte";
  import Label from "$lib/components/ui/Label.svelte";
  import Progress from "$lib/components/ui/progress/Progress.svelte";
  import Card from "$lib/components/ui/Card/Card.svelte";
  import Alert from "$lib/components/ui/alert/Alert.svelte";
  import AlertDescription from "$lib/components/ui/alert/AlertDescription.svelte";
  import AlertTitle from "$lib/components/ui/alert/AlertTitle.svelte";
  import { FileUp, BrainCircuit, Search, Loader2 } from "lucide-svelte";
  
  // Feedback Integration
  import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';

  // Svelte 5 state management
  let files = $state<FileList>();
  let verboseMode = $state(false);
  let thinkingMode = $state(false);
  let isUploading = $state(false);
  let uploadProgress = $state(0);
  let error = $state<string | null>(null);
  let analysisResult = $state<any>(null);

  // Feedback integration
let feedbackIntegration = $state<any;
  let currentInteractionId: string | null >(null);
let uploadStartTime = $state(0);

  async function handleUpload() {
    if (!files || files.length === 0) {
      error = "Please select a file to upload.";
      return;
    }

    isUploading = true;
    error = null;
    analysisResult = null;
    uploadProgress = 0;
    uploadStartTime = Date.now();

    // Track upload interaction for feedback
    currentInteractionId = feedbackIntegration?.triggerFeedback({
      filename: files[0].name,
      fileSize: files[0].size,
      fileType: files[0].type,
      verboseMode,
      thinkingMode,
      uploadStarted: new Date().toISOString()
    });

    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("verbose", verboseMode.toString());
    formData.append("thinking", thinkingMode.toString());

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents/upload", true);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percentComplete = (event.loaded / event.total) * 100;
        uploadProgress = percentComplete;
      }
    };

    xhr.onload = () => {
      isUploading = false;
      const processingTime = Date.now() - uploadStartTime;
      
      if (xhr.status === 200) {
        analysisResult = JSON.parse(xhr.responseText);
        uploadProgress = 100;
        
        // Track successful upload for feedback
        if (currentInteractionId && feedbackIntegration) {
          feedbackIntegration.markCompleted({
            success: true,
            processingTime,
            responseSize: xhr.responseText.length,
            analysisResultKeys: Object.keys(analysisResult).join(','),
            uploadCompleted: new Date().toISOString()
          });
        }
      } else {
        const errorText = JSON.parse(xhr.responseText).error || "An unknown error occurred.";
        error = errorText;
        
        // Track failed upload for feedback
        if (currentInteractionId && feedbackIntegration) {
          feedbackIntegration.markFailed({
            httpStatus: xhr.status,
            errorMessage: errorText,
            processingTime,
            uploadFailed: new Date().toISOString()
          });
        }
      }
    };

    xhr.onerror = () => {
      isUploading = false;
      const errorMsg = "Upload failed. Please check your network connection.";
      error = errorMsg;
      
      // Track network error for feedback
      if (currentInteractionId && feedbackIntegration) {
        feedbackIntegration.markFailed({
          errorType: 'network_error',
          errorMessage: errorMsg,
          processingTime: Date.now() - uploadStartTime,
          networkError: true
        });
      }
    };

    xhr.send(formData);
  }
</script>

<FeedbackIntegration
  bind:this={feedbackIntegration}
  interactionType="document_upload"
  ratingType="ui_experience"
  priority="medium"
  context={{ component: 'FileUpload' }}
  let:feedback
>
<Card class="w-full max-w-2xl mx-auto">
  <CardHeader>
    <CardTitle class="flex items-center gap-2">
      <FileUp />
      Document Upload and Analysis
    </CardTitle>
  </CardHeader>
  <CardContent class="space-y-6">
    <div class="grid w-full items-center gap-1.5">
      <Label for_="file-upload">PDF or XML Document</Label>
      <Input id="file-upload" type="file" bind:files accept=".pdf,.xml" />
    </div>

    <div class="flex items-center space-x-4">
      <div class="flex items-center gap-2">
        <input type="checkbox" id="verbose-mode" bind:checked={verboseMode} />
        <Label for_="verbose-mode" class="flex items-center gap-1"><BrainCircuit size={16} /> Verbose Mode</Label>
      </div>
      <div class="flex items-center gap-2">
        <input type="checkbox" id="thinking-mode" bind:checked={thinkingMode} />
        <Label for_="thinking-mode" class="flex items-center gap-1"><Search size={16} /> Thinking Mode</Label>
      </div>
    </div>

    <Button on:on:click={handleUpload} disabled={isUploading} class="w-full">
      {#if isUploading}
        <Loader2 class="mr-2 h-4 w-4 animate-spin" />
        Uploading...
      {:else}
        <FileUp class="mr-2 h-4 w-4" />
        Upload and Analyze
      {/if}
    </Button>

    {#if isUploading}
      <Progress value={uploadProgress} class="w-full" />
    {/if}

    {#if error}
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    {/if}

    {#if analysisResult}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <pre class="whitespace-pre-wrap text-sm">{JSON.stringify(analysisResult, null, 2)}</pre>
        </CardContent>
      </Card>
    {/if}
  </CardContent>
</Card>
</FeedbackIntegration>


