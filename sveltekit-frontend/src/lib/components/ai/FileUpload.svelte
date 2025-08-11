<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import { Progress } from "$lib/components/ui/progress";
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/Card";
  import { Alert, AlertDescription, AlertTitle } from "$lib/components/ui/alert";
  import { FileUp, BrainCircuit, Search, Loader2 } from "lucide-svelte";

  // Svelte 5 state management
  let files = $state<FileList>();
  let verboseMode = $state(false);
  let thinkingMode = $state(false);
  let isUploading = $state(false);
  let uploadProgress = $state(0);
  let error = $state<string | null>(null);
  let analysisResult = $state<any>(null);

  async function handleUpload() {
    if (!files || files.length === 0) {
      error = "Please select a file to upload.";
      return;
    }

    isUploading = true;
    error = null;
    analysisResult = null;
    uploadProgress = 0;

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
      if (xhr.status === 200) {
        analysisResult = JSON.parse(xhr.responseText);
        uploadProgress = 100;
      } else {
        error = JSON.parse(xhr.responseText).error || "An unknown error occurred.";
      }
    };

    xhr.onerror = () => {
      isUploading = false;
      error = "Upload failed. Please check your network connection.";
    };

    xhr.send(formData);
  }
</script>

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

    <Button on:click={handleUpload} disabled={isUploading} class="w-full">
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
