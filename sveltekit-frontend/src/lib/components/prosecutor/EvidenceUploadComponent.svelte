<!--
Enhanced Evidence Upload Component for Prosecutors
Features: MinIO storage, AI analysis, multi-file support, drag-drop
-->
<script lang="ts">
  import { $props, $state } from 'svelte';

  import type { Props } from "$lib/types/global";
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Badge } from '$lib/components/ui/badge';
  import { Progress } from '$lib/components/ui/progress';
  import { webGPUProcessor } from '$lib/services/webgpu-vector-processor';
  import { 
    Upload, 
    FileText, 
    Image, 
    Film, 
    Mic, 
    Archive,
    CheckCircle,
    AlertCircle,
    X,
    Eye,
    Brain,
    Zap
  } from 'lucide-svelte';


  let {
    caseId,
    allowedTypes = ['application/pdf', 'image/*', 'video/*', 'text/*'],
    maxFiles = 10,
    enableAI = true,
    enableWebGPU = true,
    onUploadComplete
  }: Props = $props();

  // State management
  let selectedFiles: File[] = $state([]);
  let uploading = $state(false);
  let uploadProgress = $state(0);
  let uploadResults: any[] = $state([]);
  let dragActive = $state(false);

  // Evidence form data
  let evidenceTitle = $state('');
  let evidenceDescription = $state('');
  let evidenceType = $state('document');
  let collectedBy = $state('');
  let location = $state('');
  let tags = $state('');
  let isAdmissible = $state(true);

  // File type icons
  const getFileIcon = (mimeType: string) => {
    if (mimeType.startsWith('image/')) return Image;
    if (mimeType.startsWith('video/')) return Film;
    if (mimeType.startsWith('audio/')) return Mic;
    if (mimeType.includes('pdf') || mimeType.startsWith('text/')) return FileText;
    if (mimeType.includes('zip') || mimeType.includes('rar')) return Archive;
    return FileText;
  };

  // Drag and drop handlers
  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    dragActive = true;
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    dragActive = false;
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    dragActive = false;
    
    const files = Array.from(e.dataTransfer?.files || []);
    addFiles(files);
  };

  // File selection handlers
  const handleFileSelect = (e: Event) => {
    const target = e.target as HTMLInputElement;
    const files = Array.from(target.files || []);
    addFiles(files);
  };

  const addFiles = (files: File[]) => {
    const newFiles = files.filter(file => {
      // Check file type
      const isAllowed = allowedTypes.some(type => 
        type === '*/*' || 
        type.endsWith('/*') ? file.type.startsWith(type.replace('/*', '/')) : 
        file.type === type
      );
      
      // Check if not already selected
      const notDuplicate = !selectedFiles.some(f => f.name === file.name && f.size === file.size);
      
      return isAllowed && notDuplicate;
    });

    if (selectedFiles.length + newFiles.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`);
      return;
    }

    selectedFiles = [...selectedFiles, ...newFiles];
  };

  const removeFile = (index: number) => {
    selectedFiles = selectedFiles.filter((_, i) => i !== index);
  };

  // Upload process with WebGPU acceleration
  const uploadEvidence = async () => {
    if (selectedFiles.length === 0 || !evidenceTitle.trim()) {
      alert('Please select files and provide a title');
      return;
    }

    uploading = true;
    uploadProgress = 0;
    uploadResults = [];

    try {
      // Use WebGPU batch processing if available
      if (enableWebGPU && selectedFiles.length > 1) {
        console.log('🚀 Using WebGPU batch processing');
        const batchResults = await webGPUProcessor.batchProcessEvidence(selectedFiles, caseId);
        uploadResults = batchResults;
        uploadProgress = 100;
      } else {
        // Process files individually
        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          
          const formData = new FormData();
          formData.append('file', file);
          formData.append('caseId', caseId);
          formData.append('title', `${evidenceTitle} - ${file.name}`);
          formData.append('description', evidenceDescription);
          formData.append('type', evidenceType);
          formData.append('tags', JSON.stringify(tags.split(',').map(t => t.trim()).filter(Boolean)));
          formData.append('collectedBy', collectedBy);
          formData.append('location', location);
          formData.append('isAdmissible', isAdmissible.toString());

          let response: Response;
        try {
          response = await fetch('/api/evidence/upload', {
            method: 'POST',
            body: formData
          });
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        } catch (error) {
          console.error('Fetch failed:', error);
          throw error;
        }

          const result = await response.json();
          uploadResults.push({ file: file.name, ...result });
          
          uploadProgress = ((i + 1) / selectedFiles.length) * 100;
        }
      }

      // Clear form
      selectedFiles = [];
      evidenceTitle = '';
      evidenceDescription = '';
      tags = '';
      
      onUploadComplete?.(uploadResults);
      
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed. Please try again.');
    } finally {
      uploading = false;
    }
  };

  // YOLO object detection preview (placeholder)
  const analyzeImageWithYOLO = async (file: File) => {
    // This would integrate with YOLO for object detection
    return {
      objects: ['person', 'document', 'weapon'],
      confidence: 0.92,
      boundingBoxes: []
    };
  };
</script>

<Card class="w-full max-w-4xl mx-auto">
  <CardHeader>
    <CardTitle class="flex items-center gap-2">
      <Upload class="w-5 h-5" />
      Evidence Upload - Prosecutor Workflow
      {#if enableWebGPU}
        <Badge variant="secondary" class="ml-2">
          <Zap class="w-3 h-3 mr-1" />
          GPU Accelerated
        </Badge>
      {/if}
    </CardTitle>
  </CardHeader>

  <CardContent class="space-y-6">
    <!-- Evidence Metadata Form -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="space-y-2">
        <Label for="evidence-title">Evidence Title *</Label>
        <Input 
          id="evidence-title" 
          bind:value={evidenceTitle}
          placeholder="e.g., Contract Agreement, Crime Scene Photo"
        />
      </div>
      
      <div class="space-y-2">
        <Label for="evidence-type">Evidence Type</Label>
        <select id="evidence-type" bind:value={evidenceType} class="w-full p-2 border rounded">
          <option value="document">Document</option>
          <option value="physical">Physical Evidence</option>
          <option value="digital">Digital Evidence</option>
          <option value="witness">Witness Statement</option>
          <option value="expert">Expert Report</option>
        </select>
      </div>
      
      <div class="space-y-2">
        <Label for="collected-by">Collected By</Label>
        <Input 
          id="collected-by" 
          bind:value={collectedBy}
          placeholder="Officer name or department"
        />
      </div>
      
      <div class="space-y-2">
        <Label for="location">Location</Label>
        <Input 
          id="location" 
          bind:value={location}
          placeholder="Collection location"
        />
      </div>
    </div>

    <div class="space-y-2">
      <Label for="description">Description</Label>
      <Textarea 
        id="description" 
        bind:value={evidenceDescription}
        placeholder="Detailed description of the evidence"
        rows={3}
      />
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div class="space-y-2">
        <Label for="tags">Tags (comma-separated)</Label>
        <Input 
          id="tags" 
          bind:value={tags}
          placeholder="contract, fraud, witness, DNA"
        />
      </div>
      
      <div class="flex items-center space-x-2">
        <input 
          type="checkbox" 
          id="admissible" 
          bind:checked={isAdmissible}
          class="w-4 h-4"
        />
        <Label for="admissible">Evidence is admissible in court</Label>
      </div>
    </div>

    <!-- File Upload Area -->
    <div 
      class="border-2 border-dashed rounded-lg p-8 text-center transition-colors {dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}"
      ondragover={handleDragOver as any}
      ondragleave={handleDragLeave as any}
      ondrop={handleDrop as any}
    >
      {#if selectedFiles.length === 0}
        <Upload class="mx-auto w-12 h-12 text-gray-400 mb-4" />
        <h3 class="text-lg font-medium text-gray-900 mb-2">
          Drop evidence files here or click to browse
        </h3>
        <p class="text-sm text-gray-500 mb-4">
          Supports PDFs, images, videos, documents ({maxFiles} files max)
        </p>
        <input
          type="file"
          multiple
          accept={allowedTypes.join(',')}
          onchange={handleFileSelect as any}
          class="hidden"
          id="file-input"
        />
        <Button variant="outline" onclick={() => document.getElementById('file-input')?.click()}>
          Select Files
        </Button>
      {:else}
        <div class="space-y-3">
          <h3 class="text-lg font-medium">Selected Files ({selectedFiles.length}/{maxFiles})</h3>
          
          {#each selectedFiles as file, index}
            <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div class="flex items-center space-x-3">
                <svelte:component this={getFileIcon(file.type)} class="w-5 h-5 text-blue-500" />
                <div>
                  <p class="font-medium text-sm">{file.name}</p>
                  <p class="text-xs text-gray-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type}
                  </p>
                </div>
              </div>
              
              <div class="flex items-center space-x-2">
                {#if enableAI}
                  <Badge variant="secondary" class="text-xs">
                    <Brain class="w-3 h-3 mr-1" />
                    AI Analysis
                  </Badge>
                {/if}
                <Button 
                  variant="ghost" 
                  size="sm"
                  onclick={() => removeFile(index)}
                >
                  <X class="w-4 h-4" />
                </Button>
              </div>
            </div>
          {/each}

          <div class="flex justify-center space-x-3 mt-4">
            <input
              type="file"
              multiple
              accept={allowedTypes.join(',')}
              onchange={handleFileSelect as any}
              class="hidden"
              id="add-more-files"
            />
            <Button 
              variant="outline" 
              onclick={() => document.getElementById('add-more-files')?.click()}
              disabled={selectedFiles.length >= maxFiles}
            >
              Add More Files
            </Button>
            <Button onclick={uploadEvidence} disabled={uploading || !evidenceTitle.trim()}>
              {#if uploading}
                Processing...
              {:else}
                Upload & Analyze Evidence
              {/if}
            </Button>
          </div>
        </div>
      {/if}
    </div>

    <!-- Upload Progress -->
    {#if uploading}
      <div class="space-y-2">
        <div class="flex justify-between text-sm">
          <span>Uploading to MinIO & processing with AI...</span>
          <span>{Math.round(uploadProgress)}%</span>
        </div>
        <Progress value={uploadProgress} class="w-full" />
      </div>
    {/if}

    <!-- Upload Results -->
    {#if uploadResults.length > 0}
      <div class="space-y-3">
        <h3 class="text-lg font-medium flex items-center gap-2">
          <CheckCircle class="w-5 h-5 text-green-500" />
          Upload Complete - Evidence Processed
        </h3>
        
        {#each uploadResults as result}
          <div class="p-4 bg-green-50 border border-green-200 rounded-lg">
            <div class="flex justify-between items-start">
              <div>
                <h4 class="font-medium text-green-900">{result.fileName || result.file}</h4>
                <p class="text-sm text-green-700">
                  Stored in MinIO • Embedded in Qdrant • AI Analyzed
                </p>
                {#if result.aiAnalysis}
                  <div class="mt-2 space-y-1">
                    <p class="text-xs text-green-600">
                      <strong>AI Summary:</strong> {result.aiAnalysis.summary?.substring(0, 100)}...
                    </p>
                    {#if result.aiAnalysis.prosecutionRelevance}
                      <Badge 
                        variant={result.aiAnalysis.prosecutionRelevance === 'high' ? 'destructive' : 'secondary'}
                        class="text-xs"
                      >
                        {result.aiAnalysis.prosecutionRelevance} relevance
                      </Badge>
                    {/if}
                  </div>
                {/if}
              </div>
              
              <div class="flex flex-col items-end space-y-1">
                <Badge variant="secondary" class="text-xs">
                  {result.embedding || 'Vector stored'}
                </Badge>
                {#if result.qdrantId}
                  <Badge variant="outline" class="text-xs">
                    Searchable
                  </Badge>
                {/if}
              </div>
            </div>
          </div>
        {/each}

        <div class="flex justify-center mt-4">
          <Button 
            variant="outline" 
            onclick={() => {
              uploadResults = [];
              selectedFiles = [];
            }}
          >
            Upload More Evidence
          </Button>
        </div>
      </div>
    {/if}

    <!-- AI Features Info -->
    {#if enableAI}
      <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <div class="flex items-start gap-3">
          <Brain class="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <h4 class="font-medium text-blue-900">AI-Powered Evidence Processing</h4>
            <ul class="text-sm text-blue-700 mt-1 space-y-1">
              <li>• Automatic text extraction and OCR</li>
              <li>• Legal relevance analysis with Gemma3Legal</li>
              <li>• Vector embeddings for semantic search</li>
              <li>• YOLO object detection for images/videos</li>
              <li>• Qdrant storage with payload filters</li>
              {#if enableWebGPU}
                <li>• WebGPU acceleration for vector operations</li>
              {/if}
            </ul>
          </div>
        </div>
      </div>
    {/if}
  </CardContent>
</Card>

<style>
  .drag-active {
    border-color: #3b82f6;
    background-color: #eff6ff;
  }
</style>



