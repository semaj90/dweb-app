<script lang="ts">
  // Use modular components and types
  import { FileUpload, Button, Badge, Progress, Card } from '$lib/components/ui/modular';
  import type { UploadFile } from '$lib/components/ui/modular/types';
  import FileUploadProgress from './upload/FileUploadProgress.svelte';

  interface FileUpload {
    id: string;
    file: File;
    preview?: string;
    tags: string[];
    progress: number;
    status: 'pending' | 'uploading' | 'success' | 'error';
    error?: string;
    hash?: string;
  }

  interface FileUploadSectionProps {
    reportId?: string;
    acceptedTypes?: string[];
    maxFileSize?: number;
    maxFiles?: number;
    multiple?: boolean;
    onupload?: (data: { files: File[]; tags: string[] }) => void;
    onfilesChanged?: (files: FileUpload[]) => void;
    onerror?: (error: string) => void;
  }

  import { $props } from 'svelte';

  let {
    reportId = '',
    acceptedTypes = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.txt', '.doc', '.docx'],
    maxFileSize = 10 * 1024 * 1024,
    maxFiles = 5,
    multiple = true,
    onupload,
    onfilesChanged,
    onerror
  }: FileUploadSectionProps = $props();

  import { browser } from "$app/environment";
  import {
    AlertCircle,
    CheckCircle,
    CloudUpload,
    File as FileIcon,
    FileText,
    Image,
    Upload,
    X
  } from 'lucide-svelte';
  import { onMount } from 'svelte';
  import { storageService, processDocumentWorkflow, documentWorkflows } from '$lib/services/minio-neo4j-pgvector-integration';
  import { ComprehensiveAISystemIntegration } from '$lib/integration/comprehensive-ai-system-integration';
  import { mcpContext72GetLibraryDocs } from '$lib/mcp-context72-get-library-docs';
  // loki is optional/hybrid local DB - guard access at runtime
  import loki from '$lib/services/loki-client';
  import TagList from './TagList.svelte';


  let uploadFiles: UploadFile[] = $state([]);
let aiSystem = $state<ComprehensiveAISystemIntegration;
  let docStatus: string >('');
let docs = $state<any >(null);
let availableTags = $state<string[] >([]);
let summaryType = $state<'key_points' | 'narrative' | 'prosecutorial' >('narrative');

  // Load available tags
  onMount(async () => {
    if (browser) {
      loadAvailableTags();
      aiSystem = new ComprehensiveAISystemIntegration();
      await aiSystem.initializeComponents();
      // Optionally fetch Svelte docs for context/help
      try {
        docs = await mcpContext72GetLibraryDocs('/sveltejs/svelte', 'file-upload|runes');
      } catch (e) {
        docs = null;
      }
    }
  });

	async function loadAvailableTags() {
		try {
			const evidence = loki.evidence.getAll();
			const allTags = evidence.flatMap((e: any) => e.tags || []);
			availableTags = [...new Set(allTags as string[])].sort();
		} catch (error) {
			console.error('Failed to load available tags:', error);
		}
	}
	function getFileIcon(file: globalThis.File) {
		const type = file.type.toLowerCase();
		const name = file.name.toLowerCase();

		if (type.startsWith('image/')) {
			return Image;
		} else if (type === 'application/pdf' || name.endsWith('.pdf')) {
			return FileText; // Using FileText for PDF since lucide doesn't have a specific PDF icon
		} else if (type.startsWith('text/') || name.endsWith('.txt') || name.endsWith('.doc') || name.endsWith('.docx')) {
			return FileText;
		}
		return FileIcon;
	}
	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
	function isFileValid(file: globalThis.File): { valid: boolean; error?: string } {
		// Check file size
		if (file.size > maxFileSize) {
			return {
				valid: false,
				error: `File size exceeds ${formatFileSize(maxFileSize)} limit`
			};
		}
		// Check file type
		const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
		if (!acceptedTypes.some(type => type.toLowerCase() === fileExtension)) {
			return {
				valid: false,
				error: `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`
			};
		}
		return { valid: true };
	}
	async function createFilePreview(file: globalThis.File): Promise<string | undefined> {
		if (!file.type.startsWith('image/')) return undefined;

		return new Promise((resolve) => {
			const reader = new FileReader();
			reader.onload = (e) => resolve(e.target?.result as string);
			reader.onerror = () => resolve(undefined);
			reader.readAsDataURL(file);
		});
	}
  // Handle files change from modular FileUpload component
  function handleFilesChange(files: UploadFile[]) {
    uploadFiles = files;
    
    // Convert to legacy FileUpload format for compatibility
    const legacyUploads = files.map(file => ({
      id: file.id,
      file: file.file,
      preview: file.preview,
      tags: [], // Initialize empty tags
      progress: file.progress || 0,
      status: file.status === 'pending' ? 'pending' as const :
              file.status === 'uploading' ? 'uploading' as const :
              file.status === 'completed' ? 'success' as const :
              'error' as const,
      error: file.error,
      hash: undefined
    }));
    
    onfilesChanged?.(legacyUploads);
  }

  // Handle file upload progress and processing
  async function handleFileUpload(file: UploadFile): Promise<void> {
    try {
      file.status = 'uploading';
      file.progress = 0;
      uploadFiles = [...uploadFiles];

      // Prepare DocumentWorkflow
      const workflow = {
        documentId: file.id,
        caseId: reportId,
        userId: 'current-user', // TODO: get from auth context
        filename: file.name,
        content: await file.file.arrayBuffer(),
        metadata: {
          filename: file.name,
          mimeType: file.type,
          fileSize: file.size,
          uploadDate: new Date(),
          caseId: reportId,
          userId: 'current-user',
          tags: [], // Will be updated later
          classification: 'evidence',
          confidentialityLevel: 'internal',
          retentionPolicy: 'standard',
          customFields: {}
        },
        stages: [],
        currentStage: 0,
        startTime: new Date(),
        status: 'pending'
      };

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        if (file.progress! < 90) {
          file.progress = (file.progress || 0) + 10;
          uploadFiles = [...uploadFiles];
        }
      }, 100);

      // Process workflow (MinIO, Neo4j, pgvector)
      await processDocumentWorkflow(workflow);
      
      // Trigger AI/ML analysis
      if (aiSystem) {
        await aiSystem.processDocument(file.id, file.name, { tags: [] });
      }

      clearInterval(progressInterval);
      file.status = 'completed';
      file.progress = 100;
      uploadFiles = [...uploadFiles];
      docStatus = 'Upload and analysis complete.';

    } catch (error) {
      file.status = 'error';
      file.error = error instanceof Error ? error.message : 'Upload failed';
      uploadFiles = [...uploadFiles];
      docStatus = 'Error: ' + file.error;
    }
  }

  // Handle file removal
  function handleFileRemove(fileId: string) {
    uploadFiles = uploadFiles.filter(f => f.id !== fileId);
    
    // Also notify parent of successful uploads for final callback
    const successfulFiles = uploadFiles
      .filter(f => f.status === 'completed')
      .map(f => f.file);
    
    if (successfulFiles.length > 0) {
      onupload?.({ files: successfulFiles, tags: [] });
    }
  }
</script>

<Card variant="evidence" class="space-y-6">
  {#snippet header()}
    <div class="space-y-2">
      <h3 class="text-lg font-semibold flex items-center gap-2">
        <CloudUpload class="w-5 h-5" />
        Evidence File Upload
      </h3>
      <p class="text-sm text-muted-foreground">
        Upload documents, images, videos, or other evidence files for comprehensive AI analysis
      </p>
    </div>
  {/snippet}

  <!-- Modular File Upload Component -->
  <FileUpload
    variant="evidence"
    {multiple}
    maxFiles={maxFiles}
    maxSize={maxFileSize}
    accept={acceptedTypes.join(',')}
    bind:files={uploadFiles}
    fileschange={handleFilesChange}
    upload={handleFileUpload}
    remove={handleFileRemove}
    dragDropText="Drop evidence files here or click to browse"
    browseText="Browse Evidence Files"
    supportedFormats={acceptedTypes.map(type => type.replace('.', '').toUpperCase())}
  />

  <!-- Analysis Controls -->
  {#if uploadFiles.length > 0}
    <div class="space-y-4">
      <!-- Summary Type Selection -->
      <div class="space-y-2">
        <label class="text-sm font-medium">Analysis Type</label>
        <select 
          bind:value={summaryType} 
          class="w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-sm focus:ring-2 focus:ring-orange-500 focus:border-orange-500"
          aria-label="Select analysis type"
        >
          <option value="key_points">Key Points Analysis</option>
          <option value="narrative">Narrative Summary</option>
          <option value="prosecutorial">Prosecutorial Analysis</option>
        </select>
      </div>

      <!-- Processing Status -->
      {#if docStatus}
        <div class="p-3 rounded-md" class:bg-green-50={docStatus.includes('complete')} class:bg-red-50={docStatus.includes('Error')} class:bg-blue-50={!docStatus.includes('complete') && !docStatus.includes('Error')}>
          <Badge variant={docStatus.includes('complete') ? 'success' : docStatus.includes('Error') ? 'destructive' : 'info'}>
            {docStatus}
          </Badge>
        </div>
      {/if}

      <!-- Context7.2 Documentation (Optional) -->
      {#if docs}
        <details class="mt-6">
          <summary class="text-sm font-medium cursor-pointer hover:text-orange-600 transition-colors">
            📚 Show Svelte 5 File Upload Documentation (Context7.2)
          </summary>
          <div class="mt-2 p-4 bg-gray-50 rounded-md text-xs font-mono overflow-auto max-h-64">
            <pre>{docs.content}</pre>
          </div>
        </details>
      {/if}
    </div>
  {/if}
</Card>

<!-- Styles are now handled by modular components and UnoCSS -->

