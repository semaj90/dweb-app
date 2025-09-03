<!-- N64 Gaming-Style MinIO Upload with Retro Progress Bars -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Upload, FileText, Image, CheckCircle, AlertCircle, Loader2, Zap } from 'lucide-svelte';
  import { toastService } from '$lib/services/toast-service';
  import { gpuService } from '$lib/services/gpu-acceleration-service';
  import { vectorService } from '$lib/services/postgresql-vector-service';
  import { embeddingService } from '$lib/services/embedding-service';
  import { telemetry } from '$lib/services/telemetry-service';
  import { uploadTelemetry } from '$lib/services/upload-telemetry-service';
  
  // Import our N64 gaming components
  import N64ProgressBar from '$lib/components/ui/gaming/n64/N64ProgressBar.svelte';
  import N64LoadingRing from '$lib/components/ui/gaming/n64/N64LoadingRing.svelte';
  import N64EvolutionLoader from '$lib/components/ui/gaming/n64/N64EvolutionLoader.svelte';

  interface Props {
    caseId?: string;
    onUploadComplete?: (result: UploadResult) => void;
    onUploadError?: (error: string) => void;
    multiple?: boolean;
    disabled?: boolean;
    accept?: string;
    maxSize?: number;
    maxConcurrency?: number;
    enableGPUProcessing?: boolean;
    enableToastNotifications?: boolean;
    maxRetries?: number;
    gamingTheme?: 'nes' | 'snes' | 'n64' | 'modern';
    retro?: boolean;
    animateEvolution?: boolean;
  }

  let {
    caseId = '',
    onUploadComplete,
    onUploadError,
    multiple = false,
    disabled = false,
    accept = '.pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.tiff',
    maxSize = 100 * 1024 * 1024,
    maxConcurrency = 3,
    enableGPUProcessing = true,
    enableToastNotifications = true,
    maxRetries = 3,
    gamingTheme = 'n64',
    retro = true,
    animateEvolution = true
  }: Props = $props();

  interface UploadResult {
    success: boolean;
    id: string;
    fileName: string;
    originalName: string;
    fileSize: number;
    url: string;
    hash: string;
    message: string;
  }

  interface FileState {
    file: File;
    status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error' | 'canceled';
    progress: number;
    controller?: AbortController | null;
    result?: UploadResult;
    error?: string;
    toastId?: string;
    gpuTaskIds?: string[];
    startTime?: Date;
    endTime?: Date;
    attempts?: number;
    nextRetryAt?: number;
    retryTimeoutId?: ReturnType<typeof setTimeout> | null;
    placeholder?: boolean;
    originalSize?: number;
    gamingProgress?: {
      theme: string;
      sparkle: boolean;
      animated: boolean;
    };
  }

  let files = $state<File[]>([]);
  let fileStates = $state<FileState[]>([]);
  let uploading = $state(false);
  let dragOver = $state(false);
  let uploadProgress = $state(0);
  let uploadStatus = $state<'idle' | 'uploading' | 'processing' | 'completed' | 'error' | 'canceled'>('idle');
  let errorMessage = $state<string | null>(null);
  let minioHealthy = $state<boolean | null>(null);
  let fileInput: HTMLInputElement | null = null;
  let liveMessage = $state('');

  let activeUploads = $state(0);
  let uploadQueue: FileState[] = [];
  let batchToastId = $state<string | null>(null);
  let performanceMetrics = $state({
    totalFiles: 0,
    completedFiles: 0,
    averageUploadTime: 0,
    totalUploadTime: 0,
    gpuTasksSubmitted: 0
  });

  // Gaming-specific state
  let evolutionStage = $state<'nes' | 'snes' | 'n64' | 'modern'>(gamingTheme);
  let showEvolutionLoader = $state(false);

  // N64 controller color themes
  const n64Themes = {
    nes: { theme: 'red', sparkle: false },
    snes: { theme: 'blue', sparkle: true },
    n64: { theme: 'gold', sparkle: true },
    modern: { theme: 'green', sparkle: true }
  };

  // Session persistence
  const STORAGE_KEY = 'n64-minio-upload-session';
  const enablePersistence = true;

  function serializeSession() {
    if (!enablePersistence) return;
    const pending = fileStates.filter(f => !['completed','canceled'].includes(f.status)).map(f => ({
      name: f.file.name,
      size: f.file.size,
      type: f.file.type,
      status: f.status === 'uploading' || f.status === 'processing' ? 'pending' : f.status,
      attempts: f.attempts || 0,
      nextRetryAt: f.nextRetryAt && f.nextRetryAt > Date.now() ? f.nextRetryAt : null,
      gamingProgress: f.gamingProgress
    }));
    if (pending.length === 0) { 
      try { sessionStorage.removeItem(STORAGE_KEY); } catch {}; 
      return; 
    }
    try { 
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ 
        ts: Date.now(), 
        files: pending,
        evolutionStage 
      })); 
    } catch {}
  }

  function restoreSession() {
    if (!enablePersistence) return;
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (!data?.files) return;
      
      // Restore evolution stage
      if (data.evolutionStage) {
        evolutionStage = data.evolutionStage;
      }
      
      const restored: FileState[] = [];
      for (const m of data.files) {
        const ph = new File([], m.name, { type: m.type || 'application/octet-stream' });
        restored.push({
          file: ph,
          placeholder: true,
          originalSize: m.size,
          status: 'pending',
          progress: 0,
          attempts: m.attempts || 0,
          nextRetryAt: m.nextRetryAt || null,
          gamingProgress: m.gamingProgress || n64Themes[evolutionStage]
        });
      }
      
      if (restored.length) {
        fileStates = [...fileStates, ...restored];
        files = [...files, ...restored.map(r => r.file)];
        liveMessage = `Restored ${restored.length} pending file(s)`;
        if (enableToastNotifications) {
          toastService.info('N64 Session Restored', `Recovered ${restored.length} pending file(s). Re-select originals to resume.`, { duration: 6000 });
        }
        ensureRetryTicker();
      }
    } catch {}
  }

  function matchPlaceholders(incoming: File[]) {
    for (const f of incoming) {
      const idx = fileStates.findIndex(ps => ps.placeholder && ps.file.name === f.name && ps.originalSize === f.size);
      if (idx !== -1) {
        const prev = fileStates[idx];
        fileStates[idx] = { ...prev, file: f, placeholder: false };
      }
    }
  }

  function isRetryable(message: string, statusCode?: number): boolean {
    const transientPatterns = [/network/i, /timeout/i, /temporar/i, /rate limit/i, /ECONNRESET/i];
    if (statusCode && (statusCode >= 500 || statusCode === 429)) return true;
    return transientPatterns.some(r => r.test(message));
  }

  function scheduleRetry(fs: FileState, reason: string) {
    fs.attempts = (fs.attempts || 1);
    if (fs.attempts >= maxRetries) {
      fs.status = 'error';
      fs.error = reason;
      telemetry.emit('upload_retry_exhausted', { file: fs.file.name, attempts: fs.attempts, reason });
      
      if (enableToastNotifications && fs.toastId) {
        toastService.failUpload(
          fs.toastId,
          `${reason} (max retries reached)`,
          () => retryFileUpload(fs)
        );
      }
      return;
    }
    
    const delay = Math.min(8000, 600 * Math.pow(2, (fs.attempts - 1))) + Math.floor(Math.random() * 300);
    fs.status = 'pending';
    fs.nextRetryAt = Date.now() + delay;
    if (fs.retryTimeoutId) { 
      clearTimeout(fs.retryTimeoutId); 
      fs.retryTimeoutId = null; 
    }
    
    if (enableToastNotifications) {
      const eta = (delay/1000).toFixed(1);
      if (fs.toastId) {
        toastService.update(fs.toastId, { 
          type: 'info', 
          message: `🎮 Retrying in ${eta}s (attempt ${fs.attempts + 1}/${maxRetries})` 
        });
      } else {
        fs.toastId = toastService.upload(
          `🎮 ${fs.file.name}`, 
          `Retrying in ${eta}s (attempt ${fs.attempts + 1}/${maxRetries})`, 
          { dismissible: false }
        );
      }
    }
    
    fs.retryTimeoutId = setTimeout(() => {
      if (fs.status === 'pending' && uploading) {
        fs.retryTimeoutId = null;
        fs.attempts = (fs.attempts || 0) + 1;
        uploadQueue.push(fs);
        processUploadQueue();
      }
    }, delay);
    
    ensureRetryTicker();
    telemetry.emit('upload_retry_scheduled', { 
      file: fs.file.name, 
      attemptNext: (fs.attempts + 1), 
      maxRetries, 
      delayMs: delay 
    });
  }

  let retryTicker = $state(0);
  let retryInterval: any = null;
  
  function ensureRetryTicker() {
    if (retryInterval) return;
    retryInterval = setInterval(() => {
      const pendingRetries = fileStates.some(f => 
        f.status === 'pending' && f.nextRetryAt && f.nextRetryAt > Date.now()
      );
      if (!pendingRetries) {
        clearInterval(retryInterval);
        retryInterval = null;
        return;
      }
      retryTicker = retryTicker + 1;
    }, 1000);
  }

  onDestroy(() => {
    if (retryInterval) clearInterval(retryInterval);
  });

  function cancelAllUploads() {
    fileStates = fileStates.map(fs => {
      if (fs.controller) {
        try { fs.controller.abort(); } catch {}
      }
      if (fs.retryTimeoutId) {
        try { clearTimeout(fs.retryTimeoutId); } catch {};
        fs.retryTimeoutId = null;
      }
      if (['uploading','pending','processing'].includes(fs.status)) {
        return { 
          ...fs, 
          status: 'canceled', 
          progress: fs.status === 'uploading' ? fs.progress : 0, 
          controller: null 
        };
      }
      return fs;
    });
    
    uploading = false;
    liveMessage = 'All uploads canceled';
    
    if (enableToastNotifications) {
      toastService.info('🎮 Uploads Canceled', 'All in‑flight and queued uploads have been canceled.', { duration: 4000 });
    }
    
    finalizeAggregateStatus();
    serializeSession();
    telemetry.emit('upload_batch_canceled_all', { remaining: fileStates.length });
  }

  // Drag and drop handlers
  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    dragOver = true;
  }

  function handleDragLeave(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    dragOver = false;
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    if (disabled || uploading) return;
    dragOver = false;
    const droppedFiles = Array.from(event.dataTransfer?.files || []);
    processFiles(droppedFiles);
  }

  function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const selectedFiles = Array.from(target.files || []);
    processFiles(selectedFiles);
  }

  function processFiles(newFiles: File[]) {
    errorMessage = null;
    const validFiles: File[] = [];
    
    for (const file of newFiles) {
      if (file.size > maxSize) {
        errorMessage = `File ${file.name} exceeds ${formatFileSize(maxSize)} limit`;
        continue;
      }
      validFiles.push(file);
    }

    matchPlaceholders(validFiles);
    
    if (multiple) {
      files = [...files, ...validFiles];
    } else {
      files = validFiles.slice(0, 1);
    }

    const existingNames = new Set(fileStates.map(f => f.file));
    for (const f of validFiles) {
      if (![...fileStates].some(fs => fs.file === f)) {
        fileStates = [...fileStates, {
          file: f,
          status: 'pending',
          progress: 0,
          gamingProgress: n64Themes[evolutionStage]
        }];
      }
    }
    
    fileStates = fileStates.filter(fs => files.includes(fs.file));
    serializeSession();
  }

  function removeFile(index: number) {
    if (uploading) return;
    const target = fileStates[index];
    if (target && target.status === 'uploading') return;
    if (target?.retryTimeoutId) {
      try { clearTimeout(target.retryTimeoutId); } catch {};
    }
    files = files.filter((_, i) => i !== index);
    fileStates = fileStates.filter((_, i) => i !== index);
    serializeSession();
  }

  function cancelUpload(index: number) {
    const fs = fileStates[index];
    if (!fs || fs.status !== 'uploading') return;
    try { fs.controller?.abort(); } catch {}
    if (fs.retryTimeoutId) {
      try { clearTimeout(fs.retryTimeoutId); } catch {};
      fs.retryTimeoutId = null;
    }
    fs.status = 'canceled';
    fs.progress = 0;
    liveMessage = `Upload canceled for ${fs.file.name}`;
    uploading = fileStates.some(f => f.status === 'uploading');
    if (!uploading) finalizeAggregateStatus();
    serializeSession();
    telemetry.emit('upload_canceled', { file: fs.file.name });
  }

  function retryFile(index: number) {
    const fs = fileStates[index];
    if (!fs || (fs.status !== 'error' && fs.status !== 'canceled')) return;
    fs.status = 'pending';
    fs.progress = 0;
    fs.error = undefined;
    liveMessage = `Retry scheduled for ${fs.file.name}`;
    serializeSession();
    telemetry.emit('upload_manual_retry', { file: fs.file.name });
  }

  function aggregateProgress() {
    if (fileStates.length === 0) return 0;
    return Math.round(fileStates.reduce((sum, f) => sum + f.progress, 0) / fileStates.length);
  }

  async function finalizeAggregateStatus() {
    const anyError = fileStates.some(f => f.status === 'error');
    const allDone = fileStates.every(f => ['completed','canceled'].includes(f.status));
    uploadProgress = aggregateProgress();

    if (anyError) {
      uploadStatus = 'error';
    } else if (allDone) {
      uploadStatus = 'completed';
      
      // Evolution stage progression on completion
      if (animateEvolution && allDone && !anyError) {
        const stages: typeof evolutionStage[] = ['nes', 'snes', 'n64', 'modern'];
        const currentIndex = stages.indexOf(evolutionStage);
        if (currentIndex < stages.length - 1) {
          evolutionStage = stages[currentIndex + 1];
          showEvolutionLoader = true;
          setTimeout(() => { showEvolutionLoader = false; }, 3000);
        }
      }
    }

    if (enableToastNotifications && batchToastId) {
      const completed = fileStates.filter(f => f.status === 'completed').length;
      const failed = fileStates.filter(f => f.status === 'error').length;
      const canceled = fileStates.filter(f => f.status === 'canceled').length;

      if (uploadStatus === 'completed' && failed === 0) {
        toastService.completeUpload(
          batchToastId,
          `🎮 All ${completed} files uploaded successfully! Average time: ${Math.round(performanceMetrics.averageUploadTime)}ms`
        );
      } else {
        toastService.update(batchToastId, {
          type: anyError ? 'warning' : 'success',
          message: `🎮 Batch complete: ${completed} success, ${failed} failed, ${canceled} canceled`
        });
        setTimeout(() => toastService.dismiss(batchToastId!), 5000);
      }
      batchToastId = null;
    }

    if (enableToastNotifications && performanceMetrics.completedFiles > 0) {
      toastService.info(
        `🎮 N64 Upload Performance`,
        `Completed ${performanceMetrics.completedFiles} files in ${Math.round(performanceMetrics.totalUploadTime / 1000)}s. GPU tasks: ${performanceMetrics.gpuTasksSubmitted}`,
        { duration: 6000 }
      );
    }

    if (uploadStatus === 'completed' && !anyError) {
      liveMessage = 'All uploads completed';
      setTimeout(() => {
        files = [];
        fileStates = [];
        uploadProgress = 0;
        uploadStatus = 'idle';
        if (fileInput) fileInput.value = '';
        performanceMetrics = {
          totalFiles: 0,
          completedFiles: 0,
          averageUploadTime: 0,
          totalUploadTime: 0,
          gpuTasksSubmitted: 0
        };
      }, 2500);
    }
  }

  function retryFileUpload(fs: FileState) {
    fs.status = 'pending';
    fs.progress = 0;
    fs.error = undefined;
    fs.toastId = undefined;
    fs.startTime = undefined;
    fs.endTime = undefined;
    fs.attempts = 1;

    if (!uploading) {
      uploadFiles();
    } else {
      uploadQueue.push(fs);
    }
  }

  async function uploadFiles() {
    if (fileStates.length === 0 || uploading) return;
    const realPending = fileStates.filter(f => f.status === 'pending' && !f.placeholder);
    if (realPending.length === 0) {
      if (enableToastNotifications) {
        toastService.info('🎮 Awaiting Files', 'Select original files to replace placeholders before uploading.', { duration: 5000 });
      }
      return;
    }

    errorMessage = null;
    uploadStatus = 'uploading';
    liveMessage = 'Starting N64-style parallel upload batch';
    uploading = true;
    activeUploads = 0;

    performanceMetrics.totalFiles = fileStates.filter(fs => fs.status === 'pending' && !fs.placeholder).length;
    telemetry.emit('upload_batch_start', { total: performanceMetrics.totalFiles, concurrency: maxConcurrency });
    performanceMetrics.completedFiles = 0;
    performanceMetrics.totalUploadTime = 0;
    performanceMetrics.gpuTasksSubmitted = 0;

    if (enableToastNotifications) {
      batchToastId = toastService.upload(
        `🎮 N64 Upload: ${performanceMetrics.totalFiles} files`,
        `Starting parallel upload with ${maxConcurrency} concurrent connections...`,
        { dismissible: false }
      );
    }

    uploadQueue = fileStates.filter(fs => fs.status === 'pending' && !fs.placeholder);

    const uploadPromises: Promise<void>[] = [];
    for (let i = 0; i < Math.min(maxConcurrency, uploadQueue.length); i++) {
      uploadPromises.push(processUploadQueue());
    }

    await Promise.all(uploadPromises);
    uploading = fileStates.some(f => f.status === 'uploading');
    if (!uploading) {
      await finalizeAggregateStatus();
    }
    serializeSession();
    telemetry.emit('upload_batch_end', {
      completed: fileStates.filter(f=>f.status==='completed').length,
      failed: fileStates.filter(f=>f.status==='error').length,
      canceled: fileStates.filter(f=>f.status==='canceled').length
    });
  }

  async function processUploadQueue(): Promise<void> {
    while (uploadQueue.length > 0 && uploading) {
      const fs = uploadQueue.shift();
      if (!fs) break;

      activeUploads++;
      await uploadSingleFile(fs);
      activeUploads--;

      uploadProgress = aggregateProgress();

      if (enableToastNotifications && batchToastId) {
        const completed = fileStates.filter(f => f.status === 'completed').length;
        const total = performanceMetrics.totalFiles;
        toastService.updateUploadProgress(
          batchToastId,
          (completed / total) * 100,
          `🎮 ${completed}/${total} files uploaded (${activeUploads} active)`
        );
      }
    }
  }

  async function uploadSingleFile(fs: FileState) {
    const file = fs.file;
    fs.status = 'uploading';
    fs.progress = 0;
    fs.error = undefined;
    fs.startTime = new Date();
    fs.attempts = (fs.attempts || 0) + 1;
    
    const controller = new AbortController();
    fs.controller = controller;
    liveMessage = `🎮 Uploading ${file.name}`;
    telemetry.emit('upload_start', { 
      file: file.name, 
      size: file.size, 
      type: file.type, 
      attempt: fs.attempts 
    });

    // Update gaming progress theme based on file type
    if (file.type.startsWith('image/')) {
      fs.gamingProgress = { theme: 'blue', sparkle: true, animated: true };
    } else if (file.type === 'application/pdf') {
      fs.gamingProgress = { theme: 'gold', sparkle: true, animated: true };
    } else {
      fs.gamingProgress = { theme: 'green', sparkle: false, animated: true };
    }

    if (enableToastNotifications) {
      fs.toastId = toastService.upload(
        `🎮 ${file.name}`,
        'Starting N64-style upload...',
        {
          dismissible: false,
          actions: [{
            label: 'Cancel',
            action: () => {
              controller.abort();
              fs.status = 'canceled';
            },
            style: 'danger'
          }]
        }
      );
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('uploadData', JSON.stringify({
      caseId,
      title: file.name,
      description: `N64-style upload: ${file.name}`,
      evidenceType: getEvidenceType(file),
      enableAiAnalysis: true,
      enableEmbeddings: true,
      enableOcr: file.type.startsWith('image/') || file.type === 'application/pdf'
    }));

    try {
      const xhr: XMLHttpRequest = new XMLHttpRequest();
      xhr.open('POST', '/api/evidence/upload');
      
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          fs.progress = Math.min(90, Math.round((e.loaded / e.total) * 90));
        }
      };
      
      const abortHandler = () => xhr.abort();
      controller.signal.addEventListener('abort', abortHandler);
      
      const resultPromise = new Promise<UploadResult[]>((resolve, reject) => {
        xhr.onreadystatechange = () => {
          if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
              try {
                const json = JSON.parse(xhr.responseText);
                resolve(json.data || []);
              } catch (e) {
                reject(e);
              }
            } else {
              reject(Object.assign(new Error(xhr.responseText || 'Upload failed'), { statusCode: xhr.status }));
            }
          }
        };
        xhr.onerror = () => reject(new Error('Network error'));
        xhr.onabort = () => reject(new Error('Upload aborted'));
      });
      
      xhr.send(formData);
      const data = await resultPromise;
      
      fs.progress = 90;
      fs.status = 'processing';
      liveMessage = `🎮 Processing ${file.name}`;

      if (enableToastNotifications && fs.toastId) {
        toastService.updateUploadProgress(fs.toastId, 90, '🎮 Upload complete, processing...');
      }

      if (data[0]) {
        fs.result = data[0];
        fs.endTime = new Date();

        // Submit GPU processing tasks
        if (enableGPUProcessing && data[0].id) {
          try {
            const gpuTasks = await gpuService.processFileWithGPU(
              data[0].id,
              await file.arrayBuffer(),
              {
                enableOCR: file.type.startsWith('image/') || file.type === 'application/pdf',
                enableEmbedding: true,
                enableAnalysis: true
              }
            );
            fs.gpuTaskIds = gpuTasks.taskIds;
            performanceMetrics.gpuTasksSubmitted += gpuTasks.taskIds.length;

            if (enableToastNotifications) {
              toastService.gpuTask(
                'N64 Processing',
                'queued',
                `${gpuTasks.taskIds.length} GPU tasks queued for ${file.name}`
              );
            }
          } catch (error) {
            console.warn('GPU processing failed:', error);
          }
        }

        // Generate embeddings
        try {
          const textContent = `Content from ${file.name}`;
          telemetry.emit('embedding_start', { file: file.name });
          let embeddingVector: number[] = [];
          let embeddingDims = 0;
          let embeddingModel = '';
          
          try {
            const embedding = await embeddingService.generateEmbedding(textContent, { preferRagService: false });
            embeddingVector = embedding.vector;
            embeddingDims = embedding.dimensions;
            embeddingModel = embedding.model;
            telemetry.emit('embedding_complete', { 
              file: file.name, 
              model: embedding.model, 
              dims: embedding.dimensions, 
              latencyMs: embedding.latencyMs, 
              source: embedding.source 
            });
          } catch (e) {
            embeddingVector = Array.from({ length: 384 }, () => Math.random() - 0.5);
            embeddingDims = 384;
            embeddingModel = 'fallback-random-384';
            telemetry.emit('embedding_error', { 
              file: file.name, 
              error: e instanceof Error ? e.message : 'unknown' 
            });
            console.warn('Embedding generation failed, using fallback vector:', e);
          }

          try {
            await vectorService.updateFileMapping(data[0].id, {
              textChunks: [textContent],
              embeddings: [embeddingVector],
              ocrText: file.type.startsWith('image/') ? 'OCR extracted text' : undefined,
              analysisResults: { fileType: file.type, size: file.size, embeddingDims, embeddingModel }
            });
          } catch (error) {
            console.warn('Vector storage failed:', error);
          }
        } catch (outerEmbeddingErr) {
          console.warn('Embedding/vector pipeline error:', outerEmbeddingErr);
        }

        // Publish Redis event
        fetch('/api/v1/redis/publish', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            channel: 'evidence_update',
            data: {
              type: 'EVIDENCE_UPLOADED',
              evidenceId: data[0].id,
              caseId,
              fileName: file.name,
              timestamp: new Date().toISOString(),
              gpuTaskIds: fs.gpuTaskIds
            }
          })
        }).catch(() => {});

        fs.progress = 100;
        fs.status = 'completed';
        performanceMetrics.completedFiles++;

        if (fs.startTime && fs.endTime) {
          const uploadTime = fs.endTime.getTime() - fs.startTime.getTime();
          performanceMetrics.totalUploadTime += uploadTime;
          performanceMetrics.averageUploadTime = performanceMetrics.totalUploadTime / performanceMetrics.completedFiles;
        }

        onUploadComplete?.(data[0]);
        liveMessage = `🎮 Upload completed for ${file.name}`;

        if (enableToastNotifications && fs.toastId) {
          toastService.completeUpload(
            fs.toastId,
            `🎮 ${file.name} uploaded successfully! ${fs.gpuTaskIds?.length || 0} GPU tasks queued.`
          );
        }
        
        serializeSession();
        telemetry.emit('upload_complete', { 
          file: file.name, 
          size: file.size, 
          durationMs: fs.endTime && fs.startTime ? (fs.endTime.getTime()-fs.startTime.getTime()) : null, 
          gpuTasks: fs.gpuTaskIds?.length || 0 
        });
      } else {
        const msg = 'No response data';
        if (isRetryable(msg)) {
          scheduleRetry(fs, msg);
        } else {
          fs.status = 'error';
          fs.error = msg;
          errorMessage = msg;
          onUploadError?.(msg);
          liveMessage = `🎮 Upload failed for ${file.name}`;
          if (enableToastNotifications && fs.toastId) {
            toastService.failUpload(fs.toastId, msg, () => retryFileUpload(fs));
          }
        }
      }
    } catch (err) {
      if ((fs.status as FileState['status']) === 'canceled') return;

      fs.endTime = new Date();
      fs.status = controller.signal.aborted ? 'canceled' : 'error';
      fs.error = err instanceof Error ? err.message : 'Upload failed';
      errorMessage = fs.error;
      onUploadError?.(fs.error);
      liveMessage = `🎮 ${fs.status === 'canceled' ? 'Canceled' : 'Failed'} upload for ${file.name}`;

      if (enableToastNotifications && fs.toastId) {
        if (fs.status === 'canceled') {
          toastService.update(fs.toastId, {
            type: 'warning',
            message: '🎮 Upload canceled by user'
          });
          setTimeout(() => toastService.dismiss(fs.toastId!), 3000);
        } else {
          if (isRetryable(fs.error || '')) {
            scheduleRetry(fs, fs.error || 'Retrying');
          } else {
            toastService.failUpload(
              fs.toastId,
              fs.error || 'Upload failed',
              () => retryFileUpload(fs)
            );
          }
        }
      }
      
      telemetry.emit(fs.status === 'canceled' ? 'upload_canceled' : 'upload_error', { 
        file: file.name, 
        error: fs.error, 
        attempt: fs.attempts 
      });
    } finally {
      fs.controller = null;
      serializeSession();
    }
  }

  function getEvidenceType(file: File): string {
    if (file.type.startsWith('image/')) return 'IMAGE';
    if (file.type === 'application/pdf') return 'PDF';
    if (file.type.startsWith('text/')) return 'TEXT';
    if (file.type.startsWith('video/')) return 'VIDEO';
    if (file.type.startsWith('audio/')) return 'AUDIO';
    return 'DOCUMENT';
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function openFileDialog() {
    if (!disabled && !uploading && fileInput) fileInput.click();
  }

  onMount(async () => {
    restoreSession();
    try {
      const res = await fetch('/api/v1/minio/health');
      if (res.ok) {
        const data = await res.json();
        minioHealthy = !!data?.ok;
      } else minioHealthy = false;
    } catch { minioHealthy = false; }
  });

  $effect(() => {
    fileStates.map(f => [f.status, f.progress, f.attempts, f.nextRetryAt, f.placeholder]);
    uploading;
    queueMicrotask(serializeSession);
  });
</script>

<!-- N64 Gaming Style MinIO Upload Zone -->
<div class="n64-upload-container" class:retro>
  <!-- Hidden file input -->
  <input 
    bind:this={fileInput} 
    type="file" 
    {accept} 
    {multiple} 
    disabled={disabled || uploading} 
    onchange={handleFileSelect} 
    style="display:none" 
  />

  <!-- Evolution Loader Overlay -->
  {#if showEvolutionLoader}
    <div class="evolution-overlay">
      <N64EvolutionLoader 
        stage={evolutionStage}
        autoEvolution={false}
        ragIntegration={enableGPUProcessing}
        yorhaMode={retro}
      />
    </div>
  {/if}

  <!-- N64-style drop zone -->
  <div
    class="n64-drop-zone"
    class:drag-over={dragOver}
    class:has-files={files.length > 0}
    class:uploading={uploading}
    class:theme-{evolutionStage}
    role="button"
    aria-disabled={disabled || uploading}
    tabindex="0"
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    onclick={openFileDialog}
    onkeydown={(e) => e.key === 'Enter' && openFileDialog()}
  >
    {#if fileStates.length === 0}
      <div class="n64-upload-prompt">
        <div class="n64-upload-icon">
          <N64LoadingRing 
            size="lg"
            theme={n64Themes[evolutionStage].theme}
            speed="medium"
            showPercentage={false}
          />
        </div>
        <div class="n64-upload-text">
          <h3>
            {dragOver ? '🎮 DROP FILES HERE!' : `🎮 N64 UPLOAD ZONE [${evolutionStage.toUpperCase()}]`}
          </h3>
          <p class="n64-subtext">
            Supports PDF, Word, Text, and Image files up to {formatFileSize(maxSize)}
          </p>
        </div>
      </div>
    {:else}
      <!-- N64-style file list -->
      <div class="n64-file-list">
        {#each fileStates as fs, index}
          <div class="n64-file-item" class:status-{fs.status} aria-live="polite">
            <div class="file-icon n64-icon">
              {#if fs.file.type.startsWith('image/')}
                <Image class="w-6 h-6" />
              {:else}
                <FileText class="w-6 h-6" />
              {/if}
            </div>
            
            <div class="file-info">
              <div class="n64-file-name">{fs.file.name}</div>
              <div class="n64-file-size">{formatFileSize(fs.file.size)}</div>
              <div class="n64-file-status">
                {#if fs.status === 'pending'}
                  🎯 PENDING {#if fs.attempts && fs.attempts > 1}• RETRY {fs.attempts - 1}{/if}
                {/if}
                {#if fs.status === 'uploading'}
                  🚀 UPLOADING {fs.progress}% (ATTEMPT {fs.attempts})
                {/if}
                {#if fs.status === 'processing'}
                  ⚡ PROCESSING...
                {/if}
                {#if fs.status === 'completed'}
                  ✅ COMPLETED (ATTEMPTS {fs.attempts})
                {/if}
                {#if fs.status === 'error'}
                  ❌ FAILED (ATTEMPTS {fs.attempts}) • {fs.error}
                {/if}
                {#if fs.status === 'canceled'}
                  ⚠️ CANCELED
                {/if}
                {#if fs.status === 'pending' && fs.nextRetryAt}
                  <span class="n64-retry-countdown">
                    RETRYING IN {Math.max(0, Math.round((fs.nextRetryAt - Date.now())/1000))}S
                  </span>
                {/if}
              </div>
              
              <!-- N64 Progress Bar -->
              {#if fs.status !== 'pending' && fs.status !== 'completed' && fs.status !== 'error' && fs.status !== 'canceled'}
                <N64ProgressBar
                  value={fs.progress}
                  max={100}
                  theme={fs.gamingProgress?.theme || n64Themes[evolutionStage].theme}
                  animated={fs.gamingProgress?.animated !== false}
                  sparkle={fs.gamingProgress?.sparkle !== false}
                  size="sm"
                  retro={retro}
                  class="n64-file-progress"
                />
              {/if}
            </div>
            
            <div class="n64-file-actions">
              {#if fs.status === 'pending' && !uploading}
                <button 
                  type="button" 
                  class="n64-action-btn remove" 
                  title="Remove" 
                  onclick={(e) => { e.stopPropagation(); removeFile(index); }} 
                  aria-label="Remove file"
                >
                  ✕
                </button>
              {:else if fs.status === 'uploading'}
                <button 
                  type="button" 
                  class="n64-action-btn cancel" 
                  title="Cancel" 
                  onclick={(e) => { e.stopPropagation(); cancelUpload(index); }} 
                  aria-label="Cancel upload"
                >
                  ⏹
                </button>
              {:else if fs.status === 'error' || fs.status === 'canceled'}
                <button 
                  type="button" 
                  class="n64-action-btn retry" 
                  title="Retry" 
                  onclick={(e) => { e.stopPropagation(); retryFile(index); uploadFiles(); }} 
                  aria-label="Retry upload"
                >
                  ⟳
                </button>
                <button 
                  type="button" 
                  class="n64-action-btn remove" 
                  title="Remove" 
                  onclick={(e) => { e.stopPropagation(); removeFile(index); }} 
                  aria-label="Remove file"
                >
                  ✕
                </button>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- N64 Upload Progress -->
  {#if uploadStatus !== 'idle'}
    <div class="n64-upload-progress">
      <N64ProgressBar
        value={uploadProgress}
        max={100}
        theme={n64Themes[evolutionStage].theme}
        animated={true}
        sparkle={uploadStatus === 'uploading'}
        size="md"
        retro={retro}
        showPercentage={true}
        class="n64-main-progress"
      />
      <div class="n64-progress-text">
        {#if uploadStatus === 'uploading'}
          <N64LoadingRing size="sm" theme={n64Themes[evolutionStage].theme} speed="fast" />
          🎮 BATCH UPLOADING... {uploadProgress}%
        {:else if uploadStatus === 'processing'}
          <N64LoadingRing size="sm" theme="blue" speed="medium" />
          ⚡ PROCESSING WITH AI... {uploadProgress}%
        {:else if uploadStatus === 'completed'}
          <CheckCircle class="w-4 h-4 text-green-500" />
          ✅ UPLOAD COMPLETED SUCCESSFULLY
        {:else if uploadStatus === 'error'}
          <AlertCircle class="w-4 h-4 text-red-500" />
          ❌ {errorMessage || 'UPLOAD FAILED'}
        {/if}
      </div>
    </div>
  {/if}

  <!-- N64 Performance Metrics -->
  {#if uploading || performanceMetrics.completedFiles > 0}
    <div class="n64-performance-metrics" class:theme-{evolutionStage}>
      <div class="n64-metrics-header">
        <Zap class="w-4 h-4" />
        <span class="n64-metrics-title">🎮 N64 PERFORMANCE METRICS [{evolutionStage.toUpperCase()}]</span>
      </div>
      <div class="n64-metrics-grid">
        <div class="n64-metric-item">
          <span class="n64-metric-label">ACTIVE UPLOADS:</span>
          <span class="n64-metric-value">{activeUploads}/{maxConcurrency}</span>
        </div>
        <div class="n64-metric-item">
          <span class="n64-metric-label">COMPLETED:</span>
          <span class="n64-metric-value">{performanceMetrics.completedFiles}/{performanceMetrics.totalFiles}</span>
        </div>
        {#if performanceMetrics.averageUploadTime > 0}
          <div class="n64-metric-item">
            <span class="n64-metric-label">AVG TIME:</span>
            <span class="n64-metric-value">{Math.round(performanceMetrics.averageUploadTime)}MS</span>
          </div>
        {/if}
        {#if performanceMetrics.gpuTasksSubmitted > 0}
          <div class="n64-metric-item">
            <span class="n64-metric-label">GPU TASKS:</span>
            <span class="n64-metric-value">{performanceMetrics.gpuTasksSubmitted}</span>
          </div>
        {/if}
      </div>
      {#if enableGPUProcessing}
        <div class="n64-gpu-status">
          <span class="n64-gpu-indicator">🚀 GPU PROCESSING ENABLED [{evolutionStage.toUpperCase()}]</span>
        </div>
      {/if}
    </div>
  {/if}

  <!-- N64 Upload Actions -->
  <div class="n64-upload-actions">
    <button
      type="button"
      class="n64-upload-button" 
      class:theme-{evolutionStage}
      disabled={fileStates.length === 0 || uploading || disabled || fileStates.every(f=>['completed','canceled'].includes(f.status))}
      onclick={uploadFiles}
      aria-label="Start upload"
    >
      {#if uploading}
        <N64LoadingRing size="sm" theme={n64Themes[evolutionStage].theme} speed="fast" />
        🎮 UPLOADING...
      {:else}
        <Upload class="w-4 h-4" />
        🎮 UPLOAD TO MINIO [{evolutionStage.toUpperCase()}]
      {/if}
    </button>
    
    {#if uploading}
      <button
        type="button"
        class="n64-clear-button cancel"
        onclick={cancelAllUploads}
        aria-label="Cancel all uploads"
      >
        ❌ CANCEL ALL
      </button>
    {/if}
    
    {#if fileStates.length > 0 && !uploading}
      <button
        type="button"
        class="n64-clear-button clear"
        onclick={() => { 
          files = []; 
          fileStates = []; 
          if (fileInput) fileInput.value = ''; 
          liveMessage = 'Cleared selected files'; 
        }}
        aria-label="Clear selected files"
      >
        🗑️ CLEAR FILES
      </button>
    {/if}
  </div>

  {#if minioHealthy === false}
    <div class="n64-error-alert" role="alert">
      <AlertCircle class="w-4 h-4" /> 
      ⚠️ MINIO HEALTH CHECK FAILED – UPLOADS MAY NOT PERSIST
    </div>
  {/if}

  <div class="sr-only" aria-live="polite">{liveMessage}</div>
</div>

<style>
  .n64-upload-container {
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
    font-family: 'Courier New', monospace;
    position: relative;
  }

  .retro {
    image-rendering: pixelated;
    image-rendering: -moz-crisp-edges;
    image-rendering: crisp-edges;
  }

  .evolution-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .n64-drop-zone {
    border: 4px solid #FFD700;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: 
      inset 0 0 20px rgba(255, 215, 0, 0.1),
      0 0 20px rgba(255, 215, 0, 0.3);
  }

  .n64-drop-zone::before {
    content: '';
    position: absolute;
    top: -4px;
    left: -4px;
    right: -4px;
    bottom: -4px;
    background: linear-gradient(45deg, #FFD700, #FFA500, #FF6B35);
    z-index: -1;
    border-radius: 0;
  }

  .n64-drop-zone:hover:not(.uploading) {
    border-color: #FFA500;
    transform: scale(1.02);
    box-shadow: 
      inset 0 0 30px rgba(255, 165, 0, 0.2),
      0 0 30px rgba(255, 165, 0, 0.5);
  }

  .n64-drop-zone.drag-over {
    border-color: #FF6B35;
    background: linear-gradient(135deg, #2e1a1a 0%, #3e1616 100%);
    transform: scale(1.05);
    box-shadow: 
      inset 0 0 40px rgba(255, 107, 53, 0.3),
      0 0 40px rgba(255, 107, 53, 0.7);
  }

  .n64-drop-zone.has-files {
    border-color: #40FF40;
    background: linear-gradient(135deg, #1a2e1a 0%, #163e16 100%);
  }

  .n64-drop-zone.uploading {
    cursor: not-allowed;
    opacity: 0.8;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }

  /* Theme-specific styles */
  .theme-nes {
    border-color: #FF3030 !important;
    background: linear-gradient(135deg, #2e1a1a 0%, #3e1616 100%) !important;
  }

  .theme-snes {
    border-color: #4090FF !important;
    background: linear-gradient(135deg, #1a1a2e 0%, #161636 100%) !important;
  }

  .theme-n64 {
    border-color: #FFD700 !important;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
  }

  .theme-modern {
    border-color: #40FF40 !important;
    background: linear-gradient(135deg, #1a2e1a 0%, #163e16 100%) !important;
  }

  .n64-upload-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
  }

  .n64-upload-icon {
    animation: float 3s ease-in-out infinite;
  }

  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
  }

  .n64-upload-text h3 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: bold;
    color: #FFD700;
    text-shadow: 
      2px 2px 0 #000,
      -2px -2px 0 #000,
      2px -2px 0 #000,
      -2px 2px 0 #000;
    letter-spacing: 1px;
  }

  .n64-subtext {
    color: #CCCCCC;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
    text-shadow: 1px 1px 0 #000;
  }

  .n64-file-list {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .n64-file-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 2px solid #FFD700;
    text-align: left;
    transition: all 0.3s ease;
    position: relative;
    box-shadow: 
      inset 0 0 10px rgba(255, 215, 0, 0.1),
      0 0 10px rgba(0, 0, 0, 0.5);
  }

  .n64-file-item::before {
    content: '';
    position: absolute;
    top: 4px;
    left: 4px;
    right: 4px;
    bottom: 4px;
    border: 1px solid rgba(255, 215, 0, 0.3);
    pointer-events: none;
  }

  .n64-file-item.status-uploading {
    border-color: #4090FF;
    background: linear-gradient(135deg, #1a1a2e 0%, #0a0a1a 100%);
    animation: processingGlow 1.5s ease-in-out infinite alternate;
  }

  .n64-file-item.status-completed {
    border-color: #40FF40;
    background: linear-gradient(135deg, #1a2e1a 0%, #0a1a0a 100%);
  }

  .n64-file-item.status-error {
    border-color: #FF3030;
    background: linear-gradient(135deg, #2e1a1a 0%, #1a0a0a 100%);
  }

  @keyframes processingGlow {
    from { box-shadow: inset 0 0 10px rgba(64, 144, 255, 0.1), 0 0 10px rgba(0, 0, 0, 0.5); }
    to { box-shadow: inset 0 0 20px rgba(64, 144, 255, 0.3), 0 0 20px rgba(64, 144, 255, 0.3); }
  }

  .file-icon.n64-icon {
    color: #FFD700;
    flex-shrink: 0;
  }

  .n64-file-name {
    font-weight: bold;
    color: #FFD700;
    word-break: break-word;
    text-shadow: 1px 1px 0 #000;
  }

  .n64-file-size {
    font-size: 0.8rem;
    color: #CCCCCC;
    text-shadow: 1px 1px 0 #000;
  }

  .n64-file-status {
    font-size: 0.75rem;
    color: #FFFFFF;
    font-weight: bold;
    text-shadow: 1px 1px 0 #000;
    letter-spacing: 0.5px;
  }

  .n64-retry-countdown {
    color: #FFA500;
    animation: blink 1s infinite;
  }

  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
  }

  .n64-file-actions {
    display: flex;
    gap: 0.5rem;
  }

  .n64-action-btn {
    padding: 0.5rem;
    background: #FFD700;
    color: #000;
    border: 2px solid #FFA500;
    cursor: pointer;
    font-family: 'Courier New', monospace;
    font-weight: bold;
    font-size: 0.8rem;
    transition: all 0.2s ease;
    box-shadow: 
      inset 1px 1px 0 rgba(255, 255, 255, 0.3),
      inset -1px -1px 0 rgba(0, 0, 0, 0.3);
  }

  .n64-action-btn:hover {
    background: #FFA500;
    transform: translateY(-1px);
  }

  .n64-action-btn.cancel {
    background: #FF3030;
    border-color: #CC0000;
    color: #FFF;
  }

  .n64-action-btn.retry {
    background: #40FF40;
    border-color: #00CC00;
    color: #000;
  }

  .n64-upload-progress {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 2px solid #FFD700;
    box-shadow: 
      inset 0 0 20px rgba(255, 215, 0, 0.1),
      0 0 10px rgba(0, 0, 0, 0.5);
  }

  .n64-progress-text {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.9rem;
    color: #FFD700;
    font-weight: bold;
    margin-top: 1rem;
    text-shadow: 1px 1px 0 #000;
    letter-spacing: 0.5px;
  }

  .n64-performance-metrics {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
    border: 2px solid #4090FF;
    box-shadow: 
      inset 0 0 20px rgba(64, 144, 255, 0.1),
      0 0 15px rgba(64, 144, 255, 0.3);
  }

  .n64-metrics-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    color: #4090FF;
  }

  .n64-metrics-title {
    font-weight: bold;
    font-size: 0.9rem;
    text-shadow: 1px 1px 0 #000;
    letter-spacing: 0.5px;
  }

  .n64-metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .n64-metric-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
    border: 1px solid #666;
    box-shadow: inset 1px 1px 0 rgba(255, 255, 255, 0.1);
  }

  .n64-metric-label {
    font-size: 0.75rem;
    color: #CCCCCC;
    font-weight: bold;
    text-shadow: 1px 1px 0 #000;
  }

  .n64-metric-value {
    font-size: 0.8rem;
    color: #FFD700;
    font-weight: bold;
    text-shadow: 1px 1px 0 #000;
  }

  .n64-gpu-status {
    text-align: center;
    padding: 0.75rem;
    background: linear-gradient(135deg, #1a2e1a 0%, #0a1a0a 100%);
    border: 1px solid #40FF40;
  }

  .n64-gpu-indicator {
    font-size: 0.8rem;
    color: #40FF40;
    font-weight: bold;
    text-shadow: 1px 1px 0 #000;
    letter-spacing: 0.5px;
  }

  .n64-upload-actions {
    margin-top: 1.5rem;
    display: flex;
    gap: 1rem;
  }

  .n64-upload-button {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    padding: 1rem 1.5rem;
    background: #FFD700;
    color: #000;
    border: 3px solid #FFA500;
    font-family: 'Courier New', monospace;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    box-shadow: 
      inset 2px 2px 0 rgba(255, 255, 255, 0.3),
      inset -2px -2px 0 rgba(0, 0, 0, 0.3),
      0 4px 8px rgba(0, 0, 0, 0.5);
  }

  .n64-upload-button:hover:not(:disabled) {
    background: #FFA500;
    transform: translateY(-2px);
    box-shadow: 
      inset 2px 2px 0 rgba(255, 255, 255, 0.3),
      inset -2px -2px 0 rgba(0, 0, 0, 0.3),
      0 6px 12px rgba(0, 0, 0, 0.7);
  }

  .n64-upload-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .n64-clear-button {
    padding: 1rem;
    background: #666666;
    color: #FFF;
    border: 2px solid #444444;
    font-family: 'Courier New', monospace;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    box-shadow: 
      inset 1px 1px 0 rgba(255, 255, 255, 0.2),
      inset -1px -1px 0 rgba(0, 0, 0, 0.3);
  }

  .n64-clear-button:hover {
    background: #777777;
    transform: translateY(-1px);
  }

  .n64-clear-button.cancel {
    background: #FF3030;
    border-color: #CC0000;
  }

  .n64-clear-button.cancel:hover {
    background: #FF5555;
  }

  .n64-error-alert {
    margin-top: 1rem;
    padding: 1rem;
    background: linear-gradient(135deg, #2e1a1a 0%, #1a0a0a 100%);
    border: 2px solid #FF3030;
    color: #FF6666;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: bold;
    text-shadow: 1px 1px 0 #000;
    box-shadow: 
      inset 0 0 10px rgba(255, 48, 48, 0.1),
      0 0 10px rgba(255, 48, 48, 0.3);
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* Responsive adjustments */
  @media (max-width: 640px) {
    .n64-upload-container {
      max-width: 100%;
    }
    
    .n64-drop-zone {
      padding: 1.5rem;
      min-height: 180px;
    }
    
    .n64-metrics-grid {
      grid-template-columns: 1fr 1fr;
    }
    
    .n64-upload-actions {
      flex-direction: column;
    }
    
    .n64-file-item {
      padding: 0.75rem;
    }
    
    .n64-upload-text h3 {
      font-size: 1rem;
    }
  }
</style>