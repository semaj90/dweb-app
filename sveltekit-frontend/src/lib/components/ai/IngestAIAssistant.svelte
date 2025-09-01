<script lang="ts">
  /**
   * Enhanced AI-Powered Document Ingest Assistant
   * Integrates with your existing AI agent store and production architecture
   * Uses Bits UI + Melt UI following your established component patterns
   */
  
  import { onMount } from 'svelte';
  import { writable, derived } from 'svelte/store';
  import Button from '$lib/components/ui/Button.svelte';
  import Input from '$lib/components/ui/input/Input.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import CardContent from '$lib/components/ui/CardContent.svelte';
  import CardHeader from '$lib/components/ui/CardHeader.svelte';
  import CardTitle from '$lib/components/ui/CardTitle.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Progress from '$lib/components/ui/progress/Progress.svelte';
  import Alert from '$lib/components/ui/alert/Alert.svelte';
  import AlertDescription from '$lib/components/ui/alert/AlertDescription.svelte';
  import Separator from '$lib/components/ui/separator/Separator.svelte';
  import Textarea from '$lib/components/ui/textarea/Textarea.svelte';
  import Label from '$lib/components/ui/label/LabelCompat.svelte';
  
  // Your established store patterns
  import { 
    aiAgentStore, 
    isProcessing, 
    systemHealth, 
    performanceMetrics,
    currentConversation 
  } from '$lib/stores/ai-agent';
  
  import { enhancedIngestService } from '$lib/services/enhanced-ingest-integration';
  
  // Component state following your patterns
let documentTitle = $state('');
let documentContent = $state('');
let caseId = $state('');
let selectedDocumentType = $state('legal');
let batchMode = $state(false);
  let batchDocuments = writable([]);
  
  // Processing state
  let ingestResults = writable([]);
  let currentProgress = writable(0);
  let processingStatus = writable('idle');
  let errors = writable([]);
  
  // Derived states following your patterns
  const canIngest = derived(
    [processingStatus],
    ([$status]) => $status === 'idle' && (documentTitle.trim() && documentContent.trim())
  );
  
  const hasResults = derived(
    ingestResults,
    ($results) => $results.length > 0
  );
  
  // Document types following your legal AI patterns
  const documentTypes = [
    { value: 'legal', label: 'Legal Document', icon: '⚖️' },
    { value: 'evidence', label: 'Evidence', icon: '🔍' },
    { value: 'case', label: 'Case File', icon: '📁' },
    { value: 'contract', label: 'Contract', icon: '📜' },
    { value: 'precedent', label: 'Legal Precedent', icon: '📚' }
  ];
  
  // Enhanced ingest function with AI integration
  async function ingestDocument() {
    if (!$canIngest) return;
    
    processingStatus.set('processing');
    currentProgress.set(10);
    
    try {
      const request = {
        title: documentTitle,
        content: documentContent,
        case_id: caseId || undefined,
        metadata: {
          document_type: selectedDocumentType,
          source: 'ai_assistant_ui',
          ai_enhanced: true,
          // Integrate with your AI agent session
          ai_session_id: $aiAgentStore.activeSessionId,
        }
      };
      
      currentProgress.set(30);
      
      // Use your enhanced ingest service
      const result = await enhancedIngestService.ingestDocument(request);
      
      currentProgress.set(70);
      
      // Generate AI summary using your existing chat system
      if (result.success) {
        await generateAISummary(result.documentId, documentContent);
      }
      
      currentProgress.set(100);
      
      // Update results
      ingestResults.update(results => [...results, {
        ...result,
        title: documentTitle,
        type: selectedDocumentType,
        timestamp: new Date()
      }]);
      
      // Clear form
      clearForm();
      
      processingStatus.set('completed');
      setTimeout(() => processingStatus.set('idle'), 2000);
      
    } catch (error) {
      console.error('Ingest failed:', error);
      errors.update(errs => [...errs, {
        id: Date.now(),
        message: error.message,
        timestamp: new Date(),
        type: 'ingest_error'
      }]);
      
      processingStatus.set('error');
      setTimeout(() => processingStatus.set('idle'), 3000);
    }
  }
  
  // AI summary generation using your existing chat patterns
  async function generateAISummary(documentId: string, content: string) {
    try {
      const prompt = `Please provide a concise legal analysis summary of this document:\n\n${content.substring(0, 1000)}...`;
      
      // Use your existing AI agent for summary
      await aiAgentStore.sendMessage(prompt, {
        document_id: documentId,
        analysis_type: 'legal_summary',
        source: 'ingest_assistant'
      });
      
    } catch (error) {
      console.warn('AI summary generation failed:', error);
    }
  }
  
  // Batch processing following your batch patterns
  async function processBatch() {
    const documents = $batchDocuments;
    if (documents.length === 0) return;
    
    processingStatus.set('batch_processing');
    currentProgress.set(0);
    
    try {
      const batchRequest = documents.map(doc => ({
        title: doc.title,
        content: doc.content,
        case_id: doc.case_id,
        metadata: {
          document_type: doc.type || 'legal',
          batch_processing: true,
          source: 'ai_assistant_batch'
        }
      }));
      
      const result = await enhancedIngestService.ingestBatch(batchRequest);
      
      currentProgress.set(100);
      
      // Update results with batch information
      ingestResults.update(results => [...results, {
        ...result,
        is_batch: true,
        timestamp: new Date()
      }]);
      
      batchDocuments.set([]);
      processingStatus.set('completed');
      setTimeout(() => processingStatus.set('idle'), 2000);
      
    } catch (error) {
      console.error('Batch processing failed:', error);
      errors.update(errs => [...errs, {
        id: Date.now(),
        message: `Batch processing failed: ${error.message}`,
        timestamp: new Date(),
        type: 'batch_error'
      }]);
      
      processingStatus.set('error');
      setTimeout(() => processingStatus.set('idle'), 3000);
    }
  }
  
  function clearForm() {
    documentTitle = '';
    documentContent = '';
    caseId = '';
  }
  
  function addToBatch() {
    if (!documentTitle.trim() || !documentContent.trim()) return;
    
    batchDocuments.update(docs => [...docs, {
      id: Date.now(),
      title: documentTitle,
      content: documentContent,
      case_id: caseId,
      type: selectedDocumentType
    }]);
    
    clearForm();
  }
  
  function removeFromBatch(id: number) {
    batchDocuments.update(docs => docs.filter(doc => doc.id !== id));
  }
  
  function dismissError(errorId: number) {
    errors.update(errs => errs.filter(err => err.id !== errorId));
  }
  
  onMount(() => {
    // Initialize AI agent connection following your patterns
    aiAgentStore.connect().catch(console.error);
  });
</script>

<!-- Component HTML following your UI patterns -->
<div class="w-full max-w-4xl mx-auto p-6 space-y-6">
  <!-- Header with system status -->
  <div class="flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-3 h-3 rounded-full {$systemHealth === 'healthy' ? 'bg-green-500' : $systemHealth === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'}"></div>
      <h1 class="text-2xl font-bold">AI-Powered Document Ingest</h1>
      <Badge variant={$systemHealth === 'healthy' ? 'success' : 'warning'}>
        {$systemHealth}
      </Badge>
    </div>
    
    <div class="flex items-center space-x-2">
      <Badge variant="outline">
        {$isProcessing ? 'Processing...' : 'Ready'}
      </Badge>
      {#if $performanceMetrics.totalRequests > 0}
        <Badge variant="secondary">
          {$performanceMetrics.successRate.toFixed(1)}% Success
        </Badge>
      {/if}
    </div>
  </div>
  
  <!-- Error Display -->
  {#each $errors as error (error.id)}
    <Alert variant="destructive" class="mb-4">
      <AlertDescription class="flex items-center justify-between">
        <span>{error.message}</span>
        <Button variant="ghost" size="sm" on:on:click={() => dismissError(error.id)}>
          ✕
        </Button>
      </AlertDescription>
    </Alert>
  {/each}
  
  <!-- Progress Indicator -->
  {#if $processingStatus !== 'idle'}
    <Card>
      <CardContent class="p-4">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium">
            {$processingStatus === 'processing' ? 'Processing Document...' : 
             $processingStatus === 'batch_processing' ? 'Processing Batch...' : 
             $processingStatus === 'completed' ? 'Completed Successfully!' : 
             'Processing Failed'}
          </span>
          <span class="text-sm text-muted-foreground">{$currentProgress}%</span>
        </div>
        <Progress value={$currentProgress} class="w-full" />
      </CardContent>
    </Card>
  {/if}
  
  <!-- Main Input Form -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <Card>
      <CardHeader>
        <CardTitle>Document Details</CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <div class="space-y-2">
          <Label for="title">Document Title</Label>
          <Input
            id="title"
            bind:value={documentTitle}
            placeholder="Enter document title..."
            disabled={$isProcessing}
          />
        </div>
        
        <div class="space-y-2">
          <Label for="case-id">Case ID (Optional)</Label>
          <Input
            id="case-id"
            bind:value={caseId}
            placeholder="CASE-2024-001"
            disabled={$isProcessing}
          />
        </div>
        
        <div class="space-y-2">
          <Label>Document Type</Label>
          <div class="grid grid-cols-2 gap-2">
            {#each documentTypes as type}
              <Button
                variant={selectedDocumentType === type.value ? 'default' : 'outline'}
                size="sm"
                on:on:click={() => selectedDocumentType = type.value}
                disabled={$isProcessing}
                class="justify-start"
              >
                <span class="mr-2">{type.icon}</span>
                {type.label}
              </Button>
            {/each}
          </div>
        </div>
        
        <div class="space-y-2">
          <Label for="content">Document Content</Label>
          <Textarea
            id="content"
            bind:value={documentContent}
            placeholder="Paste or type document content here..."
            rows={8}
            disabled={$isProcessing}
          />
        </div>
        
        <div class="flex space-x-2">
          <Button
            on:on:click={ingestDocument}
            disabled={!$canIngest || $isProcessing}
            class="flex-1"
          >
            {$isProcessing ? 'Processing...' : '🚀 Ingest Document'}
          </Button>
          
          <Button
            variant="outline"
            on:on:click={addToBatch}
            disabled={!documentTitle.trim() || !documentContent.trim() || $isProcessing}
          >
            ➕ Add to Batch
          </Button>
        </div>
      </CardContent>
    </Card>
    
    <!-- Batch Processing Panel -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          Batch Processing
          {#if $batchDocuments.length > 0}
            <Badge>{$batchDocuments.length} documents</Badge>
          {/if}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {#if $batchDocuments.length === 0}
          <div class="text-center text-muted-foreground py-8">
            <p>No documents in batch</p>
            <p class="text-sm">Add documents to process multiple files at once</p>
          </div>
        {:else}
          <div class="space-y-2 mb-4 max-h-40 overflow-y-auto">
            {#each $batchDocuments as doc (doc.id)}
              <div class="flex items-center justify-between p-2 bg-muted rounded">
                <div class="flex-1 truncate">
                  <div class="font-medium text-sm truncate">{doc.title}</div>
                  <div class="text-xs text-muted-foreground">
                    {doc.type} • {doc.content.length} chars
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  on:on:click={() => removeFromBatch(doc.id)}
                >
                  ✕
                </Button>
              </div>
            {/each}
          </div>
          
          <div class="space-y-2">
            <Button
              on:on:click={processBatch}
              disabled={$isProcessing}
              class="w-full"
            >
              {$processingStatus === 'batch_processing' ? 'Processing Batch...' : `🔥 Process ${$batchDocuments.length} Documents`}
            </Button>
            <Button
              variant="outline"
              on:on:click={() => batchDocuments.set([])}
              disabled={$isProcessing}
              size="sm"
              class="w-full"
            >
              Clear Batch
            </Button>
          </div>
        {/if}
      </CardContent>
    </Card>
  </div>
  
  <!-- Results Display -->
  {#if $hasResults}
    <Card>
      <CardHeader>
        <CardTitle>Processing Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-4">
          {#each $ingestResults as result (result.documentId || result.batchId)}
            <div class="border rounded-lg p-4">
              <div class="flex items-start justify-between mb-2">
                <div>
                  <div class="font-medium">
                    {result.is_batch ? `Batch: ${result.processed} documents` : result.title}
                  </div>
                  <div class="text-sm text-muted-foreground">
                    {result.is_batch ? `Success Rate: ${result.successRate}` : `Type: ${result.type}`}
                  </div>
                </div>
                <Badge variant="success">✓ Completed</Badge>
              </div>
              
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div class="text-muted-foreground">Processing Time</div>
                  <div class="font-medium">
                    {result.processingTime ? `${result.processingTime.toFixed(1)}ms` : 'N/A'}
                  </div>
                </div>
                <div>
                  <div class="text-muted-foreground">Document ID</div>
                  <div class="font-mono text-xs">
                    {result.documentId?.substring(0, 8) || result.batchId?.substring(0, 8)}...
                  </div>
                </div>
                <div>
                  <div class="text-muted-foreground">Embedding ID</div>
                  <div class="font-mono text-xs">
                    {result.embeddingId?.substring(0, 8)}...
                  </div>
                </div>
                <div>
                  <div class="text-muted-foreground">Timestamp</div>
                  <div class="text-xs">
                    {result.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
  
  <!-- AI Chat Integration (if active conversation exists) -->
  {#if $currentConversation.length > 0}
    <Card>
      <CardHeader>
        <CardTitle>AI Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div class="space-y-4 max-h-60 overflow-y-auto">
          {#each $currentConversation.slice(-2) as message}
            <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
              <div class="max-w-[80%] p-3 rounded-lg {message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'}">
                <div class="text-sm">
                  {message.content}
                </div>
                {#if message.sources?.length > 0}
                  <div class="text-xs opacity-75 mt-2">
                    Sources: {message.sources.length} documents
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </CardContent>
    </Card>
  {/if}
</div>

<style>
  /* Custom styles following your YoRHa theme patterns */
  :global(.progress-bar) {
    background: linear-gradient(90deg, #3b82f6 0%, #06b6d4 100%);
  }
  
  /* Enhanced focus states following your accessibility patterns */
  :global(.enhanced-focus:focus) {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
  }
</style>