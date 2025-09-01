<!--
  Comprehensive Ollama Integration Demo
  
  Demonstrates fully integrated and wired Ollama ecosystem:
  - Comprehensive Ollama Summarizer
  - LangChain + RAG integration
  - CUDA GPU acceleration
  - Multi-model support
  - Streaming responses
  - All API endpoints working
-->

<script lang="ts">
  import { onMount } from 'svelte';
  import { 
    ollamaIntegrationLayer, 
    type IntegratedChatRequest,
    type IntegratedChatResponse,
    type OllamaServiceStatus 
  } from '$lib/services/ollama-integration-layer';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import { Card, CardContent, CardHeader, CardTitle } from "$lib/components/ui/card";
  import { Textarea } from '$lib/components/ui/textarea';
  import { Input } from '$lib/components/ui/input';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { 
    Activity, 
    Brain, 
    CheckCircle, 
    AlertCircle, 
    Loader2, 
    Send, 
    FileText,
    Zap,
    Database,
    Settings
  } from 'lucide-svelte';

  // State variables
let serviceStatus = $state<OllamaServiceStatus | null >(null);
let isInitialized = $state(false);
let isLoading = $state(false);
let testMessage = $state('Analyze this legal contract for key risks and opportunities.');
let testDocument = $state(`MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into on [DATE] by and between Company A and Company B.

CONFIDENTIAL INFORMATION: Each party may disclose proprietary information, trade secrets, technical data, and business information.

OBLIGATIONS: Each party agrees to maintain confidentiality and use the information solely for evaluation purposes.

TERM: This agreement shall remain in effect for two (2) years from the date of execution.

NON-CIRCUMVENTION: Neither party shall directly or indirectly circumvent the other party in business dealings.`);
let responses = $state<Array<{
    type: string;
    request: any;
    response: any;
    timestamp: Date;
    processingTime: number;
  }> >([]);
let streamingResponse = $state('');
let isStreaming = $state(false);

  // Reactive subscriptions
let activeRequests = $state(0);
let stats = $state<any >(null);

  onMount(async () => {
    try {
      console.log('🚀 Initializing Ollama Integration Demo...');
      
      // Initialize the integration layer
      await ollamaIntegrationLayer.initialize();
      
      // Subscribe to reactive stores
      ollamaIntegrationLayer.isInitialized.subscribe(val => {
        isInitialized = val;
      });
      
      ollamaIntegrationLayer.activeRequests.subscribe(val => {
        activeRequests = val;
      });
      
      ollamaIntegrationLayer.stats.subscribe(val => {
        stats = val;
      });

      // Get initial service status
      serviceStatus = await ollamaIntegrationLayer.getServiceHealth();
      
      console.log('✅ Demo initialized successfully');
      
    } catch (error) {
      console.error('❌ Demo initialization failed:', error);
    }
  });

  // ========================================================================
  // TEST FUNCTIONS
  // ========================================================================

  async function testBasicChat() {
    if (!isInitialized) return;
    
    isLoading = true;
    const startTime = Date.now();
    
    try {
      const request: IntegratedChatRequest = {
        message: testMessage,
        model: 'gemma3-legal:latest',
        temperature: 0.3,
        useRAG: false,
        stream: false
      };

      const response = await ollamaIntegrationLayer.processChat(request);
      
      responses = [{
        type: 'Basic Chat',
        request,
        response,
        timestamp: new Date(),
        processingTime: Date.now() - startTime
      }, ...responses];

    } catch (error) {
      console.error('Basic chat test failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function testDocumentSummary() {
    if (!isInitialized) return;
    
    isLoading = true;
    const startTime = Date.now();
    
    try {
      const request: IntegratedChatRequest = {
        message: 'Please provide a comprehensive analysis of this document.',
        model: 'gemma3-legal:latest',
        documentContext: {
          type: 'contract',
          content: testDocument,
          metadata: { source: 'demo', title: 'NDA Contract' }
        },
        summaryOptions: {
          includeSummary: true,
          includeKeyPoints: true,
          includeLegalAnalysis: true,
          includeEmbeddings: false
        },
        advancedOptions: {
          useGPU: true,
          enableCaching: true,
          enableStreaming: false
        }
      };

      const response = await ollamaIntegrationLayer.processChat(request);
      
      responses = [{
        type: 'Document Summary',
        request,
        response,
        timestamp: new Date(),
        processingTime: Date.now() - startTime
      }, ...responses];

    } catch (error) {
      console.error('Document summary test failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function testRAGQuery() {
    if (!isInitialized) return;
    
    isLoading = true;
    const startTime = Date.now();
    
    try {
      const request: IntegratedChatRequest = {
        message: 'What are the standard terms for confidentiality agreements?',
        model: 'gemma3-legal:latest',
        useRAG: true,
        documentContext: {
          type: 'contract',
          content: testDocument
        }
      };

      const response = await ollamaIntegrationLayer.processChat(request);
      
      responses = [{
        type: 'RAG Query',
        request,
        response,
        timestamp: new Date(),
        processingTime: Date.now() - startTime
      }, ...responses];

    } catch (error) {
      console.error('RAG query test failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function testStreamingResponse() {
    if (!isInitialized) return;
    
    isStreaming = true;
    streamingResponse = '';
    
    try {
      const request: IntegratedChatRequest = {
        message: 'Provide a detailed legal analysis of the NDA document.',
        model: 'gemma3-legal:latest',
        documentContext: {
          type: 'contract',
          content: testDocument
        },
        summaryOptions: {
          includeSummary: true,
          includeKeyPoints: true,
          includeLegalAnalysis: true
        },
        advancedOptions: {
          enableStreaming: true,
          useGPU: true
        }
      };

      const streamGenerator = ollamaIntegrationLayer.processStreamingChatForUI(request);
      
      for await (const partialResponse of streamGenerator) {
        streamingResponse = partialResponse.response || streamingResponse;
      }

    } catch (error) {
      console.error('Streaming test failed:', error);
      streamingResponse = `Error: ${error}`;
    } finally {
      isStreaming = false;
    }
  }

  async function testDirectAPI() {
    isLoading = true;
    const startTime = Date.now();
    
    try {
      const response = await fetch('/api/ollama/comprehensive-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: testDocument,
          type: 'contract',
          options: {
            includeEmbeddings: false,
            enableRAG: true,
            useGPU: true,
            cacheResult: true
          }
        })
      });

      const result = await response.json();
      
      responses = [{
        type: 'Direct API',
        request: { content: testDocument, type: 'contract' },
        response: result,
        timestamp: new Date(),
        processingTime: Date.now() - startTime
      }, ...responses];

    } catch (error) {
      console.error('Direct API test failed:', error);
    } finally {
      isLoading = false;
    }
  }

  async function refreshServiceStatus() {
    try {
      serviceStatus = await ollamaIntegrationLayer.getServiceHealth();
      await ollamaIntegrationLayer.refreshStats();
    } catch (error) {
      console.error('Failed to refresh service status:', error);
    }
  }

  async function warmupServices() {
    isLoading = true;
    try {
      await ollamaIntegrationLayer.warmupServices();
      await refreshServiceStatus();
    } catch (error) {
      console.error('Warmup failed:', error);
    } finally {
      isLoading = false;
    }
  }

  // ========================================================================
  // HELPER FUNCTIONS
  // ========================================================================

  function getStatusBadge(status: string) {
    switch (status) {
      case 'healthy': return { variant: 'default' as const, icon: CheckCircle };
      case 'unhealthy': return { variant: 'destructive' as const, icon: AlertCircle };
      default: return { variant: 'secondary' as const, icon: Activity };
    }
  }
</script>

<!-- Page Header -->
<div class="container mx-auto p-6">
  <div class="flex items-center justify-between mb-6">
    <div>
      <h1 class="text-3xl font-bold flex items-center gap-3">
        <Brain class="w-8 h-8 text-blue-600" />
        Comprehensive Ollama Integration Demo
      </h1>
      <p class="text-muted-foreground mt-2">
        Complete integration testing for all Ollama services and APIs
      </p>
    </div>
    
    <div class="flex items-center gap-2">
      <Badge variant={isInitialized ? 'default' : 'secondary'}>
        {isInitialized ? 'Ready' : 'Initializing'}
      </Badge>
      <Button variant="outline" size="sm" on:on:click={refreshServiceStatus}>
        <Settings class="w-4 h-4" />
      </Button>
    </div>
  </div>

  <!-- Service Status Dashboard -->
  {#if serviceStatus}
    <Card class="mb-6">
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <Activity class="w-5 h-5" />
          Service Status Dashboard
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {#each Object.entries(serviceStatus) as [serviceName, service]}
            {@const badge = getStatusBadge(service.status)}
            <div class="text-center">
              <div class="flex items-center justify-center mb-2">
                <Badge variant={badge.variant} class="flex items-center gap-1">
                  <svelte:component this={badge.icon} class="w-3 h-3" />
                  {service.status}
                </Badge>
              </div>
              <p class="text-sm font-medium capitalize">{serviceName}</p>
              {#if 'nodes' in service}
                <p class="text-xs text-muted-foreground">Nodes: {service.nodes}</p>
              {/if}
              {#if 'health' in service}
                <p class="text-xs text-muted-foreground">Health: {service.health}%</p>
              {/if}
            </div>
          {/each}
        </div>
        
        {#if stats}
          <div class="mt-4 pt-4 border-t">
            <div class="grid grid-cols-4 gap-4 text-center">
              <div>
                <p class="text-sm font-medium">Requests Processed</p>
                <p class="text-2xl font-bold text-blue-600">{stats.performance.requestsProcessed}</p>
              </div>
              <div>
                <p class="text-sm font-medium">Average Latency</p>
                <p class="text-2xl font-bold text-green-600">{Math.round(stats.performance.averageLatency)}ms</p>
              </div>
              <div>
                <p class="text-sm font-medium">Cache Hit Rate</p>
                <p class="text-2xl font-bold text-purple-600">{Math.round(stats.performance.cacheHitRate * 100)}%</p>
              </div>
              <div>
                <p class="text-sm font-medium">Active Requests</p>
                <p class="text-2xl font-bold text-orange-600">{activeRequests}</p>
              </div>
            </div>
          </div>
        {/if}
      </CardContent>
    </Card>
  {/if}

  <div class="grid lg:grid-cols-2 gap-6">
    <!-- Test Configuration Panel -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center gap-2">
          <FileText class="w-5 h-5" />
          Test Configuration
        </CardTitle>
      </CardHeader>
      <CardContent class="space-y-4">
        <!-- Test Message -->
        <div>
          <label class="text-sm font-medium">Test Message:</label>
          <Input bind:value={testMessage} placeholder="Enter test message..." />
        </div>
        
        <!-- Test Document -->
        <div>
          <label class="text-sm font-medium">Test Document:</label>
          <Textarea bind:value={testDocument} rows={8} placeholder="Enter test document..." />
        </div>

        <!-- Test Actions -->
        <div class="grid grid-cols-2 gap-2">
          <Button 
            on:on:click={testBasicChat} 
            disabled={!isInitialized || isLoading}
            class="flex items-center gap-2"
          >
            {#if isLoading}
              <Loader2 class="w-4 h-4 animate-spin" />
            {:else}
              <Send class="w-4 h-4" />
            {/if}
            Basic Chat
          </Button>
          
          <Button 
            on:on:click={testDocumentSummary} 
            disabled={!isInitialized || isLoading}
            variant="secondary"
            class="flex items-center gap-2"
          >
            <FileText class="w-4 h-4" />
            Document Summary
          </Button>
          
          <Button 
            on:on:click={testRAGQuery} 
            disabled={!isInitialized || isLoading}
            variant="outline"
            class="flex items-center gap-2"
          >
            <Database class="w-4 h-4" />
            RAG Query
          </Button>
          
          <Button 
            on:on:click={testDirectAPI} 
            disabled={!isInitialized || isLoading}
            variant="outline"
            class="flex items-center gap-2"
          >
            <Zap class="w-4 h-4" />
            Direct API
          </Button>
        </div>

        <!-- Streaming Test -->
        <div class="border-t pt-4">
          <Button 
            on:on:click={testStreamingResponse} 
            disabled={!isInitialized || isStreaming}
            variant="secondary"
            class="w-full flex items-center gap-2"
          >
            {#if isStreaming}
              <Loader2 class="w-4 h-4 animate-spin" />
              Streaming...
            {:else}
              <Activity class="w-4 h-4" />
              Test Streaming
            {/if}
          </Button>
          
          {#if streamingResponse}
            <div class="mt-4 p-3 bg-muted rounded-lg">
              <p class="text-sm font-medium">Streaming Response:</p>
              <ScrollArea class="h-32 mt-2">
                <p class="text-sm whitespace-pre-wrap">{streamingResponse}</p>
              </ScrollArea>
            </div>
          {/if}
        </div>

        <!-- System Actions -->
        <div class="border-t pt-4 space-y-2">
          <Button 
            on:on:click={warmupServices} 
            disabled={isLoading}
            variant="outline"
            class="w-full flex items-center gap-2"
          >
            <Zap class="w-4 h-4" />
            Warmup All Services
          </Button>
        </div>
      </CardContent>
    </Card>

    <!-- Response Display -->
    <Card>
      <CardHeader>
        <CardTitle class="flex items-center justify-between">
          <span class="flex items-center gap-2">
            <Brain class="w-5 h-5" />
            Test Results ({responses.length})
          </span>
          {#if responses.length > 0}
            <Button 
              variant="outline" 
              size="sm" 
              on:on:click={() => responses = []}
            >
              Clear
            </Button>
          {/if}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea class="h-96">
          {#if responses.length === 0}
            <div class="text-center text-muted-foreground py-8">
              <FileText class="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No test results yet</p>
              <p class="text-sm">Run a test to see results here</p>
            </div>
          {:else}
            <div class="space-y-4">
              {#each responses as result}
                <Card>
                  <CardHeader class="pb-2">
                    <div class="flex items-center justify-between">
                      <Badge variant="outline">{result.type}</Badge>
                      <span class="text-xs text-muted-foreground">
                        {result.processingTime}ms • {result.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                  </CardHeader>
                  <CardContent class="pt-2">
                    <!-- Response Content -->
                    {#if result.response.response}
                      <div class="mb-3">
                        <p class="text-sm font-medium">Response:</p>
                        <p class="text-sm bg-muted p-2 rounded mt-1 whitespace-pre-wrap">
                          {result.response.response.substring(0, 300)}{result.response.response.length > 300 ? '...' : ''}
                        </p>
                      </div>
                    {/if}

                    <!-- Summary -->
                    {#if result.response.summary}
                      <div class="mb-3">
                        <p class="text-sm font-medium">Summary:</p>
                        <p class="text-sm bg-blue-50 p-2 rounded mt-1">
                          {result.response.summary.summary?.substring(0, 200)}...
                        </p>
                        {#if result.response.summary.keyPoints?.length}
                          <div class="mt-2">
                            <p class="text-xs font-medium">Key Points:</p>
                            <ul class="text-xs list-disc list-inside mt-1">
                              {#each result.response.summary.keyPoints.slice(0, 3) as point}
                                <li>{point}</li>
                              {/each}
                            </ul>
                          </div>
                        {/if}
                      </div>
                    {/if}

                    <!-- Performance -->
                    {#if result.response.performance}
                      <div class="flex gap-4 text-xs text-muted-foreground">
                        <span>Duration: {result.response.performance.duration}ms</span>
                        <span>Tokens: {result.response.performance.tokens}</span>
                        <span>Model: {result.response.performance.model}</span>
                      </div>
                    {/if}

                    <!-- Integration Info -->
                    {#if result.response.integration}
                      <div class="mt-2 p-2 bg-green-50 rounded text-xs">
                        <p><strong>Integration:</strong> {result.response.integration.processingPath}</p>
                        <p><strong>Services:</strong> {result.response.integration.servicesUsed.join(', ')}</p>
                      </div>
                    {/if}
                  </CardContent>
                </Card>
              {/each}
            </div>
          {/if}
        </ScrollArea>
      </CardContent>
    </Card>
  </div>
</div>

<style>
  .container {
    max-width: 100%;
  }
</style>