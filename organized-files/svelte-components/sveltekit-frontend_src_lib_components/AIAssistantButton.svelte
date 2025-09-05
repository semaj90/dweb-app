// Production AI Assistant Component - bits-ui Implementation
// File: AIAssistantButton.svelte

<script lang="ts">
  interface Props {
    onresponse?: (event?: any) => void;
    onerror?: (event?: any) => void;
  }
  let {
    query = '',
    isProcessing = false,
    systemStatus = 'unknown',
    responseTime = 0
  } = $props();



    import { Button } from 'bits-ui';
  import { Badge } from 'bits-ui';
  import { Loader2, Brain, Zap, AlertTriangle } from 'lucide-svelte';
  
          
    
  let apiLogs = [];
  let currentTest = null;
  
  const logAPI = (endpoint, status, time, error = null) => {
    apiLogs = [{
      endpoint,
      status,
      time,
      error,
      timestamp: Date.now()
    }, ...apiLogs.slice(0, 9)];
  };

  const testGemma3 = async () => {
    if (isProcessing) return;
    
    isProcessing = true;
    currentTest = 'gemma3';
    const startTime = Date.now();
    
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: query || 'Legal AI status check',
          stream: false,
          options: { 
            temperature: 0.1,
            num_ctx: 4096,
            num_gpu: 1 // Force GPU
          }
        })
      });
      
      const time = Date.now() - startTime;
      responseTime = time;
      
      if (response.ok) {
        const result = await response.json();
        systemStatus = 'operational';
        logAPI('Gemma3 Legal', 200, time);
        onresponse?.();
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      systemStatus = 'offline';
      logAPI('Gemma3 Legal', 0, Date.now() - startTime, error.message);
      onerror?.();
    }
    
    isProcessing = false;
    currentTest = null;
  };

  const testSynthesis = async () => {
    if (isProcessing) return;
    
    isProcessing = true;
    currentTest = 'synthesis';
    const startTime = Date.now();
    
    try {
      const response = await fetch('/api/evidence/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceIds: ['test-1', 'test-2'],
          synthesisType: 'correlation',
          caseId: 'api-test',
          title: 'API Validation Test',
          prompt: query
        })
      });
      
      const time = Date.now() - startTime;
      responseTime = time;
      
      // 401 is expected without auth
      if (response.status === 401 || response.ok) {
        systemStatus = 'operational';
        logAPI('Synthesis API', response.status, time);
        const result = await response.json();
        onresponse?.(),
          metadata: { time, status: response.status }
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      logAPI('Synthesis API', 0, Date.now() - startTime, error.message);
      onerror?.();
    }
    
    isProcessing = false;
    currentTest = null;
  };

  const testRAG = async () => {
    if (isProcessing) return;
    
    isProcessing = true;
    currentTest = 'rag';
    const startTime = Date.now();
    
    try {
      const response = await fetch('/api/enhanced-rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query || 'legal evidence analysis',
          useContextRAG: true,
          maxResults: 10
        })
      });
      
      const time = Date.now() - startTime;
      responseTime = time;
      
      if (response.ok) {
        const result = await response.json();
        logAPI('RAG Studio', 200, time);
        onresponse?.(),
          metadata: { time, documents: result.documents?.length }
        });
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      logAPI('RAG Studio', 0, Date.now() - startTime, error.message);
      onerror?.();
    }
    
    isProcessing = false;
    currentTest = null;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'operational': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      default: return 'bg-yellow-500';
    }
  };
</script>

<div class="space-y-4">
  <!-- Action Buttons -->
  <div class="grid grid-cols-3 gap-3">
    <Button.Root
      onclick={testGemma3}
      disabled={isProcessing}
      class="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
    >
      {#if currentTest === 'gemma3'}
        <Loader2 class="w-4 h-4 animate-spin" />
      {:else}
        <Brain class="w-4 h-4" />
      {/if}
      Gemma3
    </Button.Root>

    <Button.Root
      onclick={testSynthesis}
      disabled={isProcessing}
      class="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white"
    >
      {#if currentTest === 'synthesis'}
        <Loader2 class="w-4 h-4 animate-spin" />
      {:else}
        <Zap class="w-4 h-4" />
      {/if}
      Synthesis
    </Button.Root>

    <Button.Root
      onclick={testRAG}
      disabled={isProcessing}
      class="flex items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white"
    >
      {#if currentTest === 'rag'}
        <Loader2 class="w-4 h-4 animate-spin" />
      {:else}
        <AlertTriangle class="w-4 h-4" />
      {/if}
      RAG
    </Button.Root>
  </div>

  <!-- Status & Metrics -->
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      <div class="w-3 h-3 rounded-full {getStatusColor(systemStatus)}"></div>
      <span class="text-sm font-medium capitalize">{systemStatus}</span>
    </div>
    
    {#if responseTime > 0}
      <Badge.Root class="bg-slate-100 text-slate-700">
        {responseTime}ms
      </Badge.Root>
    {/if}
  </div>

  <!-- API Logs -->
  {#if apiLogs.length > 0}
    <div class="bg-black text-green-400 p-3 rounded-lg max-h-32 overflow-y-auto font-mono text-xs">
      {#each apiLogs.slice(0, 5) as log}
        <div class="mb-1">
          <span class="opacity-60">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
          <span class="ml-2 {log.error ? 'text-red-400' : 'text-green-400'}">
            {log.endpoint}: {log.status} ({log.time}ms)
          </span>
          {#if log.error}
            <span class="ml-2 text-red-300">- {log.error}</span>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>
