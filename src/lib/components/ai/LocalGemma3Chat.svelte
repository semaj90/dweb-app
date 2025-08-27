<script lang="ts">
    // High-performance local Gemma3 chat with WebAssembly inference
    // Integrates with existing OCR, AI service workers, and synthesis orchestrator
    
    import { onMount, onDestroy } from 'svelte';
    import { writable, derived } from 'svelte/store';
    import { gemma3Service, type LegalAnalysisResult } from '$lib/services/gemma3-local-service';
    import type { Gemma3GenerationResult } from '$lib/wasm/gemma3-inference.d.ts';
    import Button from '$lib/components/ui/Button.svelte';
    import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
    
    // Props
    export let initialPrompt = '';
    export let analysisMode: 'chat' | 'document-analysis' | 'contract-review' = 'chat';
    export let enableStreaming = true;
    export let showPerformanceStats = true;
    
    // Reactive state
    let initialized = false;
    let loading = false;
    let error = '';
    
    // Chat state
    const messages = writable<Array<{
        id: string;
        role: 'user' | 'assistant' | 'system';
        content: string;
        timestamp: number;
        metadata?: {
            processingTime?: number;
            method?: string;
            tokensPerSecond?: number;
            confidence?: number;
        };
    }>>([]);
    
    const currentInput = writable('');
    const isGenerating = writable(false);
    const streamingText = writable('');
    
    // Performance metrics
    const performanceStats = writable({
        initialized: false,
        wasmAvailable: false,
        modelLoaded: false,
        ollamaHealthy: false,
        requests: { total: 0, webAssembly: 0, ollama: 0, cacheHits: 0 },
        performance: { averageResponseTime: 0, wasmTokensPerSecond: 0, memoryUsage: 0 },
        configuration: { webGPUEnabled: false, threadingEnabled: false, cacheSize: 0 }
    });
    
    // Document analysis state
    const analysisResult = writable<LegalAnalysisResult | null>(null);
    const analysisLoading = writable(false);
    
    // Derived stores
    const canSend = derived(
        [currentInput, isGenerating, initialized], 
        ([$currentInput, $isGenerating, $initialized]) => 
            $currentInput.trim() && !$isGenerating && $initialized
    );
    
    const totalMessages = derived(messages, $messages => $messages.length);
    
    // Component lifecycle
    onMount(async () => {
        await initializeService();
        if (initialPrompt) {
            currentInput.set(initialPrompt);
        }
        
        // Add welcome message based on analysis mode
        if (analysisMode === 'chat') {
            addSystemMessage('Local Gemma3 Legal AI initialized. Ask me anything about legal documents, contracts, or compliance.');
        } else {
            addSystemMessage(`${analysisMode === 'document-analysis' ? 'Document Analysis' : 'Contract Review'} mode active. Upload or paste content for AI-powered legal analysis.`);
        }
        
        updatePerformanceStats();
        
        // Set up periodic stats updates
        const statsInterval = setInterval(updatePerformanceStats, 5000);
        
        return () => {
            clearInterval(statsInterval);
        };
    });
    
    onDestroy(() => {
        gemma3Service.dispose();
    });
    
    // Service initialization
    async function initializeService(): Promise<void> {
        loading = true;
        error = '';
        
        try {
            const success = await gemma3Service.initialize();
            if (success) {
                initialized = true;
                console.log('[LocalGemma3Chat] Service initialized successfully');
            } else {
                throw new Error('Failed to initialize Gemma3 service');
            }
        } catch (err) {
            error = err instanceof Error ? err.message : 'Initialization failed';
            console.error('[LocalGemma3Chat] Initialization error:', err);
        } finally {
            loading = false;
        }
    }
    
    // Message handling
    async function sendMessage(): Promise<void> {
        const input = $currentInput.trim();
        if (!input || $isGenerating || !initialized) return;
        
        const userMessage = {
            id: crypto.randomUUID(),
            role: 'user' as const,
            content: input,
            timestamp: Date.now()
        };
        
        messages.update(msgs => [...msgs, userMessage]);
        currentInput.set('');
        isGenerating.set(true);
        streamingText.set('');
        
        try {
            if (enableStreaming) {
                await handleStreamingResponse(input);
            } else {
                await handleDirectResponse(input);
            }
        } catch (err) {
            console.error('[LocalGemma3Chat] Generation error:', err);
            addErrorMessage(err instanceof Error ? err.message : 'Generation failed');
        } finally {
            isGenerating.set(false);
            streamingText.set('');
            updatePerformanceStats();
        }
    }
    
    async function handleStreamingResponse(input: string): Promise<void> {
        const assistantMessage = {
            id: crypto.randomUUID(),
            role: 'assistant' as const,
            content: '',
            timestamp: Date.now(),
            metadata: {}
        };
        
        messages.update(msgs => [...msgs, assistantMessage]);
        
        const startTime = performance.now();
        let tokenCount = 0;
        
        try {
            for await (const chunk of gemma3Service.generateStream(input, {
                max_tokens: analysisMode === 'chat' ? 1024 : 2048,
                temperature: 0.1,
                use_cache: true
            })) {
                streamingText.set(chunk.text);
                tokenCount = Math.ceil(chunk.text.length / 4);
                
                // Update the message in real-time
                messages.update(msgs => {
                    const lastMsg = msgs[msgs.length - 1];
                    if (lastMsg && lastMsg.role === 'assistant') {
                        lastMsg.content = chunk.text;
                        lastMsg.metadata = {
                            processingTime: performance.now() - startTime,
                            tokensPerSecond: tokenCount / ((performance.now() - startTime) / 1000)
                        };
                    }
                    return [...msgs];
                });
                
                if (chunk.done) break;
            }
        } catch (error) {
            console.error('[LocalGemma3Chat] Streaming error:', error);
            addErrorMessage('Streaming response failed');
        }
    }
    
    async function handleDirectResponse(input: string): Promise<void> {
        const result = await gemma3Service.generate(input, {
            max_tokens: analysisMode === 'chat' ? 1024 : 2048,
            temperature: 0.1,
            use_cache: true
        });
        
        if (result.success) {
            const assistantMessage = {
                id: crypto.randomUUID(),
                role: 'assistant' as const,
                content: result.text || '',
                timestamp: Date.now(),
                metadata: {
                    processingTime: result.processing_time_ms,
                    method: result.method,
                    tokensPerSecond: result.tokens_per_second
                }
            };
            
            messages.update(msgs => [...msgs, assistantMessage]);
        } else {
            addErrorMessage(result.error || 'Generation failed');
        }
    }
    
    // Document analysis
    async function analyzeDocument(title: string, content: string): Promise<void> {
        if (!initialized) return;
        
        analysisLoading.set(true);
        error = '';
        
        try {
            const result = await gemma3Service.analyzeDocument(
                title, 
                content, 
                analysisMode === 'contract-review' ? 'risk-focused' : 'comprehensive'
            );
            
            analysisResult.set(result);
            
            // Add analysis summary as a message
            const analysisMessage = {
                id: crypto.randomUUID(),
                role: 'assistant' as const,
                content: formatAnalysisForChat(result),
                timestamp: Date.now(),
                metadata: {
                    processingTime: result.processingTime,
                    method: result.method,
                    confidence: result.confidence
                }
            };
            
            messages.update(msgs => [...msgs, analysisMessage]);
            
        } catch (err) {
            error = err instanceof Error ? err.message : 'Analysis failed';
            console.error('[LocalGemma3Chat] Analysis error:', err);
        } finally {
            analysisLoading.set(false);
            updatePerformanceStats();
        }
    }
    
    // File upload handling
    async function handleFileUpload(event: Event): Promise<void> {
        const input = event.target as HTMLInputElement;
        const file = input.files?.[0];
        
        if (!file) return;
        
        try {
            const content = await readFileContent(file);
            await analyzeDocument(file.name, content);
        } catch (err) {
            console.error('[LocalGemma3Chat] File upload error:', err);
            error = 'Failed to process uploaded file';
        }
    }
    
    async function readFileContent(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }
    
    // Utility functions
    function addSystemMessage(content: string): void {
        const systemMessage = {
            id: crypto.randomUUID(),
            role: 'system' as const,
            content,
            timestamp: Date.now()
        };
        
        messages.update(msgs => [...msgs, systemMessage]);
    }
    
    function addErrorMessage(content: string): void {
        const errorMessage = {
            id: crypto.randomUUID(),
            role: 'system' as const,
            content: `❌ Error: ${content}`,
            timestamp: Date.now()
        };
        
        messages.update(msgs => [...msgs, errorMessage]);
    }
    
    function updatePerformanceStats(): void {
        const stats = gemma3Service.getServiceStats();
        performanceStats.set(stats);
    }
    
    function formatAnalysisForChat(analysis: LegalAnalysisResult): string {
        return `## Document Analysis Results

**Summary:**
${analysis.summary}

**Key Terms:** ${analysis.keyTerms.join(', ')}

**Identified Entities:**
${analysis.entities.map(e => `• ${e.type}: ${e.value} (${Math.round(e.confidence * 100)}%)`).join('\n')}

**Risk Assessment:**
${analysis.risks.map(r => `• ${r.type} (${r.severity}): ${r.description}`).join('\n')}

**Recommendations:**
${analysis.recommendations.map(rec => `• ${rec}`).join('\n')}

**Confidence:** ${Math.round(analysis.confidence * 100)}% | **Processing Time:** ${Math.round(analysis.processingTime)}ms | **Method:** ${analysis.method}`;
    }
    
    function formatTimestamp(timestamp: number): string {
        return new Date(timestamp).toLocaleTimeString();
    }
    
    function clearChat(): void {
        messages.set([]);
        analysisResult.set(null);
        error = '';
    }
    
    // Keyboard shortcuts
    function handleKeydown(event: KeyboardEvent): void {
        if (event.key === 'Enter' && !event.shiftKey && $canSend) {
            event.preventDefault();
            sendMessage();
        }
    }
</script>

<!-- Chat Interface -->
<div class="gemma3-chat-container">
    <!-- Header -->
    <Card class="mb-4">
        <CardHeader class="pb-2">
            <CardTitle class="flex items-center justify-between">
                <span class="flex items-center gap-2">
                    🧠 Local Gemma3 Legal AI
                    {#if $performanceStats.wasmAvailable}
                        <span class="badge-webassembly">WebAssembly</span>
                    {:else}
                        <span class="badge-ollama">Ollama</span>
                    {/if}
                </span>
                
                <div class="flex items-center gap-2">
                    {#if analysisMode === 'document-analysis' || analysisMode === 'contract-review'}
                        <label class="btn-file-upload">
                            📄 Upload Document
                            <input 
                                type="file" 
                                accept=".txt,.pdf,.doc,.docx" 
                                on:change={handleFileUpload}
                                hidden
                            />
                        </label>
                    {/if}
                    
                    <Button variant="outline" size="sm" on:click={clearChat}>
                        Clear
                    </Button>
                </div>
            </CardTitle>
        </CardHeader>
        
        {#if showPerformanceStats}
            <CardContent class="pt-0">
                <div class="performance-stats">
                    <div class="stat-item">
                        <span class="stat-label">Status:</span>
                        <span class="stat-value {initialized ? 'text-green-600' : 'text-red-600'}">
                            {initialized ? 'Ready' : 'Initializing...'}
                        </span>
                    </div>
                    
                    <div class="stat-item">
                        <span class="stat-label">Requests:</span>
                        <span class="stat-value">{$performanceStats.requests.total}</span>
                    </div>
                    
                    <div class="stat-item">
                        <span class="stat-label">Avg Response:</span>
                        <span class="stat-value">{$performanceStats.performance.averageResponseTime}ms</span>
                    </div>
                    
                    {#if $performanceStats.wasmAvailable}
                        <div class="stat-item">
                            <span class="stat-label">Tokens/sec:</span>
                            <span class="stat-value">{Math.round($performanceStats.performance.wasmTokensPerSecond)}</span>
                        </div>
                        
                        <div class="stat-item">
                            <span class="stat-label">Memory:</span>
                            <span class="stat-value">{Math.round($performanceStats.performance.memoryUsage)}MB</span>
                        </div>
                    {/if}
                </div>
            </CardContent>
        {/if}
    </Card>

    <!-- Error Display -->
    {#if error}
        <Card class="mb-4 border-red-200 bg-red-50">
            <CardContent class="text-red-700">
                ⚠️ {error}
            </CardContent>
        </Card>
    {/if}

    <!-- Loading State -->
    {#if loading}
        <Card class="mb-4">
            <CardContent class="text-center py-8">
                <div class="loading-spinner"></div>
                <p class="mt-2 text-gray-600">Initializing Gemma3 inference engine...</p>
            </CardContent>
        </Card>
    {/if}

    <!-- Chat Messages -->
    <Card class="chat-messages-container mb-4">
        <CardContent class="p-0">
            <div class="messages-scroll" class:loading={$isGenerating}>
                {#each $messages as message (message.id)}
                    <div class="message message-{message.role}">
                        <div class="message-header">
                            <span class="message-role">
                                {message.role === 'user' ? '👤' : message.role === 'assistant' ? '🤖' : 'ℹ️'}
                                {message.role}
                            </span>
                            <span class="message-timestamp">{formatTimestamp(message.timestamp)}</span>
                        </div>
                        
                        <div class="message-content">
                            {#if message.role === 'assistant' && message.id === $messages[$messages.length - 1]?.id && $streamingText}
                                {$streamingText}
                                <span class="streaming-cursor">|</span>
                            {:else}
                                {@html message.content.replace(/\n/g, '<br>')}
                            {/if}
                        </div>
                        
                        {#if message.metadata}
                            <div class="message-metadata">
                                {#if message.metadata.processingTime}
                                    <span>⏱️ {Math.round(message.metadata.processingTime)}ms</span>
                                {/if}
                                {#if message.metadata.tokensPerSecond}
                                    <span>🚀 {Math.round(message.metadata.tokensPerSecond)} tok/s</span>
                                {/if}
                                {#if message.metadata.method}
                                    <span>🔧 {message.metadata.method}</span>
                                {/if}
                                {#if message.metadata.confidence}
                                    <span>📊 {Math.round(message.metadata.confidence * 100)}%</span>
                                {/if}
                            </div>
                        {/if}
                    </div>
                {/each}
                
                {#if $isGenerating && !$streamingText}
                    <div class="message message-assistant">
                        <div class="message-header">
                            <span class="message-role">🤖 assistant</span>
                            <span class="message-timestamp">{formatTimestamp(Date.now())}</span>
                        </div>
                        <div class="message-content">
                            <div class="thinking-indicator">
                                <span class="dot"></span>
                                <span class="dot"></span>
                                <span class="dot"></span>
                            </div>
                        </div>
                    </div>
                {/if}
            </div>
        </CardContent>
    </Card>

    <!-- Input Area -->
    <Card>
        <CardContent class="p-4">
            <div class="input-container">
                <textarea
                    bind:value={$currentInput}
                    on:keydown={handleKeydown}
                    placeholder={analysisMode === 'chat' 
                        ? "Ask about legal documents, contracts, or compliance..." 
                        : "Paste document content for analysis or upload a file..."}
                    rows="3"
                    class="chat-input"
                    disabled={!initialized || $isGenerating}
                ></textarea>
                
                <div class="input-actions">
                    <div class="input-info">
                        {#if $currentInput}
                            <span class="text-sm text-gray-500">
                                {$currentInput.length} chars
                            </span>
                        {/if}
                    </div>
                    
                    <Button 
                        on:click={sendMessage}
                        disabled={!$canSend}
                        variant={$canSend ? "default" : "outline"}
                    >
                        {#if $isGenerating}
                            <div class="loading-spinner-small"></div>
                            Generating...
                        {:else}
                            Send
                        {/if}
                    </Button>
                </div>
            </div>
        </CardContent>
    </Card>
</div>

<style>
    .gemma3-chat-container {
        max-width: 1200px;
        margin: 0 auto;
        height: 100vh;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }

    .badge-webassembly {
        background: linear-gradient(45deg, #4338ca, #7c3aed);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-ollama {
        background: linear-gradient(45deg, #059669, #0891b2);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.375rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .performance-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.875rem;
    }

    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .stat-label {
        font-weight: 500;
        color: #6b7280;
    }

    .stat-value {
        font-weight: 600;
        color: #111827;
    }

    .chat-messages-container {
        flex: 1;
        overflow: hidden;
    }

    .messages-scroll {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: #f9fafb;
    }

    .messages-scroll.loading {
        opacity: 0.8;
    }

    .message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background: white;
        border: 1px solid #e5e7eb;
    }

    .message-user {
        margin-left: 2rem;
        border-color: #3b82f6;
        background: #eff6ff;
    }

    .message-assistant {
        margin-right: 2rem;
        border-color: #10b981;
        background: #ecfdf5;
    }

    .message-system {
        margin: 0 1rem;
        border-color: #f59e0b;
        background: #fffbeb;
        font-style: italic;
    }

    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }

    .message-role {
        font-weight: 600;
        color: #374151;
    }

    .message-timestamp {
        color: #9ca3af;
        font-size: 0.75rem;
    }

    .message-content {
        line-height: 1.6;
        color: #111827;
        white-space: pre-wrap;
    }

    .message-metadata {
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e5e7eb;
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: #6b7280;
    }

    .streaming-cursor {
        animation: blink 1s infinite;
        color: #3b82f6;
        font-weight: bold;
    }

    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    .thinking-indicator {
        display: flex;
        gap: 0.25rem;
        align-items: center;
    }

    .thinking-indicator .dot {
        width: 0.5rem;
        height: 0.5rem;
        background: #9ca3af;
        border-radius: 50%;
        animation: thinking 1.5s infinite ease-in-out;
    }

    .thinking-indicator .dot:nth-child(1) { animation-delay: 0s; }
    .thinking-indicator .dot:nth-child(2) { animation-delay: 0.2s; }
    .thinking-indicator .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes thinking {
        0%, 80%, 100% { opacity: 0.3; transform: scale(1); }
        40% { opacity: 1; transform: scale(1.2); }
    }

    .input-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .chat-input {
        width: 100%;
        padding: 0.75rem;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        resize: vertical;
        font-family: inherit;
        font-size: 0.875rem;
        line-height: 1.5;
    }

    .chat-input:focus {
        outline: none;
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    .chat-input:disabled {
        background-color: #f3f4f6;
        opacity: 0.6;
    }

    .input-actions {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .input-info {
        font-size: 0.75rem;
        color: #6b7280;
    }

    .btn-file-upload {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        transition: all 0.2s ease;
    }

    .btn-file-upload:hover {
        background: #e5e7eb;
        border-color: #9ca3af;
    }

    .loading-spinner {
        width: 2rem;
        height: 2rem;
        border: 2px solid #e5e7eb;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }

    .loading-spinner-small {
        width: 1rem;
        height: 1rem;
        border: 2px solid #e5e7eb;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .gemma3-chat-container {
            padding: 0.5rem;
            height: 100vh;
        }

        .performance-stats {
            flex-direction: column;
            gap: 0.5rem;
        }

        .messages-scroll {
            height: 300px;
        }

        .message {
            margin-left: 0.5rem;
            margin-right: 0.5rem;
        }

        .message-user {
            margin-left: 1rem;
        }

        .message-assistant {
            margin-right: 1rem;
        }
    }
</style>