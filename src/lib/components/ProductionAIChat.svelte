<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import type { ComponentProps } from 'svelte';
  
  interface ChatMessage {
    id: string;
    content: string;
    sender: 'user' | 'ai';
    timestamp: number;
    metadata?: {
      processingTime?: number;
      model?: string;
      workerId?: string;
      contextUsed?: number;
      tokensUsed?: number;
    };
  }

  interface SystemHealth {
    status: string;
    workerId: string;
    connections: number;
    metrics: {
      requestsProcessed: number;
      averageResponseTime: number;
      errorRate: number;
      workerPool: {
        totalWorkers: number;
        busyWorkers: number;
        queueLength: number;
        activeJobs: number;
      };
    };
    multiCore: {
      enabled: boolean;
      totalWorkers: number;
      currentWorker: string;
    };
    services: {
      ollama: string;
      goService: string;
      svelteKit: string;
    };
  }

  // Configuration following Context7 best practices
  const ORCHESTRATOR_URL = 'http://localhost:40000';
  const WEBSOCKET_URL = 'ws://localhost:40000';

  // Svelte 5 reactive state
  let messages = $state<ChatMessage[]>([]);
  let currentMessage = $state('');
  let isLoading = $state(false);
  let isConnected = $state(false);
  let systemHealth = $state<SystemHealth | null>(null);
  let selectedModel = $state('gemma3-legal:latest');
  let useRAG = $state(true);
  let websocket = $state<WebSocket | null>(null);
  let userId = $state(`user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  let sessionId = $state(`session_${Date.now()}`);

  // Connection management
  let reconnectAttempts = $state(0);
  let maxReconnectAttempts = $state(5);
  let reconnectDelay = $state(1000);

  // Performance monitoring
  let lastResponseTime = $state(0);
  let totalRequests = $state(0);
  let successfulRequests = $state(0);

  onMount(() => {
    checkSystemHealth();
    initializeWebSocket();
    
    // Health check interval
    const healthInterval = setInterval(checkSystemHealth, 30000);
    
    return () => {
      if (websocket) {
        websocket.close();
      }
      clearInterval(healthInterval);
    };
  });

  async function checkSystemHealth() {
    try {
      const response = await fetch(`${ORCHESTRATOR_URL}/health`);
      if (response.ok) {
        systemHealth = await response.json();
      }
    } catch (error) {
      console.error('Health check failed:', error);
      systemHealth = null;
    }
  }

  function initializeWebSocket() {
    try {
      websocket = new WebSocket(WEBSOCKET_URL);
      
      websocket.onopen = () => {
        console.log('🔗 Connected to Production AI Orchestrator');
        isConnected = true;
        reconnectAttempts = 0;
      };
      
      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('WebSocket message parse error:', error);
        }
      };
      
      websocket.onclose = () => {
        console.log('🔌 WebSocket connection closed');
        isConnected = false;
        attemptReconnection();
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnected = false;
      };
    } catch (error) {
      console.error('WebSocket initialization failed:', error);
      isConnected = false;
    }
  }

  function attemptReconnection() {
    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts++;
      console.log(`🔄 Attempting reconnection ${reconnectAttempts}/${maxReconnectAttempts}...`);
      
      setTimeout(() => {
        initializeWebSocket();
      }, reconnectDelay * reconnectAttempts);
    } else {
      console.error('❌ Max reconnection attempts reached');
    }
  }

  function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'welcome':
        console.log(`🎯 Connected to worker ${data.workerId}`);
        break;
      case 'chat_response':
        handleChatResponse(data);
        break;
      case 'health_update':
        systemHealth = data.status;
        break;
      case 'error':
        console.error('WebSocket error:', data.error);
        break;
    }
  }

  function handleChatResponse(data: any) {
    isLoading = false;
    lastResponseTime = data.processingTime || 0;
    successfulRequests++;
    
    const aiMessage: ChatMessage = {
      id: data.chatId || Date.now().toString(),
      content: data.response?.response || data.response || 'No response received',
      sender: 'ai',
      timestamp: Date.now(),
      metadata: {
        processingTime: data.processingTime,
        workerId: data.workerId,
        model: selectedModel,
        contextUsed: data.response?.contextUsed,
        tokensUsed: data.response?.tokensUsed
      }
    };
    
    messages = [...messages, aiMessage];
  }

  async function sendMessage() {
    if (!currentMessage.trim()) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: currentMessage.trim(),
      sender: 'user',
      timestamp: Date.now()
    };
    
    messages = [...messages, userMessage];
    const messageToSend = currentMessage.trim();
    currentMessage = '';
    isLoading = true;
    totalRequests++;
    
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Send via WebSocket for real-time response
        websocket.send(JSON.stringify({
          type: 'chat',
          data: {
            message: messageToSend,
            userId,
            sessionId,
            model: selectedModel,
            useRAG
          }
        }));
      } else {
        // Fallback to REST API
        const response = await fetch(`${ORCHESTRATOR_URL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            message: messageToSend,
            userId,
            sessionId,
            model: selectedModel,
            useRAG
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          const aiMessage: ChatMessage = {
            id: data.chatId || Date.now().toString(),
            content: data.response || 'Response received',
            sender: 'ai',
            timestamp: Date.now(),
            metadata: data.metadata
          };
          
          messages = [...messages, aiMessage];
          lastResponseTime = data.metadata?.processingTime || 0;
          successfulRequests++;
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        isLoading = false;
      }
    } catch (error) {
      console.error('Send message error:', error);
      isLoading = false;
      
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: 'Failed to send message. Please check your connection and try again.',
        sender: 'ai',
        timestamp: Date.now()
      };
      
      messages = [...messages, errorMessage];
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function formatTimestamp(timestamp: number): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  function formatResponseTime(ms: number): string {
    return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  }

  function getSuccessRate(): string {
    if (totalRequests === 0) return '100%';
    return `${Math.round((successfulRequests / totalRequests) * 100)}%`;
  }

  function clearChat() {
    messages = [];
    sessionId = `session_${Date.now()}`;
  }

  // Available models
  const availableModels = [
    'gemma3-legal:latest',
    'gemma3:latest', 
    'llama3.2:latest',
    'nomic-embed-text'
  ];
</script>

<div class="production-ai-chat">
  <!-- Header with system status -->
  <div class="chat-header">
    <div class="header-left">
      <h2>🎯 Production Legal AI Assistant</h2>
      <div class="connection-status" class:connected={isConnected} class:disconnected={!isConnected}>
        {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        {#if systemHealth}
          <span class="worker-info">Worker: {systemHealth.workerId}</span>
        {/if}
      </div>
    </div>
    
    <div class="header-right">
      {#if systemHealth}
        <div class="system-metrics">
          <div class="metric">
            <span class="metric-label">Workers:</span>
            <span class="metric-value">{systemHealth.metrics.workerPool.busyWorkers}/{systemHealth.metrics.workerPool.totalWorkers}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Queue:</span>
            <span class="metric-value">{systemHealth.metrics.workerPool.queueLength}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Avg Response:</span>
            <span class="metric-value">{Math.round(systemHealth.metrics.averageResponseTime)}ms</span>
          </div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Configuration panel -->
  <div class="config-panel">
    <div class="config-item">
      <label for="model-select">AI Model:</label>
      <select id="model-select" bind:value={selectedModel}>
        {#each availableModels as model}
          <option value={model}>{model}</option>
        {/each}
      </select>
    </div>
    
    <div class="config-item">
      <label>
        <input type="checkbox" bind:checked={useRAG} />
        Use RAG (Retrieval Augmented Generation)
      </label>
    </div>
    
    <div class="config-item">
      <button onclick={clearChat} class="clear-button">Clear Chat</button>
    </div>
    
    <div class="performance-stats">
      <span>Success Rate: {getSuccessRate()}</span>
      <span>Last Response: {formatResponseTime(lastResponseTime)}</span>
      <span>Total Requests: {totalRequests}</span>
    </div>
  </div>

  <!-- Messages container -->
  <div class="messages-container">
    {#each messages as message (message.id)}
      <div class="message" class:user-message={message.sender === 'user'} class:ai-message={message.sender === 'ai'}>
        <div class="message-header">
          <span class="sender">
            {message.sender === 'user' ? '👤 You' : '🤖 Legal AI'}
          </span>
          <span class="timestamp">{formatTimestamp(message.timestamp)}</span>
        </div>
        
        <div class="message-content">
          {message.content}
        </div>
        
        {#if message.metadata}
          <div class="message-metadata">
            {#if message.metadata.processingTime}
              <span class="meta-item">⏱️ {formatResponseTime(message.metadata.processingTime)}</span>
            {/if}
            {#if message.metadata.workerId}
              <span class="meta-item">🔧 Worker: {message.metadata.workerId}</span>
            {/if}
            {#if message.metadata.contextUsed}
              <span class="meta-item">📚 Context: {message.metadata.contextUsed} docs</span>
            {/if}
            {#if message.metadata.tokensUsed}
              <span class="meta-item">🔤 Tokens: {message.metadata.tokensUsed}</span>
            {/if}
          </div>
        {/if}
      </div>
    {/each}
    
    {#if isLoading}
      <div class="message ai-message loading">
        <div class="message-header">
          <span class="sender">🤖 Legal AI</span>
          <span class="timestamp">Processing...</span>
        </div>
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
          Analyzing your legal question with {selectedModel}...
        </div>
      </div>
    {/if}
  </div>

  <!-- Input area -->
  <div class="input-area">
    <div class="input-container">
      <textarea
        bind:value={currentMessage}
        onkeypress={handleKeyPress}
        placeholder="Ask your legal question here..."
        rows="3"
        disabled={!isConnected}
      ></textarea>
      
      <button 
        onclick={sendMessage}
        disabled={!currentMessage.trim() || isLoading || !isConnected}
        class="send-button"
      >
        {isLoading ? '⏳' : '🚀'} Send
      </button>
    </div>
  </div>
</div>

<style>
  .production-ai-chat {
    display: flex;
    flex-direction: column;
    height: 700px;
    max-width: 1200px;
    margin: 0 auto;
    border: 2px solid #2d3748;
    border-radius: 12px;
    overflow: hidden;
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .chat-header {
    background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #4a5568;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
  }

  .header-left h2 {
    margin: 0 0 0.5rem 0;
    font-size: 1.4rem;
    font-weight: 700;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 500;
  }

  .connection-status.connected {
    color: #68d391;
  }

  .connection-status.disconnected {
    color: #fc8181;
  }

  .worker-info {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
  }

  .system-metrics {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
  }

  .metric {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .metric-label {
    color: #a0aec0;
    font-size: 0.7rem;
  }

  .metric-value {
    color: white;
    font-weight: 600;
  }

  .config-panel {
    background: #2d3748;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #4a5568;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
    color: white;
  }

  .config-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
  }

  .config-item label {
    font-weight: 500;
    white-space: nowrap;
  }

  .config-item select {
    background: #4a5568;
    color: white;
    border: 1px solid #6b7280;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
  }

  .clear-button {
    background: #e53e3e;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    cursor: pointer;
    transition: background 0.2s;
  }

  .clear-button:hover {
    background: #c53030;
  }

  .performance-stats {
    margin-left: auto;
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
    color: #a0aec0;
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    background: #1a202c;
  }

  .message {
    margin-bottom: 1.5rem;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in;
  }

  .message.user-message {
    margin-left: auto;
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    color: white;
  }

  .message.ai-message {
    margin-right: auto;
    background: #2d3748;
    border: 1px solid #4a5568;
    color: #e2e8f0;
  }

  .message.loading {
    animation: pulse 2s ease-in-out infinite;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
  }

  .message-content {
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .message-metadata {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
  }

  .meta-item {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    color: #a0aec0;
  }

  .typing-indicator {
    display: inline-flex;
    gap: 0.3rem;
    margin-right: 0.75rem;
    align-items: center;
  }

  .typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4299e1;
    animation: typing 1.5s ease-in-out infinite;
  }

  .typing-indicator span:nth-child(1) { animation-delay: 0s; }
  .typing-indicator span:nth-child(2) { animation-delay: 0.5s; }
  .typing-indicator span:nth-child(3) { animation-delay: 1s; }

  .input-area {
    background: #2d3748;
    padding: 1.5rem;
    border-top: 1px solid #4a5568;
  }

  .input-container {
    display: flex;
    gap: 1rem;
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    background: #1a202c;
    color: white;
    border: 1px solid #4a5568;
    border-radius: 8px;
    padding: 1rem;
    resize: none;
    font-family: inherit;
    font-size: 0.95rem;
    line-height: 1.5;
  }

  textarea:focus {
    outline: none;
    border-color: #4299e1;
    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
  }

  textarea:disabled {
    background: #374151;
    color: #9ca3af;
  }

  .send-button {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 1rem 2rem;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    white-space: nowrap;
  }

  .send-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(66, 153, 225, 0.3);
  }

  .send-button:disabled {
    background: #6b7280;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
  }

  /* Scrollbar styling */
  .messages-container::-webkit-scrollbar {
    width: 8px;
  }

  .messages-container::-webkit-scrollbar-track {
    background: #1a202c;
    border-radius: 4px;
  }

  .messages-container::-webkit-scrollbar-thumb {
    background: #4a5568;
    border-radius: 4px;
  }

  .messages-container::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .chat-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .config-panel {
      flex-direction: column;
      gap: 1rem;
    }

    .performance-stats {
      margin-left: 0;
      flex-direction: column;
      gap: 0.5rem;
    }

    .system-metrics {
      justify-content: center;
    }

    .input-container {
      flex-direction: column;
    }

    .send-button {
      align-self: flex-end;
    }
  }
</style>