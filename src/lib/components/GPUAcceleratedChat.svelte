<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  
  interface ChatMessage {
    id: string;
    content: string;
    sender: 'user' | 'ai';
    timestamp: number;
    metadata?: {
      processingTime?: number;
      model?: string;
      gpuAccelerated?: boolean;
      tokens?: number;
    };
  }

  interface OrchestatorHealth {
    status: string;
    activeRequests: number;
    gpuEnabled: boolean;
    uptime: number;
    services: {
      redis: boolean;
      database: boolean;
      ollama: boolean;
      cuda: string | boolean;
    };
  }

  // Stores
  const messages = writable<ChatMessage[]>([]);
  const isLoading = writable(false);
  const health = writable<OrchestatorHealth | null>(null);
  const connectionStatus = writable<'connected' | 'disconnected' | 'connecting'>('disconnected');

  // Configuration
  const ORCHESTRATOR_URL = 'http://localhost:4001';
  const WEBSOCKET_URL = 'ws://localhost:4001';

  let messageInput = '';
  let websocket: WebSocket | null = null;
  let userId = 'user_' + Math.random().toString(36).substr(2, 9);
  let sessionId = 'session_' + Date.now();

  // Initialize connection
  onMount(() => {
    checkOrchestratorHealth();
    initializeWebSocket();
    
    // Health check interval
    const healthInterval = setInterval(checkOrchestratorHealth, 30000);
    
    return () => {
      if (websocket) {
        websocket.close();
      }
      clearInterval(healthInterval);
    };
  });

  async function checkOrchestratorHealth() {
    try {
      const response = await fetch(`${ORCHESTRATOR_URL}/health`);
      if (response.ok) {
        const healthData = await response.json();
        health.set(healthData);
      }
    } catch (error) {
      console.error('Health check failed:', error);
      health.set(null);
    }
  }

  function initializeWebSocket() {
    connectionStatus.set('connecting');
    
    try {
      // Add a small delay to ensure the orchestrator is ready
      setTimeout(() => {
        websocket = new WebSocket(WEBSOCKET_URL);
        
        websocket.onopen = () => {
          console.log('🔗 WebSocket connected to GPU orchestrator');
          connectionStatus.set('connected');
        };
        
        websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
          } catch (error) {
            console.error('WebSocket message parse error:', error);
          }
        };
        
        websocket.onclose = (event) => {
          console.log('🔌 WebSocket connection closed', event.code, event.reason);
          connectionStatus.set('disconnected');
          
          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            if (!websocket || websocket.readyState === WebSocket.CLOSED) {
              console.log('🔄 Attempting WebSocket reconnection...');
              initializeWebSocket();
            }
          }, 3000);
        };
        
        websocket.onerror = (error) => {
          console.error('WebSocket error:', error);
          connectionStatus.set('disconnected');
        };
        
        // Set a connection timeout
        setTimeout(() => {
          if (websocket && websocket.readyState === WebSocket.CONNECTING) {
            console.warn('WebSocket connection timeout, closing...');
            websocket.close();
          }
        }, 5000);
      }, 1000);
    } catch (error) {
      console.error('WebSocket initialization failed:', error);
      connectionStatus.set('disconnected');
    }
  }

  function handleWebSocketMessage(data: unknown) {
    if (data.type === 'response') {
      isLoading.set(false);
      
      const aiMessage: ChatMessage = {
        id: data.id || Date.now().toString(),
        content: data.response || 'Sorry, I encountered an error processing your request.',
        sender: 'ai',
        timestamp: data.timestamp || Date.now(),
        metadata: data.metadata
      };
      
      messages.update(msgs => [...msgs, aiMessage]);
    } else if (data.type === 'error') {
      isLoading.set(false);
      console.error('AI processing error:', data.error);
      
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'ai',
        timestamp: Date.now()
      };
      
      messages.update(msgs => [...msgs, errorMessage]);
    }
  }

  async function sendMessage() {
    if (!messageInput.trim()) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: messageInput.trim(),
      sender: 'user',
      timestamp: Date.now()
    };
    
    messages.update(msgs => [...msgs, userMessage]);
    
    const messageToSend = messageInput.trim();
    messageInput = '';
    isLoading.set(true);
    
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Send via WebSocket
        websocket.send(JSON.stringify({
          type: 'chat',
          content: messageToSend,
          userId,
          sessionId,
          id: userMessage.id,
          enableTTS: false
        }));
      } else {
        // Fallback to REST API
        const response = await fetch(`${ORCHESTRATOR_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            message: messageToSend,
            userId,
            sessionId,
            enableTTS: false
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          
          const aiMessage: ChatMessage = {
            id: Date.now().toString(),
            content: data.response || 'Response received',
            sender: 'ai',
            timestamp: data.timestamp || Date.now(),
            metadata: data.metadata
          };
          
          messages.update(msgs => [...msgs, aiMessage]);
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        isLoading.set(false);
      }
    } catch (error) {
      console.error('Send message error:', error);
      isLoading.set(false);
      
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: 'Failed to send message. Please check your connection and try again.',
        sender: 'ai',
        timestamp: Date.now()
      };
      
      messages.update(msgs => [...msgs, errorMessage]);
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

  function formatProcessingTime(ms: number): string {
    return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  }
</script>

<div class="gpu-chat-container">
  <div class="chat-header">
    <h2>🎯 GPU-Accelerated Legal AI Chat</h2>
    <div class="status-indicators">
      <div class="connection-status" class:connected={$connectionStatus === 'connected'} 
           class:connecting={$connectionStatus === 'connecting'} 
           class:disconnected={$connectionStatus === 'disconnected'}>
        {$connectionStatus === 'connected' ? '🟢' : $connectionStatus === 'connecting' ? '🟡' : '🔴'}
        {$connectionStatus}
      </div>
      
      {#if $health}
        <div class="health-status" class:healthy={$health.status === 'healthy'} 
             class:degraded={$health.status === 'degraded'}>
          <span class="health-indicator">
            {$health.status === 'healthy' ? '✅' : '⚠️'}
          </span>
          <span class="health-text">
            System: {$health.status}
          </span>
          <span class="service-status">
            Redis: {$health.services.redis ? '✅' : '❌'}
            DB: {$health.services.database ? '✅' : '❌'}
            Ollama: {$health.services.ollama ? '✅' : '❌'}
            GPU: {$health.gpuEnabled ? '✅' : '❌'}
          </span>
        </div>
      {/if}
    </div>
  </div>

  <div class="chat-messages">
    {#each $messages as message (message.id)}
      <div class="message" class:user={message.sender === 'user'} class:ai={message.sender === 'ai'}>
        <div class="message-header">
          <span class="sender">{message.sender === 'user' ? '👤 You' : '🤖 Legal AI'}</span>
          <span class="timestamp">{formatTimestamp(message.timestamp)}</span>
        </div>
        <div class="message-content">
          {message.content}
        </div>
        {#if message.metadata}
          <div class="message-metadata">
            {#if message.metadata.model}
              <span class="metadata-item">Model: {message.metadata.model}</span>
            {/if}
            {#if message.metadata.processingTime}
              <span class="metadata-item">Processing: {formatProcessingTime(message.metadata.processingTime)}</span>
            {/if}
            {#if message.metadata.tokens}
              <span class="metadata-item">Tokens: {message.metadata.tokens}</span>
            {/if}
            {#if message.metadata.gpuAccelerated !== undefined}
              <span class="metadata-item">GPU: {message.metadata.gpuAccelerated ? '✅' : '❌'}</span>
            {/if}
          </div>
        {/if}
      </div>
    {/each}
    
    {#if $isLoading}
      <div class="message ai loading">
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
          <span>Analyzing your legal question with Gemma3...</span>
        </div>
      </div>
    {/if}
  </div>

  <div class="chat-input">
    <div class="input-container">
      <textarea
        bind:value={messageInput}
        on:keypress={handleKeyPress}
        placeholder="Ask me any legal question..."
        rows="2"
        disabled={$connectionStatus === 'disconnected'}
      ></textarea>
      <button 
        onclick={sendMessage} 
        disabled={!messageInput.trim() || $isLoading || $connectionStatus === 'disconnected'}
        class="send-button"
      >
        {$isLoading ? '⏳' : '🚀'}
        Send
      </button>
    </div>
  </div>
</div>

<style>
  .gpu-chat-container {
    display: flex;
    flex-direction: column;
    height: 600px;
    max-width: 800px;
    margin: 0 auto;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
  }

  .chat-header {
    background: rgba(255, 255, 255, 0.95);
    padding: 1rem;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .chat-header h2 {
    margin: 0;
    color: #2d3748;
    font-size: 1.2rem;
    font-weight: 600;
  }

  .status-indicators {
    display: flex;
    gap: 1rem;
    align-items: center;
    font-size: 0.8rem;
  }

  .connection-status {
    padding: 0.25rem 0.5rem;
    border-radius: 20px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .connection-status.connected {
    background: #c6f6d5;
    color: #22543d;
  }

  .connection-status.connecting {
    background: #fefcbf;
    color: #744210;
  }

  .connection-status.disconnected {
    background: #fed7d7;
    color: #822727;
  }

  .health-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
  }

  .health-status.healthy {
    background: #c6f6d5;
    color: #22543d;
  }

  .health-status.degraded {
    background: #fefcbf;
    color: #744210;
  }

  .service-status {
    display: flex;
    gap: 0.25rem;
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.95);
  }

  .message {
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 12px;
    max-width: 85%;
  }

  .message.user {
    margin-left: auto;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }

  .message.ai {
    margin-right: auto;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    color: #2d3748;
  }

  .message.loading {
    animation: pulse 2s ease-in-out infinite;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    font-weight: 600;
  }

  .message-content {
    line-height: 1.5;
    white-space: pre-wrap;
  }

  .message-metadata {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    font-size: 0.7rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .metadata-item {
    background: rgba(0, 0, 0, 0.1);
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
  }

  .typing-indicator {
    display: inline-flex;
    gap: 0.2rem;
    margin-right: 0.5rem;
  }

  .typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #667eea;
    animation: typing 1.5s ease-in-out infinite;
  }

  .typing-indicator span:nth-child(1) { animation-delay: 0s; }
  .typing-indicator span:nth-child(2) { animation-delay: 0.5s; }
  .typing-indicator span:nth-child(3) { animation-delay: 1s; }

  .chat-input {
    background: rgba(255, 255, 255, 0.95);
    padding: 1rem;
    border-top: 1px solid #e2e8f0;
  }

  .input-container {
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    resize: none;
    font-family: inherit;
    font-size: 0.9rem;
    line-height: 1.4;
  }

  textarea:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  textarea:disabled {
    background: #f7fafc;
    color: #a0aec0;
  }

  .send-button {
    padding: 0.75rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .send-button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }

  .send-button:disabled {
    background: #a0aec0;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Scrollbar styling */
  .chat-messages::-webkit-scrollbar {
    width: 6px;
  }

  .chat-messages::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }

  .chat-messages::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }

  .chat-messages::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
  }
</style>