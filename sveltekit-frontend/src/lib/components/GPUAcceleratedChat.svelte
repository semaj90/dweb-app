<script lang="ts">

  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import type { GPUChatMessage, GPUProcessingStatus } from '$lib/types/search';
  
  // Configuration - Dynamic port selection
  const PRIMARY_PORT = 5173;
  const FALLBACK_PORTS = [5174, 5175, 8080, 8081];
  let currentPort = $state(PRIMARY_PORT);
  let wsPort = $state(PRIMARY_PORT + 2); // WebSocket on +2 offset
  
  // State management
  let messages = $state<GPUChatMessage[]>([]);
  let inputMessage = $state('');
  let isConnected = $state(false);
  let isTyping = $state(false);
  let gpuStatus = $state<GPUProcessingStatus | null>(null);
let ws = $state<WebSocket | null >(null);
  let reconnectTimeout: NodeJS.Timeout;
  let healthCheckInterval: NodeJS.Timeout;
  let currentRoom = $state<string | null>(null);
  let uploadedFiles = $state<File[]>([]);
  let isSpeaking = $state(false);
  let batchMode = $state(false);
  let batchItems = $state<string[]>([]);
  
  // Voice/TTS configuration
  let voiceEnabled = $state(false);
  let selectedVoice = $state('neural');
  
  // Multi-user support
  let clientId = $state<string>('');
  let connectedUsers = $state<number>(0);
  
  // Initialize WebSocket with port detection
  async function connectWebSocket() {
    // Try to detect available port
    const availablePort = await detectAvailablePort();
    currentPort = availablePort;
    wsPort = availablePort + 2;
    
    const wsUrl = `ws://localhost:${wsPort}`;
    console.log(`Connecting to WebSocket at ${wsUrl}...`);
    
    try {
      ws = new WebSocket(wsUrl);
      
      ws.on:open=() => {
        console.log('✅ WebSocket connected on port', wsPort);
        isConnected = true;
        clearTimeout(reconnectTimeout);
        
        // Send handshake
        ws?.send(JSON.stringify({
          type: 'handshake',
          clientId: sessionStorage.getItem('clientId') || generateClientId()
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.on:close=() => {
        console.log('WebSocket disconnected');
        isConnected = false;
        
        // Attempt reconnection
        reconnectTimeout = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connectWebSocket();
        }, 3000);
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      isConnected = false;
    }
  }
  
  // Detect available port
  async function detectAvailablePort(): Promise<number> {
    // Try primary port first
    try {
      const response = await fetch(`http://localhost:${PRIMARY_PORT}/api/health`, {
        signal: AbortSignal.timeout(1000)
      });
      if (response.ok) {
        return PRIMARY_PORT;
      }
    } catch {}
    
    // Try fallback ports
    for (const port of FALLBACK_PORTS) {
      try {
        const response = await fetch(`http://localhost:${port}/api/health`, {
          signal: AbortSignal.timeout(1000)
        });
        if (response.ok) {
          console.log(`Using fallback port ${port}`);
          return port;
        }
      } catch {}
    }
    
    // Default to primary if all fail
    return PRIMARY_PORT;
  }
  
  // Generate client ID
  function generateClientId(): string {
    const id = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('clientId', id);
    return id;
  }
  
  // Handle WebSocket messages
  function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'connected':
        clientId = data.clientId;
        gpuStatus = data.gpuConfig;
        console.log('Connected with ID:', clientId);
        break;
        
      case 'chat_response':
        const message: GPUChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.response || data.content,
          timestamp: new Date(),
          metadata: data.metadata
        };
        messages = [...messages, message];
        isTyping = false;
        
        // TTS if enabled
        if (voiceEnabled && data.response) {
          speakText(data.response);
        }
        break;
        
      case 'typing':
        isTyping = data.isTyping;
        break;
        
      case 'user_joined':
        connectedUsers++;
        showNotification(`User joined room: ${data.clientId}`, 'info');
        break;
        
      case 'user_left':
        connectedUsers = Math.max(0, connectedUsers - 1);
        break;
        
      case 'document_processed':
        showNotification('Document processed successfully', 'success');
        handleDocumentResult(data);
        break;
        
      case 'batch_complete':
        handleBatchResults(data.results);
        batchMode = false;
        break;
        
      case 'error':
        console.error('Server error:', data.error);
        showNotification('Error: ' + data.error, 'error');
        isTyping = false;
        break;
    }
  }
  
  // Send message
  async function sendMessage() {
    if (!inputMessage.trim()) return;
    
    const userMessage: GPUChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };
    
    messages = [...messages, userMessage];
    const messageContent = inputMessage;
    inputMessage = '';
    isTyping = true;
    
    // Send via WebSocket if connected
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'chat',
        content: messageContent,
        room: currentRoom
      }));
    } else {
      // Fallback to HTTP API
      try {
        const response = await fetch(`http://localhost:${currentPort}/api/gpu-chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: messageContent,
            sessionId: sessionStorage.getItem('sessionId') || crypto.randomUUID()
          })
        });
        
        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        const aiMessage: GPUChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
          metadata: data.metadata
        };
        
        messages = [...messages, aiMessage];
        
        if (voiceEnabled && data.response) {
          speakText(data.response);
        }
      } catch (error) {
        console.error('Failed to send message:', error);
        showNotification('Failed to send message', 'error');
      } finally {
        isTyping = false;
      }
    }
  }
  
  // Handle document upload
  async function handleFileUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    
    const file = input.files[0];
    uploadedFiles = [...uploadedFiles, file];
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`http://localhost:${currentPort}/api/document/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Add system message about upload
        messages = [...messages, {
          id: crypto.randomUUID(),
          role: 'system',
          content: `Document "${file.name}" uploaded and processed. ${result.summary || ''}`,
          timestamp: new Date()
        }];
        
        // Send to WebSocket for processing
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'document_upload',
            document: {
              name: file.name,
              content: result.content,
              embeddings: result.embeddings
            }
          }));
        }
      }
    } catch (error) {
      console.error('Upload failed:', error);
      showNotification('Failed to upload document', 'error');
    }
  }
  
  // Text-to-Speech
  async function speakText(text: string) {
    if (!voiceEnabled || isSpeaking) return;
    
    isSpeaking = true;
    
    // Use Web Speech API as fallback
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.onend = () => {
        isSpeaking = false;
      };
      
      speechSynthesis.speak(utterance);
    } else if (ws && ws.readyState === WebSocket.OPEN) {
      // Request TTS from server
      ws.send(JSON.stringify({
        type: 'tts_request',
        text,
        voice: selectedVoice
      }));
    }
  }
  
  // Batch processing
  function addToBatch() {
    if (inputMessage.trim()) {
      batchItems = [...batchItems, inputMessage];
      inputMessage = '';
    }
  }
  
  function processBatch() {
    if (batchItems.length === 0) return;
    
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'batch',
        items: batchItems
      }));
      
      showNotification(`Processing ${batchItems.length} items in batch...`, 'info');
      batchItems = [];
    }
  }
  
  // Join/Leave room for multi-user
  function joinRoom(roomId: string) {
    currentRoom = roomId;
    
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'join_room',
        room: roomId
      }));
    }
  }
  
  function leaveRoom() {
    if (currentRoom && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'leave_room'
      }));
    }
    
    currentRoom = null;
    connectedUsers = 0;
  }
  
  // Handle document processing result
  function handleDocumentResult(data: any) {
    messages = [...messages, {
      id: crypto.randomUUID(),
      role: 'system',
      content: `Document analysis complete:\n${data.summary || 'Processing complete'}`,
      timestamp: new Date(),
      metadata: {
        documentId: data.documentId,
        embeddings: data.embeddings?.length || 0
      }
    }];
  }
  
  // Handle batch results
  function handleBatchResults(results: any[]) {
    const summary = `Batch processing complete:\n${results.map(r => `• ${r.summary || r.content}`).join('\n')}`;
    
    messages = [...messages, {
      id: crypto.randomUUID(),
      role: 'system',
      content: summary,
      timestamp: new Date(),
      metadata: {
        batchSize: results.length
      }
    }];
  }
  
  // Check GPU status
  async function checkGPUStatus() {
    try {
      const response = await fetch(`http://localhost:${currentPort}/api/gpu-status`);
      if (response.ok) {
        gpuStatus = await response.json();
      }
    } catch (error) {
      console.error('Failed to check GPU status:', error);
    }
  }
  
  // Health check
  async function performHealthCheck() {
    try {
      const response = await fetch(`http://localhost:${currentPort}/api/health`);
      const health = await response.json();
      
      if (!health.healthy) {
        showNotification('System health check failed', 'warning');
      }
    } catch (error) {
      console.error('Health check failed:', error);
    }
  }
  
  // Show notification
  function showNotification(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Add visual notification
    const notification = {
      id: crypto.randomUUID(),
      message,
      type,
      timestamp: new Date()
    };
    
    // You could store notifications in state and display them
  }
  
  // Handle keyboard shortcuts
  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    } else if (event.key === 'Enter' && event.shiftKey && batchMode) {
      event.preventDefault();
      addToBatch();
    }
  }
  
  // Lifecycle
  onMount(() => {
    // Generate or retrieve session ID
    if (!sessionStorage.getItem('sessionId')) {
      sessionStorage.setItem('sessionId', crypto.randomUUID());
    }
    
    // Connect WebSocket
    connectWebSocket();
    
    // Check GPU status
    checkGPUStatus();
    
    // Set up health check interval
    healthCheckInterval = setInterval(performHealthCheck, 30000);
    
    // Add welcome message
    messages = [{
      id: crypto.randomUUID(),
      role: 'system',
      content: `🚀 GPU-Accelerated Legal AI Chat
      
• CUDA acceleration enabled
• TensorRT optimization active
• Multi-user support ready
• Voice input/output available
• Batch processing supported
• Document upload enabled

Type your legal question or upload a document to begin!`,
      timestamp: new Date()
    }];
  });
  
  onDestroy(() => {
    if (ws) {
      ws.close();
    }
    clearTimeout(reconnectTimeout);
    clearInterval(healthCheckInterval);
  });
</script>

<div class="gpu-chat-container">
  <!-- Header -->
  <header class="chat-header">
    <div class="header-content">
      <h1 class="title">
        <span class="gpu-badge">GPU</span>
        <span class="ai-badge">AI</span>
        Legal Assistant
      </h1>
      
      <div class="status-indicators">
        <!-- Connection Status -->
        <div class="connection-status" class:connected={isConnected}>
          <span class="status-dot"></span>
          {isConnected ? 'Connected' : 'Disconnected'}
          <span class="port-info">:{currentPort}</span>
        </div>
        
        <!-- GPU Status -->
        {#if gpuStatus}
          <div class="gpu-status">
            <span class="gpu-icon">🎮</span>
            GPU: {gpuStatus.gpuAvailable ? 'Active' : 'Inactive'}
            {#if gpuStatus.cuda}
              <span class="cuda-info">CUDA {gpuStatus.cuda.version}</span>
            {/if}
            {#if gpuStatus.tensorRT?.enabled}
              <span class="tensorrt-badge">TensorRT</span>
            {/if}
          </div>
        {/if}
        
        <!-- Multi-user indicator -->
        {#if currentRoom}
          <div class="room-status">
            <span>Room: {currentRoom}</span>
            <span class="user-count">👥 {connectedUsers}</span>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Feature toggles -->
    <div class="feature-toggles">
      <label class="toggle">
        <input type="checkbox" bind:checked={voiceEnabled} />
        <span>🔊 Voice</span>
      </label>
      
      <label class="toggle">
        <input type="checkbox" bind:checked={batchMode} />
        <span>📦 Batch Mode</span>
      </label>
      
      <label class="file-upload">
        <input type="file" change={handleFileUpload} accept=".pdf,.txt,.doc,.docx" />
        <span>📎 Upload Document</span>
      </label>
      
      {#if !currentRoom}
        <button on:onclick={() => joinRoom('legal-team')} class="join-room-btn">
          Join Room
        </button>
      {:else}
        <button on:onclick={leaveRoom} class="leave-room-btn">
          Leave Room
        </button>
      {/if}
    </div>
  </header>
  
  <!-- Messages Container -->
  <div class="messages-container">
    <div class="messages-scroll">
      {#each messages as message (message.id)}
        <div class="message {message.role}">
          <div class="message-header">
            <span class="message-role">
              {#if message.role === 'user'}
                👤 You
              {:else if message.role === 'assistant'}
                🤖 AI Assistant
              {:else}
                ⚙️ System
              {/if}
            </span>
            <span class="message-time">
              {message.timestamp.toLocaleTimeString()}
            </span>
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
                <span class="metadata-item">
                  ⚡ {message.metadata.processingTime}ms
                </span>
              {/if}
              {#if message.metadata.gpuUsed}
                <span class="metadata-item gpu-used">GPU</span>
              {/if}
              {#if message.metadata.tensorRT}
                <span class="metadata-item tensorrt">TensorRT</span>
              {/if}
              {#if message.metadata.port}
                <span class="metadata-item">Port: {message.metadata.port}</span>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
      
      {#if isTyping}
        <div class="message assistant typing">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      {/if}
      
      {#if isSpeaking}
        <div class="speaking-indicator">
          <span class="speaker-icon">🔊</span>
          Speaking...
        </div>
      {/if}
    </div>
  </div>
  
  <!-- Batch Items Display -->
  {#if batchMode && batchItems.length > 0}
    <div class="batch-items">
      <h4>Batch Queue ({batchItems.length} items)</h4>
      {#each batchItems as item, i}
        <div class="batch-item">
          {i + 1}. {item}
        </div>
      {/each}
      <button on:onclick={processBatch} class="process-batch-btn">
        Process Batch
      </button>
    </div>
  {/if}
  
  <!-- Uploaded Files Display -->
  {#if uploadedFiles.length > 0}
    <div class="uploaded-files">
      <h4>Uploaded Documents</h4>
      {#each uploadedFiles as file}
        <div class="file-item">
          📄 {file.name} ({Math.round(file.size / 1024)}KB)
        </div>
      {/each}
    </div>
  {/if}
  
  <!-- Input Area -->
  <div class="input-container">
    <textarea
      bind:value={inputMessage}
      onkeypress={handleKeyPress}
      placeholder={batchMode ? "Type message (Shift+Enter to add to batch)..." : "Type your legal question..."}
      class="message-input"
      rows="3"
    />
    
    <div class="input-actions">
      {#if batchMode}
        <button on:onclick={addToBatch} class="add-batch-btn">
          ➕ Add to Batch
        </button>
      {/if}
      
      <button
        on:onclick={sendMessage}
        disabled={!inputMessage.trim() || !isConnected}
        class="send-button"
      >
        <span class="send-icon">🚀</span>
        Send
      </button>
    </div>
  </div>
</div>

<style>
  .gpu-chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
    color: #e0e0e0;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  /* Header */
  .chat-header {
    background: rgba(30, 35, 48, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(100, 255, 218, 0.2);
    padding: 1rem 2rem;
  }
  
  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
  }
  
  .title {
    font-size: 1.5rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .gpu-badge {
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    color: #000;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .ai-badge {
    background: linear-gradient(135deg, #ff00ff 0%, #00ffff 100%);
    color: #000;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
  }
  
  .status-indicators {
    display: flex;
    gap: 2rem;
    align-items: center;
  }
  
  .connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 2rem;
    font-size: 0.875rem;
  }
  
  .port-info {
    opacity: 0.7;
    font-size: 0.75rem;
  }
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ff4444;
    animation: pulse 2s infinite;
  }
  
  .connection-status.connected .status-dot {
    background: #44ff44;
  }
  
  .gpu-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }
  
  .cuda-info {
    padding: 0.125rem 0.5rem;
    background: rgba(118, 185, 0, 0.2);
    border-radius: 0.25rem;
    font-size: 0.75rem;
  }
  
  .tensorrt-badge {
    padding: 0.125rem 0.5rem;
    background: linear-gradient(135deg, #76b900 0%, #00d4aa 100%);
    color: #000;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
  }
  
  .room-status {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    background: rgba(255, 0, 255, 0.1);
    border: 1px solid rgba(255, 0, 255, 0.3);
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }
  
  .user-count {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }
  
  /* Feature toggles */
  .feature-toggles {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    padding: 0.25rem 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.5rem;
    transition: all 0.3s ease;
  }
  
  .toggle:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  
  .toggle input[type="checkbox"] {
    cursor: pointer;
  }
  
  .file-upload {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    padding: 0.25rem 0.75rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.5rem;
    transition: all 0.3s ease;
  }
  
  .file-upload:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  
  .file-upload input[type="file"] {
    display: none;
  }
  
  .join-room-btn, .leave-room-btn {
    padding: 0.25rem 1rem;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    border: none;
    border-radius: 0.5rem;
    color: #000;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  .leave-room-btn {
    background: linear-gradient(135deg, #ff4444 0%, #ff8844 100%);
  }
  
  /* Messages */
  .messages-container {
    flex: 1;
    overflow: hidden;
    padding: 2rem;
  }
  
  .messages-scroll {
    height: 100%;
    overflow-y: auto;
    max-width: 1200px;
    margin: 0 auto;
  }
  
  .message {
    margin-bottom: 1.5rem;
    animation: slideIn 0.3s ease-out;
  }
  
  .message.user {
    margin-left: auto;
    max-width: 70%;
  }
  
  .message.assistant,
  .message.system {
    max-width: 70%;
  }
  
  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
    opacity: 0.7;
  }
  
  .message-content {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.75rem;
    padding: 1rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }
  
  .message.user .message-content {
    background: linear-gradient(135deg, rgba(0, 136, 255, 0.1) 0%, rgba(0, 204, 255, 0.1) 100%);
    border-color: rgba(0, 204, 255, 0.3);
  }
  
  .message.assistant .message-content {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 255, 204, 0.05) 100%);
    border-color: rgba(0, 255, 136, 0.2);
  }
  
  .message.system .message-content {
    background: linear-gradient(135deg, rgba(255, 136, 0, 0.05) 0%, rgba(255, 204, 0, 0.05) 100%);
    border-color: rgba(255, 204, 0, 0.2);
  }
  
  .message-metadata {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 0.5rem;
    font-size: 0.75rem;
    opacity: 0.6;
  }
  
  .metadata-item {
    padding: 0.25rem 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.25rem;
  }
  
  .gpu-used {
    background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 255, 204, 0.2) 100%);
    color: #00ff88;
  }
  
  .tensorrt {
    background: linear-gradient(135deg, rgba(118, 185, 0, 0.2) 0%, rgba(0, 212, 170, 0.2) 100%);
    color: #76b900;
  }
  
  /* Typing Indicator */
  .typing-indicator {
    display: flex;
    gap: 0.25rem;
    padding: 1rem;
  }
  
  .typing-indicator span {
    width: 8px;
    height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: typing 1.4s infinite;
  }
  
  .typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
  }
  
  .typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
  }
  
  /* Speaking Indicator */
  .speaking-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(255, 136, 0, 0.1);
    border-radius: 0.5rem;
    margin-top: 1rem;
    animation: pulse 1s infinite;
  }
  
  /* Batch Items */
  .batch-items {
    background: rgba(30, 35, 48, 0.95);
    border: 1px solid rgba(100, 255, 218, 0.2);
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 0 2rem 1rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .batch-items h4 {
    margin: 0 0 0.75rem 0;
    color: #00ff88;
  }
  
  .batch-item {
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.25rem;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
  }
  
  .process-batch-btn {
    width: 100%;
    padding: 0.75rem;
    background: linear-gradient(135deg, #ff00ff 0%, #00ffff 100%);
    border: none;
    border-radius: 0.5rem;
    color: #000;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  /* Uploaded Files */
  .uploaded-files {
    background: rgba(30, 35, 48, 0.95);
    border: 1px solid rgba(100, 255, 218, 0.2);
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 0 2rem 1rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .uploaded-files h4 {
    margin: 0 0 0.75rem 0;
    color: #00ccff;
  }
  
  .file-item {
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.25rem;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
  }
  
  /* Input Area */
  .input-container {
    display: flex;
    gap: 1rem;
    padding: 2rem;
    background: rgba(30, 35, 48, 0.95);
    border-top: 1px solid rgba(100, 255, 218, 0.2);
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
  }
  
  .message-input {
    flex: 1;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 0.75rem;
    padding: 1rem;
    color: #e0e0e0;
    font-size: 1rem;
    resize: none;
    transition: all 0.3s ease;
  }
  
  .message-input:focus {
    outline: none;
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
  }
  
  .input-actions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .send-button, .add-batch-btn {
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    border: none;
    border-radius: 0.75rem;
    color: #000;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  
  .add-batch-btn {
    background: linear-gradient(135deg, #ff00ff 0%, #00ffff 100%);
    padding: 0.75rem 1.5rem;
  }
  
  .send-button:hover:not(:disabled),
  .add-batch-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
  }
  
  .send-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  /* Animations */
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
  
  @keyframes typing {
    0%, 60%, 100% {
      transform: translateY(0);
      opacity: 0.5;
    }
    30% {
      transform: translateY(-10px);
      opacity: 1;
    }
  }
  
  /* Scrollbar Styling */
  .messages-scroll::-webkit-scrollbar {
    width: 8px;
  }
  
  .messages-scroll::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }
  
  .messages-scroll::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 100%);
    border-radius: 4px;
  }
  
  .messages-scroll::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #00ff88 20%, #00ccff 80%);
  }
</style>


