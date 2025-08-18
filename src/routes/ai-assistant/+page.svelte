<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  // State management
  let isLoading = $state(true);
  let chatMessages = $state([]);
  let currentMessage = $state('');
  let isTyping = $state(false);
  let selectedModel = $state('gemma3-legal');
  let conversationMode = $state('legal-analysis');
  let systemStatus = $state({
    ollama: 'connected',
    vectorDB: 'active',
    ragPipeline: 'ready',
    legalKB: 'loaded'
  });

  // Available models and modes
  let availableModels = [
    { id: 'gemma3-legal', name: 'Gemma 3 Legal', description: 'Specialized legal analysis model' },
    { id: 'llama3-8b', name: 'Llama 3 8B', description: 'General purpose reasoning' },
    { id: 'mixtral-8x7b', name: 'Mixtral 8x7B', description: 'Expert mixture model' },
    { id: 'qwen2-7b', name: 'Qwen 2 7B', description: 'Advanced language understanding' }
  ];

  let conversationModes = [
    { id: 'legal-analysis', name: 'Legal Analysis', description: 'Analyze legal documents and cases' },
    { id: 'case-research', name: 'Case Research', description: 'Research legal precedents and citations' },
    { id: 'document-review', name: 'Document Review', description: 'Review contracts and legal documents' },
    { id: 'compliance-check', name: 'Compliance Check', description: 'Check regulatory compliance' },
    { id: 'general-counsel', name: 'General Counsel', description: 'General legal consultation' }
  ];

  // Mock conversation history
  let mockMessages = [
    {
      id: 'msg-001',
      type: 'system',
      content: 'Legal AI Assistant initialized. Ready to assist with legal analysis and research.',
      timestamp: new Date(Date.now() - 60000).toISOString(),
      model: 'system'
    },
    {
      id: 'msg-002',
      type: 'user',
      content: 'Can you help me analyze a contract for potential liability issues?',
      timestamp: new Date(Date.now() - 45000).toISOString(),
      model: 'user'
    },
    {
      id: 'msg-003',
      type: 'assistant',
      content: 'I can definitely help you analyze contracts for liability issues. Please provide the contract text or key sections you\'d like me to review. I\'ll examine:\n\n• Limitation of liability clauses\n• Indemnification provisions\n• Force majeure terms\n• Termination conditions\n• Risk allocation mechanisms\n\nWould you like to upload the contract or paste specific sections for analysis?',
      timestamp: new Date(Date.now() - 40000).toISOString(),
      model: 'gemma3-legal',
      confidence: 0.94,
      citations: ['Contract Law Principles', 'Liability Analysis Framework']
    }
  ];

  // Functions
  function navigateHome() {
    goto('/');
  }

  async function sendMessage() {
    if (!currentMessage.trim()) return;
    
    // Add user message
    const userMessage = {
      id: `msg-${Date.now()}`,
      type: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
      model: 'user'
    };
    
    chatMessages = [...chatMessages, userMessage];
    const messageText = currentMessage;
    currentMessage = '';
    isTyping = true;

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = {
        id: `msg-${Date.now()}-ai`,
        type: 'assistant',
        content: generateAIResponse(messageText),
        timestamp: new Date().toISOString(),
        model: selectedModel,
        confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0
        citations: generateCitations(messageText)
      };
      
      chatMessages = [...chatMessages, aiResponse];
      isTyping = false;
    }, 2000 + Math.random() * 2000);
  }

  function generateAIResponse(message) {
    const responses = [
      `Based on your query about "${message.substring(0, 50)}...", I can provide comprehensive legal analysis. Let me break this down into key areas of concern and provide relevant insights from legal precedents.`,
      `I understand you're asking about "${message.substring(0, 50)}...". From a legal perspective, there are several important considerations to address. I'll analyze this systematically using current legal frameworks.`,
      `Your question regarding "${message.substring(0, 50)}..." raises important legal implications. I'll provide a detailed analysis covering relevant statutes, case law, and best practices.`,
      `Thank you for your inquiry about "${message.substring(0, 50)}...". This requires careful legal analysis. I'll examine the applicable legal principles and provide actionable recommendations.`
    ];
    
    return responses[Math.floor(Math.random() * responses.length)];
  }

  function generateCitations(message) {
    const citations = [
      ['Legal Precedent Database', 'Contract Law Review'],
      ['Federal Regulations', 'Case Law Analysis'],
      ['Legal Research Library', 'Compliance Guidelines'],
      ['Statute Analysis', 'Professional Standards']
    ];
    
    return citations[Math.floor(Math.random() * citations.length)];
  }

  function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function getStatusColor(status) {
    switch (status) {
      case 'connected': case 'active': case 'ready': case 'loaded': return '#00ff41';
      case 'processing': case 'loading': return '#ffbf00';
      case 'error': case 'disconnected': return '#ff6b6b';
      default: return '#888';
    }
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'connected': return '🔗';
      case 'active': return '✅';
      case 'ready': return '⚡';
      case 'loaded': return '📚';
      case 'processing': return '⚙️';
      case 'error': return '❌';
      default: return '❓';
    }
  }

  function clearConversation() {
    chatMessages = [mockMessages[0]]; // Keep system message
  }

  function exportConversation() {
    console.log('Exporting conversation:', chatMessages);
  }

  // Initialize component
  onMount(() => {
    chatMessages = mockMessages;
    setTimeout(() => {
      isLoading = false;
    }, 800);

    // Simulate status updates
    const statusInterval = setInterval(() => {
      if (Math.random() > 0.9) {
        const statuses = Object.keys(systemStatus);
        const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
        systemStatus = {
          ...systemStatus,
          [randomStatus]: Math.random() > 0.8 ? 'processing' : systemStatus[randomStatus]
        };
      }
    }, 5000);

    return () => clearInterval(statusInterval);
  });
</script>

<svelte:head>
  <title>AI Assistant - YoRHa Legal AI</title>
  <meta name="description" content="Multi-agent AI assistant with specialized legal knowledge">
</svelte:head>

<!-- Loading Screen -->
{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">🤖</div>
      <div class="loading-text">INITIALIZING AI ASSISTANT...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <!-- Main Interface -->
  <div class="assistant-interface">
    
    <!-- Header -->
    <header class="assistant-header">
      <div class="header-left">
        <button class="back-button" onclick={navigateHome}>
          ← COMMAND CENTER
        </button>
        <div class="header-title">
          <h1>🤖 AI ASSISTANT</h1>
          <div class="header-subtitle">Multi-Agent AI with Specialized Legal Knowledge</div>
        </div>
      </div>
      
      <div class="header-controls">
        <div class="model-selector">
          <label class="selector-label">MODEL:</label>
          <select bind:value={selectedModel} class="model-select">
            {#each availableModels as model}
              <option value={model.id}>{model.name}</option>
            {/each}
          </select>
        </div>
        
        <div class="mode-selector">
          <label class="selector-label">MODE:</label>
          <select bind:value={conversationMode} class="mode-select">
            {#each conversationModes as mode}
              <option value={mode.id}>{mode.name}</option>
            {/each}
          </select>
        </div>
      </div>
    </header>

    <!-- System Status Bar -->
    <section class="status-bar">
      <div class="status-items">
        {#each Object.entries(systemStatus) as [key, status]}
          <div class="status-item">
            <span class="status-icon" style="color: {getStatusColor(status)}">
              {getStatusIcon(status)}
            </span>
            <span class="status-label">{key.toUpperCase()}:</span>
            <span class="status-value" style="color: {getStatusColor(status)}">
              {status.toUpperCase()}
            </span>
          </div>
        {/each}
      </div>
      
      <div class="status-actions">
        <button class="status-btn" onclick={clearConversation}>CLEAR CHAT</button>
        <button class="status-btn" onclick={exportConversation}>EXPORT</button>
      </div>
    </section>

    <!-- Chat Container -->
    <main class="chat-container">
      
      <!-- Messages Area -->
      <div class="messages-area">
        {#each chatMessages as message (message.id)}
          <div class="message-wrapper {message.type}">
            
            <!-- System Message -->
            {#if message.type === 'system'}
              <div class="system-message">
                <div class="system-icon">⚙️</div>
                <div class="system-content">
                  <div class="system-text">{message.content}</div>
                  <div class="system-timestamp">{new Date(message.timestamp).toLocaleString()}</div>
                </div>
              </div>
            
            <!-- User Message -->
            {:else if message.type === 'user'}
              <div class="user-message">
                <div class="message-content">
                  <div class="message-text">{message.content}</div>
                  <div class="message-timestamp">{new Date(message.timestamp).toLocaleTimeString()}</div>
                </div>
                <div class="message-avatar user-avatar">👤</div>
              </div>
            
            <!-- Assistant Message -->
            {:else if message.type === 'assistant'}
              <div class="assistant-message">
                <div class="message-avatar ai-avatar">🤖</div>
                <div class="message-content">
                  <div class="message-header">
                    <div class="ai-model-tag">{availableModels.find(m => m.id === message.model)?.name || message.model}</div>
                    {#if message.confidence}
                      <div class="confidence-score">
                        CONFIDENCE: {Math.round(message.confidence * 100)}%
                      </div>
                    {/if}
                  </div>
                  
                  <div class="message-text">{message.content}</div>
                  
                  {#if message.citations && message.citations.length > 0}
                    <div class="citations">
                      <div class="citations-label">SOURCES:</div>
                      <div class="citations-list">
                        {#each message.citations as citation}
                          <span class="citation-tag">{citation}</span>
                        {/each}
                      </div>
                    </div>
                  {/if}
                  
                  <div class="message-timestamp">{new Date(message.timestamp).toLocaleTimeString()}</div>
                </div>
              </div>
            {/if}
          </div>
        {/each}

        <!-- Typing Indicator -->
        {#if isTyping}
          <div class="message-wrapper assistant">
            <div class="assistant-message">
              <div class="message-avatar ai-avatar">🤖</div>
              <div class="typing-indicator">
                <div class="typing-dots">
                  <div class="typing-dot"></div>
                  <div class="typing-dot"></div>
                  <div class="typing-dot"></div>
                </div>
                <div class="typing-text">AI is analyzing...</div>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Input Area -->
      <div class="input-area">
        <div class="conversation-info">
          <div class="current-mode">
            <span class="mode-label">ACTIVE MODE:</span>
            <span class="mode-name">{conversationModes.find(m => m.id === conversationMode)?.name}</span>
          </div>
          <div class="current-model">
            <span class="model-label">MODEL:</span>
            <span class="model-name">{availableModels.find(m => m.id === selectedModel)?.name}</span>
          </div>
        </div>
        
        <div class="input-container">
          <textarea
            bind:value={currentMessage}
            onkeydown={handleKeyPress}
            placeholder="Ask me anything about legal matters..."
            class="message-input"
            rows="3"
            disabled={isTyping}
          ></textarea>
          
          <div class="input-actions">
            <div class="input-suggestions">
              <button class="suggestion-btn" onclick={() => currentMessage = "Analyze this contract for liability issues"}>
                📄 Contract Analysis
              </button>
              <button class="suggestion-btn" onclick={() => currentMessage = "What are the key compliance requirements for..."}>
                ✅ Compliance Check
              </button>
              <button class="suggestion-btn" onclick={() => currentMessage = "Research case law related to..."}>
                🔍 Case Research
              </button>
            </div>
            
            <button 
              class="send-button" 
              onclick={sendMessage}
              disabled={!currentMessage.trim() || isTyping}
            >
              {isTyping ? '⚙️ PROCESSING' : '🚀 SEND'}
            </button>
          </div>
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="assistant-footer">
      <div class="footer-info">
        <div class="disclaimer">
          ⚠️ AI responses are for informational purposes only and do not constitute legal advice.
        </div>
        <div class="system-info">
          Legal AI Assistant v2.1.5 | Enhanced RAG Pipeline | Last Updated: {new Date().toLocaleString()}
        </div>
      </div>
    </footer>
  </div>
{/if}

<style>
  /* === GLOBAL VARIABLES === */
  :global(:root) {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #ffbf00;
    --yorha-accent-cool: #4ecdc4;
    --yorha-success: #00ff41;
    --yorha-warning: #ffbf00;
    --yorha-error: #ff6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #1a1a1a;
    --yorha-darker: #0a0a0a;
    --yorha-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  }

  /* === LOADING SCREEN === */
  .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background: var(--yorha-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    font-family: 'JetBrains Mono', monospace;
    color: var(--yorha-light);
  }

  .loading-content {
    text-align: center;
    animation: fadeInUp 0.8s ease-out;
  }

  .loading-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    color: var(--yorha-accent-warm);
    animation: pulse 2s ease-in-out infinite;
  }

  .loading-text {
    font-size: 1.2rem;
    color: var(--yorha-muted);
    letter-spacing: 2px;
    margin-bottom: 2rem;
  }

  .loading-bar {
    width: 300px;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 auto;
  }

  .loading-progress {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm), var(--yorha-success));
    animation: loading 2s ease-in-out infinite;
  }

  /* === MAIN INTERFACE === */
  .assistant-interface {
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
    animation: fadeIn 0.5s ease-out;
  }

  /* === HEADER === */
  .assistant-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    border-bottom: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
    backdrop-filter: blur(10px);
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 2rem;
  }

  .back-button {
    background: transparent;
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    transform: translateX(-5px);
  }

  .header-title h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(45deg, var(--yorha-accent-warm), var(--yorha-success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .header-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-top: 0.5rem;
    letter-spacing: 1px;
  }

  .header-controls {
    display: flex;
    gap: 2rem;
    align-items: center;
  }

  .model-selector,
  .mode-selector {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .selector-label {
    font-size: 0.8rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .model-select,
  .mode-select {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-light);
    padding: 0.6rem 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    border-radius: 4px;
  }

  .model-select:focus,
  .mode-select:focus {
    outline: none;
    border-color: var(--yorha-accent-cool);
    box-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
  }

  /* === STATUS BAR === */
  .status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: rgba(42, 42, 42, 0.8);
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
    flex-shrink: 0;
  }

  .status-items {
    display: flex;
    gap: 2rem;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .status-icon {
    font-size: 1rem;
  }

  .status-label {
    color: var(--yorha-muted);
    font-weight: 600;
  }

  .status-value {
    font-weight: 600;
  }

  .status-actions {
    display: flex;
    gap: 1rem;
  }

  .status-btn {
    background: transparent;
    border: 1px solid var(--yorha-accent-warm);
    color: var(--yorha-accent-warm);
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 3px;
  }

  .status-btn:hover {
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
  }

  /* === CHAT CONTAINER === */
  .chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .message-wrapper {
    display: flex;
    width: 100%;
  }

  .message-wrapper.user {
    justify-content: flex-end;
  }

  .message-wrapper.assistant,
  .message-wrapper.system {
    justify-content: flex-start;
  }

  /* === SYSTEM MESSAGE === */
  .system-message {
    display: flex;
    align-items: center;
    gap: 1rem;
    max-width: 80%;
    padding: 1rem;
    background: rgba(255, 191, 0, 0.1);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
  }

  .system-icon {
    font-size: 1.5rem;
    color: var(--yorha-accent-warm);
  }

  .system-content {
    flex: 1;
  }

  .system-text {
    color: var(--yorha-light);
    line-height: 1.5;
    margin-bottom: 0.5rem;
  }

  .system-timestamp {
    font-size: 0.7rem;
    color: var(--yorha-muted);
  }

  /* === USER MESSAGE === */
  .user-message {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    max-width: 70%;
  }

  .user-message .message-content {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.5rem;
  }

  .user-avatar {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  /* === ASSISTANT MESSAGE === */
  .assistant-message {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    max-width: 80%;
  }

  .assistant-message .message-content {
    background: rgba(42, 42, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.5rem;
  }

  .ai-avatar {
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
  }

  .message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    font-weight: 700;
    flex-shrink: 0;
  }

  .message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
  }

  .ai-model-tag {
    background: rgba(255, 191, 0, 0.2);
    color: var(--yorha-accent-warm);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid var(--yorha-accent-warm);
  }

  .confidence-score {
    font-size: 0.7rem;
    color: var(--yorha-success);
    font-weight: 600;
    letter-spacing: 1px;
  }

  .message-text {
    color: var(--yorha-light);
    line-height: 1.6;
    margin-bottom: 0.8rem;
    white-space: pre-wrap;
  }

  .citations {
    margin-bottom: 0.8rem;
  }

  .citations-label {
    font-size: 0.7rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  .citations-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .citation-tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.7rem;
    border: 1px solid var(--yorha-accent-cool);
  }

  .message-timestamp {
    font-size: 0.7rem;
    color: var(--yorha-muted);
  }

  /* === TYPING INDICATOR === */
  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
  }

  .typing-dots {
    display: flex;
    gap: 0.3rem;
  }

  .typing-dot {
    width: 6px;
    height: 6px;
    background: var(--yorha-accent-warm);
    border-radius: 50%;
    animation: typingBounce 1.4s ease-in-out infinite both;
  }

  .typing-dot:nth-child(2) {
    animation-delay: 0.2s;
  }

  .typing-dot:nth-child(3) {
    animation-delay: 0.4s;
  }

  .typing-text {
    font-size: 0.8rem;
    color: var(--yorha-muted);
    font-style: italic;
  }

  /* === INPUT AREA === */
  .input-area {
    flex-shrink: 0;
    background: rgba(26, 26, 26, 0.9);
    border-top: 1px solid rgba(255, 191, 0, 0.3);
    padding: 1.5rem 2rem;
  }

  .conversation-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    font-size: 0.8rem;
  }

  .mode-label,
  .model-label {
    color: var(--yorha-muted);
    margin-right: 0.5rem;
  }

  .mode-name,
  .model-name {
    color: var(--yorha-accent-warm);
    font-weight: 600;
  }

  .input-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .message-input {
    background: rgba(42, 42, 42, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.5);
    color: var(--yorha-light);
    padding: 1rem 1.5rem;
    font-family: inherit;
    font-size: 1rem;
    resize: vertical;
    min-height: 80px;
    border-radius: 8px;
    transition: all 0.3s ease;
  }

  .message-input:focus {
    outline: none;
    border-color: var(--yorha-accent-warm);
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  .message-input::placeholder {
    color: var(--yorha-muted);
    opacity: 0.7;
  }

  .message-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .input-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .input-suggestions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .suggestion-btn {
    background: rgba(78, 205, 196, 0.2);
    border: 1px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-accent-cool);
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .suggestion-btn:hover {
    background: rgba(78, 205, 196, 0.3);
    border-color: var(--yorha-accent-cool);
  }

  .send-button {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 6px;
    font-size: 0.9rem;
  }

  .send-button:hover:not(:disabled) {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .send-button:disabled {
    background: rgba(255, 255, 255, 0.1);
    color: var(--yorha-muted);
    cursor: not-allowed;
    transform: none;
  }

  /* === FOOTER === */
  .assistant-footer {
    flex-shrink: 0;
    background: rgba(26, 26, 26, 0.9);
    border-top: 1px solid rgba(255, 191, 0, 0.3);
    padding: 1rem 2rem;
  }

  .footer-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
  }

  .disclaimer {
    color: var(--yorha-warning);
    font-weight: 600;
  }

  .system-info {
    color: var(--yorha-muted);
  }

  /* === ANIMATIONS === */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(50px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.1); }
  }

  @keyframes typingBounce {
    0%, 80%, 100% {
      transform: scale(0);
      opacity: 0.5;
    }
    40% {
      transform: scale(1);
      opacity: 1;
    }
  }

  /* === RESPONSIVE DESIGN === */
  @media (max-width: 1200px) {
    .header-controls {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-end;
    }
  }

  @media (max-width: 768px) {
    .assistant-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .header-left {
      flex-direction: column;
      gap: 1rem;
    }

    .status-bar {
      flex-direction: column;
      gap: 1rem;
    }

    .status-items {
      flex-wrap: wrap;
      justify-content: center;
    }

    .input-actions {
      flex-direction: column;
      gap: 1rem;
    }

    .input-suggestions {
      justify-content: center;
    }

    .footer-info {
      flex-direction: column;
      gap: 0.5rem;
      text-align: center;
    }

    .user-message,
    .assistant-message {
      max-width: 95%;
    }
  }

  @media (max-width: 480px) {
    .assistant-interface {
      font-size: 0.9rem;
    }

    .messages-area {
      padding: 1rem;
    }

    .input-area {
      padding: 1rem;
    }

    .message-input {
      min-height: 60px;
    }

    .input-suggestions {
      flex-direction: column;
    }
  }
</style>