<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // State management
  let isLoading = $state(true);
  let messages = $state([]);
  let currentMessage = $state('');
  let isTyping = $state(false);
  let selectedModel = $state('gemma3-legal');
  
  // Mock initial messages
  const initialMessages = [
    {
      id: '1',
      type: 'system',
      content: 'Legal AI Chat Interface initialized. How can I assist you today?',
      timestamp: new Date(Date.now() - 30000),
      model: 'system'
    }
  ];

  function navigateHome() {
    goto('/');
  }

  async function sendMessage() {
    if (!currentMessage.trim()) return;
    
    const userMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date(),
      model: 'user'
    };
    
    messages = [...messages, userMessage];
    const messageText = currentMessage;
    currentMessage = '';
    isTyping = true;

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `I understand you're asking about "${messageText.substring(0, 50)}...". Let me provide a comprehensive legal analysis based on current law and precedents.`,
        timestamp: new Date(),
        model: selectedModel,
        confidence: Math.random() * 0.3 + 0.7
      };
      
      messages = [...messages, aiResponse];
      isTyping = false;
    }, 2000);
  }

  function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  onMount(() => {
    messages = initialMessages;
    setTimeout(() => {
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>AI Chat Interface - YoRHa Legal AI</title>
</svelte:head>

{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">💬</div>
      <div class="loading-text">INITIALIZING CHAT INTERFACE...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <div class="chat-interface">
    <header class="chat-header">
      <button class="back-button" onclick={navigateHome}>
        ← COMMAND CENTER
      </button>
      <div class="header-title">
        <h1>💬 AI CHAT INTERFACE</h1>
        <div class="header-subtitle">Interactive Chat with Legal AI Models</div>
      </div>
      <select bind:value={selectedModel} class="model-select">
        <option value="gemma3-legal">Gemma 3 Legal</option>
        <option value="llama3-8b">Llama 3 8B</option>
        <option value="mixtral-8x7b">Mixtral 8x7B</option>
      </select>
    </header>

    <main class="chat-main">
      <div class="messages-area">
        {#each messages as message (message.id)}
          <div class="message-wrapper {message.type}">
            {#if message.type === 'system'}
              <div class="system-message">
                <div class="system-icon">⚙️</div>
                <div class="system-text">{message.content}</div>
              </div>
            {:else if message.type === 'user'}
              <div class="user-message">
                <div class="message-text">{message.content}</div>
                <div class="user-avatar">👤</div>
              </div>
            {:else}
              <div class="ai-message">
                <div class="ai-avatar">🤖</div>
                <div class="message-content">
                  <div class="ai-model">{selectedModel}</div>
                  <div class="message-text">{message.content}</div>
                  {#if message.confidence}
                    <div class="confidence">Confidence: {Math.round(message.confidence * 100)}%</div>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        {/each}

        {#if isTyping}
          <div class="message-wrapper assistant">
            <div class="ai-message">
              <div class="ai-avatar">🤖</div>
              <div class="typing-indicator">
                <div class="typing-dots">
                  <div class="dot"></div>
                  <div class="dot"></div>
                  <div class="dot"></div>
                </div>
                <div class="typing-text">AI is thinking...</div>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <div class="input-area">
        <textarea
          bind:value={currentMessage}
          onkeydown={handleKeyPress}
          placeholder="Ask me about legal matters..."
          class="message-input"
          disabled={isTyping}
        ></textarea>
        <button 
          class="send-button" 
          onclick={sendMessage}
          disabled={!currentMessage.trim() || isTyping}
        >
          {isTyping ? '⚙️' : '🚀'} SEND
        </button>
      </div>
    </main>
  </div>
{/if}

<style>
  :global(:root) {
    --yorha-primary: #c4b49a;
    --yorha-accent-warm: #ffbf00;
    --yorha-accent-cool: #4ecdc4;
    --yorha-success: #00ff41;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #1a1a1a;
    --yorha-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  }

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
    font-family: 'JetBrains Mono', monospace;
    color: var(--yorha-light);
  }

  .loading-content {
    text-align: center;
  }

  .loading-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    color: var(--yorha-accent-warm);
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

  .chat-interface {
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    border-bottom: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
  }

  .back-button {
    background: transparent;
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  .header-title h1 {
    margin: 0;
    font-size: 2rem;
    background: linear-gradient(45deg, var(--yorha-accent-warm), var(--yorha-success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
  }

  .header-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-top: 0.5rem;
  }

  .model-select {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-light);
    padding: 0.8rem 1rem;
    font-family: inherit;
    cursor: pointer;
  }

  .chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
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

  .system-text {
    color: var(--yorha-light);
  }

  .user-message {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    max-width: 70%;
  }

  .user-message .message-text {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 4px 18px;
  }

  .user-avatar {
    width: 40px;
    height: 40px;
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
  }

  .ai-message {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    max-width: 80%;
  }

  .ai-avatar {
    width: 40px;
    height: 40px;
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
  }

  .message-content {
    background: rgba(42, 42, 42, 0.8);
    padding: 1rem 1.5rem;
    border-radius: 4px 18px 18px 18px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .ai-model {
    font-size: 0.7rem;
    color: var(--yorha-accent-warm);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  .message-text {
    color: var(--yorha-light);
    line-height: 1.6;
    margin-bottom: 0.5rem;
  }

  .confidence {
    font-size: 0.7rem;
    color: var(--yorha-success);
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
    background: rgba(42, 42, 42, 0.8);
    border-radius: 4px 18px 18px 18px;
  }

  .typing-dots {
    display: flex;
    gap: 0.3rem;
  }

  .dot {
    width: 6px;
    height: 6px;
    background: var(--yorha-accent-warm);
    border-radius: 50%;
    animation: bounce 1.4s ease-in-out infinite both;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  .typing-text {
    font-size: 0.8rem;
    color: var(--yorha-muted);
  }

  .input-area {
    display: flex;
    gap: 1rem;
    padding: 2rem;
    background: rgba(26, 26, 26, 0.9);
    border-top: 1px solid rgba(255, 191, 0, 0.3);
  }

  .message-input {
    flex: 1;
    background: rgba(42, 42, 42, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.5);
    color: var(--yorha-light);
    padding: 1rem;
    font-family: inherit;
    resize: none;
    min-height: 60px;
    border-radius: 8px;
  }

  .message-input:focus {
    outline: none;
    border-color: var(--yorha-accent-warm);
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  .message-input::placeholder {
    color: var(--yorha-muted);
  }

  .send-button {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 6px;
    transition: all 0.3s ease;
  }

  .send-button:hover:not(:disabled) {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .send-button:disabled {
    background: rgba(255, 255, 255, 0.1);
    color: var(--yorha-muted);
    cursor: not-allowed;
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
  }

  @media (max-width: 768px) {
    .chat-header {
      flex-direction: column;
      gap: 1rem;
    }

    .input-area {
      flex-direction: column;
    }

    .user-message,
    .ai-message {
      max-width: 95%;
    }
  }
</style>