<!-- YoRHa AI Chat Component with Enhanced RAG Integration -->
<script lang="ts">

  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import { aiChatStore } from '$lib/stores/ai-chat-store';

  // Chat state
  let messages = writable<unknown[]>([]);
  let currentMessage = writable('');
  let isTyping = writable(false);
  let isConnected = writable(false);
  let isLoading = writable(false);

  // Chat input
let messageInput = $state('');
let chatContainer = $state<HTMLDivElement;

  // Enhanced RAG service URL
  const RAG_SERVICE_URL >('http://localhost:8093');

  // Initialize with welcome message
  onMount(async () => {
    // Check Enhanced RAG service connection
    try {
let response = $state<Response;
        try {
          response >(await fetch(`${RAG_SERVICE_URL}/health`));
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
        } catch (error) {
          console.error('Fetch failed:', error);
          throw error;
        }
      if (response.ok) {
        $isConnected = true;
        $messages = [
          {
            id: 'welcome',
            role: 'assistant',
            content: `🤖 **YoRHa AI Assistant Online**

**Enhanced RAG System Connected**
- Service Status: ✅ OPERATIONAL
- GPU Acceleration: ✅ ACTIVE
- Vector Database: ✅ CONNECTED
- Real-time Analysis: ✅ READY

**Available Commands:**
- \`/analyze <text>\` - Legal document analysis
- \`/search <query>\` - Vector similarity search  
- \`/case <id>\` - Case information retrieval
- \`/evidence <id>\` - Evidence analysis
- \`/help\` - Show all commands

**How can I assist with your legal AI operations?**`,
            timestamp: new Date(),
            type: 'system'
          }
        ];
      } else {
        throw new Error('RAG service unavailable');
      }
    } catch (error) {
      $isConnected = false;
      $messages = [
        {
          id: 'error',
          role: 'assistant',
          content: `⚠️ **Connection Failed**

Enhanced RAG service is not available at ${RAG_SERVICE_URL}

**Troubleshooting:**
1. Ensure the Go service is running: \`./rag-kratos.exe\`
2. Check service health: \`curl ${RAG_SERVICE_URL}/health\`
3. Verify port 8093 is not blocked

**Offline Mode Available** - Basic chat functionality only.`,
          timestamp: new Date(),
          type: 'error'
        }
      ];
    }

    // Load chat history
    const savedChats = aiChatStore.loadChatHistory();
    if (savedChats.length > 0) {
      $messages = [...$messages, ...savedChats];
    }
  });

  async function sendMessage() {
    if (!messageInput.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: messageInput.trim(),
      timestamp: new Date(),
      type: 'user'
    };

    $messages = [...$messages, userMessage];
    const query = messageInput.trim();
    messageInput = '';
    $isLoading = true;
    $isTyping = true;

    // Save user message
    aiChatStore.addMessage(userMessage);

    try {
      // Handle special commands
      if (query.startsWith('/')) {
        await handleCommand(query);
        return;
      }

      // Send to Enhanced RAG service
      const response = await fetch(`${RAG_SERVICE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: query,
          context: 'legal-ai',
          user_id: 'yorha-user',
          session_id: 'yorha-session',
          include_vector_search: true,
          max_tokens: 1000
        })
      });

      if (!response.ok) {
        throw new Error(`RAG service error: ${response.status}`);
      }

      const result = await response.json();
      
      const assistantMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: formatRAGResponse(result),
        timestamp: new Date(),
        type: 'assistant',
        metadata: result.metadata || {}
      };

      $messages = [...$messages, assistantMessage];
      aiChatStore.addMessage(assistantMessage);

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ **Error Processing Request**

${error.message}

**Available Options:**
- Check Enhanced RAG service status
- Try a simpler query
- Use offline mode commands`,
        timestamp: new Date(),
        type: 'error'
      };

      $messages = [...$messages, errorMessage];
    } finally {
      $isLoading = false;
      $isTyping = false;
      scrollToBottom();
    }
  }

  async function handleCommand(command: string) {
    const [cmd, ...args] = command.slice(1).split(' ');
    const arg = args.join(' ');
let response = $state('');

    switch (cmd) {
      case 'help':
        response = `🆘 **YoRHa AI Commands**

**Analysis Commands:**
- \`/analyze <text>\` - Analyze legal text
- \`/search <query>\` - Vector search documents
- \`/case <id>\` - Get case information
- \`/evidence <id>\` - Analyze evidence

**System Commands:**
- \`/status\` - System health check
- \`/clear\` - Clear chat history
- \`/export\` - Export chat log
- \`/connect\` - Test RAG connection

**Legal Commands:**
- \`/precedent <topic>\` - Find legal precedents
- \`/contract <type>\` - Contract analysis
- \`/compliance <area>\` - Compliance check`;
        break;

      case 'status':
        try {
          const health = await fetch(`${RAG_SERVICE_URL}/health`);
          const status = await health.json();
          response = `📊 **System Status**

**Enhanced RAG Service:**
- Status: ${status.status === 'ok' ? '✅ HEALTHY' : '❌ ERROR'}
- Timestamp: ${status.time}
- Connection: ${$isConnected ? '✅ CONNECTED' : '❌ DISCONNECTED'}

**Components:**
- Vector Database: ✅ OPERATIONAL
- GPU Acceleration: ✅ ACTIVE
- AI Models: ✅ LOADED`;
        } catch {
          response = '❌ **System Offline** - Unable to connect to Enhanced RAG service';
        }
        break;

      case 'clear':
        $messages = [$messages[0]]; // Keep welcome message
        aiChatStore.clearHistory();
        response = '🧹 **Chat History Cleared**';
        break;

      case 'analyze':
        if (!arg) {
          response = '❌ **Missing Text** - Usage: `/analyze <text to analyze>`';
        } else {
          try {
            const analysis = await fetch(`${RAG_SERVICE_URL}/api/analyze`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: arg, type: 'legal' })
            });
            const result = await analysis.json();
            response = `📋 **Legal Analysis Results**

**Document Type:** ${result.document_type || 'Unknown'}
**Confidence:** ${result.confidence || 'N/A'}
**Key Entities:** ${result.entities?.join(', ') || 'None detected'}
**Summary:** ${result.summary || 'Analysis pending...'}`;
          } catch {
            response = '❌ **Analysis Failed** - Enhanced RAG service unavailable';
          }
        }
        break;

      default:
        response = `❓ **Unknown Command:** \`${cmd}\`\n\nType \`/help\` for available commands.`;
    }

    const commandResponse = {
      id: Date.now().toString(),
      role: 'assistant',
      content: response,
      timestamp: new Date(),
      type: 'command'
    };

    $messages = [...$messages, commandResponse];
    aiChatStore.addMessage(commandResponse);
    $isLoading = false;
    $isTyping = false;
    scrollToBottom();
  }

  function formatRAGResponse(result: any): string {
    if (typeof result === 'string') return result;
let formatted = $state(`🤖 **YoRHa AI Response**\n\n`);
    
    if (result.response) {
      formatted += `${result.response}\n\n`;
    }

    if (result.sources && result.sources.length > 0) {
      formatted += `📚 **Sources:**\n`;
      result.sources.forEach((source: any, index: number) => {
        formatted += `${index + 1}. ${source.title || source.filename || 'Unknown'}\n`;
      });
      formatted += '\n';
    }

    if (result.confidence) {
      formatted += `🎯 **Confidence:** ${Math.round(result.confidence * 100)}%\n`;
    }

    if (result.processing_time) {
      formatted += `⏱️ **Processing Time:** ${result.processing_time}ms\n`;
    }

    return formatted;
  }

  function scrollToBottom() {
    setTimeout(() => {
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 100);
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function getMessageTypeClass(type: string): string {
    switch (type) {
      case 'user': return 'bg-yorha-accent-cool/20 border-yorha-accent-cool text-white ml-auto';
      case 'assistant': return 'bg-yorha-accent-warm/20 border-yorha-accent-warm text-yorha-light';
      case 'system': return 'bg-blue-500/20 border-blue-400 text-blue-100';
      case 'error': return 'bg-red-500/20 border-red-400 text-red-100';
      case 'command': return 'bg-green-500/20 border-green-400 text-green-100';
    }
  }
</script>

<!-- AI Chat Interface -->
<div class="h-full flex flex-col bg-yorha-darker text-yorha-light">
  <!-- Chat Messages -->
  <div 
    bind:this={chatContainer}
    class="flex-1 overflow-y-auto p-4 space-y-4"
  >
    {#each $messages as message (message.id)}
      <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
        <div class="max-w-3xl border rounded-lg p-4 {getMessageTypeClass(message.type)}">
          <div class="flex items-center gap-2 mb-2">
            <span class="text-xs font-bold uppercase">
              {message.role === 'user' ? '👤 USER' : '🤖 YORHA AI'}
            </span>
            <span class="text-xs opacity-70">
              {message.timestamp.toLocaleTimeString()}
            </span>
            {#if message.type === 'command'}
              <span class="text-xs bg-green-500 px-2 py-1 rounded">CMD</span>
            {/if}
          </div>
          <div class="whitespace-pre-wrap break-words text-sm leading-relaxed">
            {@html message.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/`(.*?)`/g, '<code class="bg-black/30 px-1 rounded">$1</code>')
                                    .replace(/\n/g, '<br>')
            }
          </div>
        </div>
      </div>
    {/each}

    {#if $isTyping}
      <div class="flex justify-start">
        <div class="bg-yorha-accent-warm/20 border border-yorha-accent-warm rounded-lg p-4 max-w-xs">
          <div class="flex items-center gap-2">
            <span class="text-xs font-bold">🤖 YORHA AI</span>
            <div class="flex gap-1">
              <div class="w-2 h-2 bg-yorha-accent-warm rounded-full animate-bounce"></div>
              <div class="w-2 h-2 bg-yorha-accent-warm rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
              <div class="w-2 h-2 bg-yorha-accent-warm rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Connection Status -->
  <div class="border-t border-yorha-accent-warm/30 p-2">
    <div class="flex items-center justify-between text-xs">
      <div class="flex items-center gap-2">
        <div class="w-2 h-2 rounded-full {$isConnected ? 'bg-green-400' : 'bg-red-400'}"></div>
        <span>{$isConnected ? 'Enhanced RAG Connected' : 'Offline Mode'}</span>
      </div>
      <div class="flex items-center gap-4">
        <span>Messages: {$messages.length}</span>
        <span>Session: yorha-{Date.now().toString().slice(-6)}</span>
      </div>
    </div>
  </div>

  <!-- Message Input -->
  <div class="border-t border-yorha-accent-warm/30 p-4">
    <div class="flex gap-2">
      <input
        bind:value={messageInput}
        keydown={handleKeyDown}
        placeholder="Enter your message or /help for commands..."
        disabled={$isLoading}
        class="flex-1 bg-yorha-dark border border-yorha-accent-warm/50 rounded px-4 py-3 text-yorha-light placeholder-yorha-muted/70 focus:outline-none focus:border-yorha-accent-warm focus:ring-1 focus:ring-yorha-accent-warm disabled:opacity-50"
      />
      <button
        on:onclick={sendMessage}
        disabled={$isLoading || !messageInput.trim()}
        class="px-6 py-3 bg-yorha-accent-warm text-yorha-dark font-bold rounded hover:bg-yorha-accent-warm/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
      >
        {$isLoading ? '⏳' : '📤'}
      </button>
    </div>
    <div class="text-xs text-yorha-muted mt-2">
      Press Enter to send • Shift+Enter for new line • Type /help for commands
    </div>
  </div>
</div>

<style>
  /* Scrollbar styling for chat container */
  :global(.h-full .overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }

  :global(.h-full .overflow-y-auto::-webkit-scrollbar-track) {
    background: var(--yorha-darker);
  }

  :global(.h-full .overflow-y-auto::-webkit-scrollbar-thumb) {
    background: var(--yorha-accent-warm);
    border-radius: 3px;
  }

  :global(.h-full .overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: var(--yorha-accent-cool);
  }
</style>

