<script lang="ts">
  import { createDialog, melt } from '@melt-ui/svelte'
  import { fade, fly, scale } from 'svelte/transition'
  import { onMount } from 'svelte'
  import { 
    Bot, 
    Send, 
    Sparkles, 
    Brain, 
    FileText, 
    Search,
    Zap,
    X,
    Maximize2,
    Minimize2,
    Settings
  } from 'lucide-svelte'
  
  interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: Date
    status?: 'sending' | 'sent' | 'error'
    metadata?: {
      tokens?: number
      model?: string
      processingTime?: number
    }
  }
  
  interface Props {
    isOpen?: boolean
    onClose?: () => void
    caseContext?: {
      id: string
      title: string
    }
  }
  
  let { 
    isOpen = $bindable(true), 
    onClose = () => {},
    caseContext
  }: Props = $props()
  
  let messages = $state<Message[]>([
    {
      id: '1',
      role: 'system',
      content: 'YoRHa Legal AI Assistant initialized. Ready to analyze case data and provide legal insights.',
      timestamp: new Date()
    }
  ])
  
  let inputValue = $state('')
  let isTyping = $state(false)
  let isExpanded = $state(false)
  let showSettings = $state(false)
  let messageContainer: HTMLDivElement
  
  // AI Modes
  const aiModes = [
    { id: 'analyze', label: 'Analyze', icon: Brain, description: 'Deep case analysis' },
    { id: 'search', label: 'Search', icon: Search, description: 'Find relevant cases' },
    { id: 'draft', label: 'Draft', icon: FileText, description: 'Generate documents' },
    { id: 'quick', label: 'Quick', icon: Zap, description: 'Fast responses' }
  ]
  
  let selectedMode = $state('analyze')
  
  // Simulated AI response
  const simulateAIResponse = async (userMessage: string) => {
    isTyping = true
    
    // Add typing indicator
    const typingMessage: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }
    messages = [...messages, typingMessage]
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Generate response based on mode
    let response = ''
    switch (selectedMode) {
      case 'analyze':
        response = `Based on my analysis of "${userMessage}", I've identified several key patterns in the evidence. The documentation suggests a connection to previous cases involving similar digital signatures. Would you like me to elaborate on specific aspects?`
        break
      case 'search':
        response = `I found 3 relevant cases matching your query. The most significant is Case #2024-789 which shares similar evidence patterns. All cases involve digital forensics and encrypted communications.`
        break
      case 'draft':
        response = `I've prepared a draft document based on your request. The structure includes: 1) Executive Summary, 2) Evidence Analysis, 4) Recommendations. Shall I proceed with generating the full document?`
        break
      default:
        response = `Understood. Processing your request: "${userMessage}". How can I assist you further with this case?`
    }
    
    // Update the typing message with actual content
    messages = messages.map(msg => 
      msg.id === typingMessage.id 
        ? { ...msg, content: response, metadata: { tokens: 150, processingTime: 1.5 } }
        : msg
    )
    
    isTyping = false
  }
  
  const sendMessage = async () => {
    if (!inputValue.trim()) return
    
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
      status: 'sent'
    }
    
    messages = [...messages, userMessage]
    inputValue = ''
    
    // Scroll to bottom
    setTimeout(() => {
      if (messageContainer) {
        messageContainer.scrollTop = messageContainer.scrollHeight
      }
    }, 100)
    
    await simulateAIResponse(userMessage.content)
  }
  
  onMount(() => {
    // Auto-scroll to bottom on new messages
    const observer = new MutationObserver(() => {
      if (messageContainer) {
        messageContainer.scrollTop = messageContainer.scrollHeight
      }
    })
    
    if (messageContainer) {
      observer.observe(messageContainer, { childList: true })
    }
    
    return () => observer.disconnect()
  })
</script>

{#if isOpen}
  <div
    class="fixed {isExpanded ? 'inset-4' : 'bottom-4 right-4 w-[400px] h-[600px]'} z-50 flex flex-col nier-transition"
    in:fly={{ y: 100, duration: 300 }}
    out:fly={{ y: 100, duration: 200 }}
  >
    <!-- Main Container -->
    <div class="nier-panel flex flex-col h-full relative">
      <!-- Animated Border Effect -->
      <div class="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
        <div class="absolute inset-0 bg-gradient-to-r from-transparent via-digital-green/20 to-transparent -translate-x-full animate-[shimmer_2s_infinite]"></div>
      </div>
      
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b border-nier-light-gray dark:border-nier-gray/30">
        <div class="flex items-center gap-3">
          <div class="relative">
            <div class="w-10 h-10 bg-nier-gradient-digital rounded-lg flex items-center justify-center animate-pulse">
              <Bot class="w-6 h-6 text-nier-black" />
            </div>
            <div class="absolute -bottom-1 -right-1 w-3 h-3 bg-digital-green rounded-full animate-ping"></div>
          </div>
          
          <div>
            <h3 class="font-display font-semibold nier-heading">AI Legal Assistant</h3>
            {#if caseContext}
              <p class="text-xs text-nier-gray dark:text-nier-silver">
                Analyzing: {caseContext.title}
              </p>
            {/if}
          </div>
        </div>
        
        <div class="flex items-center gap-2">
          <button
            onclick={() => showSettings = !showSettings}
            class="p-2 rounded-lg hover:bg-nier-white/50 dark:hover:bg-nier-black/50 nier-transition"
          >
            <Settings class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
          </button>
          
          <button
            onclick={() => isExpanded = !isExpanded}
            class="p-2 rounded-lg hover:bg-nier-white/50 dark:hover:bg-nier-black/50 nier-transition"
          >
            {#if isExpanded}
              <Minimize2 class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
            {:else}
              <Maximize2 class="w-4 h-4 text-nier-gray dark:text-nier-silver" />
            {/if}
          </button>
          
          <button
            onclick={onClose}
            class="p-2 rounded-lg hover:bg-harvard-crimson/10 nier-transition"
          >
            <X class="w-4 h-4 text-harvard-crimson" />
          </button>
        </div>
      </div>
      
      <!-- AI Mode Selector -->
      {#if showSettings}
        <div class="p-4 border-b border-nier-light-gray dark:border-nier-gray/30" transition:fly={{ y: -10 }}>
          <p class="text-sm font-medium mb-3 text-nier-gray dark:text-nier-silver">AI Analysis Mode</p>
          <div class="grid grid-cols-2 gap-2">
            {#each aiModes as mode}
              <button
                onclick={() => selectedMode = mode.id}
                class="p-3 rounded-lg border nier-transition text-left"
                class:bg-digital-green/10={selectedMode === mode.id}
                class:border-digital-green={selectedMode === mode.id}
                class:border-nier-light-gray={selectedMode !== mode.id}
                class:dark:border-nier-gray/30={selectedMode !== mode.id}
              >
                <div class="flex items-center gap-2 mb-1">
                  <svelte:component 
                    this={mode.icon} 
                    class="w-4 h-4 {selectedMode === mode.id ? 'text-digital-green' : 'text-nier-gray dark:text-nier-silver'}"
                  />
                  <span class="text-sm font-medium {selectedMode === mode.id ? 'text-digital-green' : ''}">{mode.label}</span>
                </div>
                <p class="text-xs text-nier-gray dark:text-nier-silver">{mode.description}</p>
              </button>
            {/each}
          </div>
        </div>
      {/if}
      
      <!-- Messages Container -->
      <div
        bind:this={messageContainer}
        class="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {#each messages as message (message.id)}
          <div
            class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}"
            in:scale={{ duration: 200, start: 0.95 }}
          >
            <div class="max-w-[80%]">
              {#if message.role === 'assistant'}
                <div class="flex items-start gap-2 mb-1">
                  <div class="w-6 h-6 bg-nier-gradient-digital rounded flex items-center justify-center">
                    <Bot class="w-4 h-4 text-nier-black" />
                  </div>
                  <span class="text-xs text-nier-gray dark:text-nier-silver">AI Assistant</span>
                </div>
              {/if}
              
              <div
                class="px-4 py-3 rounded-2xl {message.role === 'user' 
                  ? 'bg-harvard-crimson text-nier-white ml-auto' 
                  : message.role === 'system'
                  ? 'bg-nier-white/50 dark:bg-nier-black/50 text-nier-gray dark:text-nier-silver italic'
                  : 'bg-nier-white dark:bg-nier-dark-gray border border-nier-light-gray dark:border-digital-green/20'}"
              >
                {#if message.content}
                  <p class="text-sm whitespace-pre-wrap">{message.content}</p>
                {:else}
                  <!-- Typing Indicator -->
                  <div class="flex gap-1">
                    <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <div class="w-2 h-2 bg-digital-green rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                  </div>
                {/if}
                
                {#if message.metadata}
                  <div class="flex gap-3 mt-2 pt-2 border-t border-nier-light-gray/50 dark:border-nier-gray/30">
                    {#if message.metadata.tokens}
                      <span class="text-xs text-nier-gray dark:text-nier-silver">
                        {message.metadata.tokens} tokens
                      </span>
                    {/if}
                    {#if message.metadata.processingTime}
                      <span class="text-xs text-nier-gray dark:text-nier-silver">
                        {message.metadata.processingTime}s
                      </span>
                    {/if}
                  </div>
                {/if}
              </div>
              
              <div class="flex items-center gap-2 mt-1">
                <span class="text-xs text-nier-gray dark:text-nier-silver">
                  {new Intl.DateTimeFormat('en-US', { 
                    hour: 'numeric', 
                    minute: 'numeric' 
                  }).format(message.timestamp)}
                </span>
              </div>
            </div>
          </div>
        {/each}
      </div>
      
      <!-- Input Area -->
      <div class="p-4 border-t border-nier-light-gray dark:border-nier-gray/30">
        <form onsubmit|preventDefault={sendMessage} class="flex gap-2">
          <div class="flex-1 relative">
            <input
              bind:value={inputValue}
              placeholder={isTyping ? "AI is thinking..." : "Ask about the case..."}
              disabled={isTyping}
              class="nier-input pr-10"
            />
            <Sparkles class="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-nier-gray dark:text-nier-silver animate-pulse" />
          </div>
          
          <button
            type="submit"
            disabled={!inputValue.trim() || isTyping}
            class="nier-button-digital px-4"
            class:opacity-50={!inputValue.trim() || isTyping}
          >
            <Send class="w-4 h-4" />
          </button>
        </form>
        
        <!-- Quick Actions -->
        <div class="flex gap-2 mt-3">
          <button class="text-xs px-3 py-1 rounded-full bg-nier-white/50 dark:bg-nier-black/50 hover:bg-digital-green/10 nier-transition">
            Summarize Case
          </button>
          <button class="text-xs px-3 py-1 rounded-full bg-nier-white/50 dark:bg-nier-black/50 hover:bg-digital-green/10 nier-transition">
            Find Precedents
          </button>
          <button class="text-xs px-3 py-1 rounded-full bg-nier-white/50 dark:bg-nier-black/50 hover:bg-digital-green/10 nier-transition">
            Generate Report
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  /* @unocss-include */
  @keyframes shimmer {
    to {
      transform: translateX(200%);
    }
  }
</style>
