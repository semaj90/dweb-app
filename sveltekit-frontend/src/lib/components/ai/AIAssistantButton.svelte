<script lang="ts">
  import { $props, $state, $derived } from 'svelte';
  import { goto } from '$app/navigation';
  import { createTooltip, melt } from '@melt-ui/svelte';
  import { Brain, MessageSquare, Sparkles, Mic, MicOff, Settings } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { Badge } from '$lib/components/ui/badge/index.js';
  
  interface Props {
    variant?: 'floating' | 'inline' | 'compact' | 'full';
    position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
    showStatus?: boolean;
    showBadge?: boolean;
    disabled?: boolean;
    voiceEnabled?: boolean;
    class?: string;
    onclick?: () => void;
  }
  
  let {
    variant = 'floating',
    position = 'bottom-right',
    showStatus = true,
    showBadge = true,
    disabled = false,
    voiceEnabled = true,
    class: className = '',
    onclick
  }: Props = $props();
  
  // AI Assistant state
  let isActive = $state(false);
  let isListening = $state(false);
  let unreadCount = $state(3); // Mock unread suggestions
  let aiStatus = $state<'idle' | 'processing' | 'listening' | 'connected'>('connected');
  
  // Tooltip for compact variants
  const tooltipBuilder = variant === 'compact' ? createTooltip({
    openDelay: 500,
    closeDelay: 100
  }) : null;
  
  const trigger = tooltipBuilder?.elements.trigger;
  const tooltipContent = tooltipBuilder?.elements.content;
  const open = tooltipBuilder?.states.open;
  
  // Dynamic classes
  const buttonClasses = $derived(() => {
    const base = 'ai-assistant-btn transition-all duration-300 font-mono';
    
    const variants = {
      floating: 'fixed z-50 rounded-full shadow-2xl hover:shadow-yorha-accent/20 border-2',
      inline: 'relative rounded-lg shadow-md hover:shadow-lg border',
      compact: 'relative rounded-md shadow-sm hover:shadow-md border',
      full: 'w-full rounded-lg shadow-md hover:shadow-lg border p-4'
    };
    
    const positions = {
      'bottom-right': 'bottom-6 right-6',
      'bottom-left': 'bottom-6 left-6', 
      'top-right': 'top-6 right-6',
      'top-left': 'top-6 left-6'
    };
    
    const statusColors = {
      idle: 'bg-yorha-bg-secondary border-yorha-border-primary text-yorha-text-primary',
      processing: 'bg-yorha-primary/10 border-yorha-primary text-yorha-primary animate-pulse',
      listening: 'bg-red-500/10 border-red-500 text-red-400 animate-pulse',
      connected: 'bg-yorha-accent-gold/10 border-yorha-accent-gold text-yorha-accent-gold'
    };
    
    let classes = `${base} ${variants[variant]} ${statusColors[aiStatus]}`;
    
    if (variant === 'floating') {
      classes += ` ${positions[position]}`;
    }
    
    if (disabled) {
      classes += ' opacity-50 cursor-not-allowed';
    } else {
      classes += ' cursor-pointer hover:scale-105 active:scale-95';
    }
    
    return cn(classes, className);
  });
  
  // Handle click action
  function handleClick() {
    if (disabled) return;
    
    if (onclick) {
      onclick();
    } else {
      // Navigate to AI assistant page
      goto('/aiassistant');
    }
    
    isActive = true;
  }
  
  // Voice input toggle
  function toggleVoiceInput() {
    if (!voiceEnabled) return;
    isListening = !isListening;
    aiStatus = isListening ? 'listening' : 'connected';
  }
  
  // Status indicator component
  function StatusIndicator() {
    const statusConfig = {
      idle: { color: 'bg-gray-400', pulse: false },
      processing: { color: 'bg-yorha-primary', pulse: true },
      listening: { color: 'bg-red-500', pulse: true },
      connected: { color: 'bg-yorha-accent-gold', pulse: false }
    };
    
    const config = statusConfig[aiStatus];
    
    return {
      class: `w-2 h-2 rounded-full ${config.color} ${config.pulse ? 'animate-pulse' : ''}`,
      title: aiStatus.charAt(0).toUpperCase() + aiStatus.slice(1)
    };
  }
</script>

<!-- Floating Variant -->
{#if variant === 'floating'}
  <button
    class={buttonClasses}
    onclick={handleClick}
    {disabled}
    aria-label="Open AI Assistant"
  >
    <div class="relative p-4">
      <Brain class="w-8 h-8" />
      
      {#if showStatus}
        <div 
          class="absolute -top-1 -right-1 {StatusIndicator().class}"
          title={StatusIndicator().title}
        ></div>
      {/if}
      
      {#if showBadge && unreadCount > 0}
        <Badge 
          class="absolute -top-2 -right-2 bg-yorha-accent-gold text-yorha-bg-primary text-xs min-w-[1.25rem] h-5 flex items-center justify-center"
        >
          {unreadCount > 9 ? '9+' : unreadCount}
        </Badge>
      {/if}
    </div>
  </button>

<!-- Inline Variant -->
{:else if variant === 'inline'}
  <button
    class={buttonClasses}
    onclick={handleClick}
    {disabled}
  >
    <div class="flex items-center gap-3 px-4 py-3">
      <div class="relative">
        <Brain class="w-6 h-6" />
        {#if showStatus}
          <div 
            class="absolute -bottom-1 -right-1 {StatusIndicator().class}"
            title={StatusIndicator().title}
          ></div>
        {/if}
      </div>
      
      <div class="flex flex-col items-start">
        <span class="font-semibold text-sm">AI Assistant</span>
        <span class="text-xs text-yorha-text-secondary">
          {aiStatus === 'connected' ? 'Ready to help' : 
           aiStatus === 'processing' ? 'Processing...' :
           aiStatus === 'listening' ? 'Listening...' : 'Offline'}
        </span>
      </div>
      
      {#if voiceEnabled}
        <button
          class="ml-auto p-1 hover:bg-yorha-bg-hover rounded"
          onclick|stopPropagation={toggleVoiceInput}
          aria-label={isListening ? 'Stop listening' : 'Start voice input'}
        >
          {#if isListening}
            <MicOff class="w-4 h-4 text-red-400" />
          {:else}
            <Mic class="w-4 h-4" />
          {/if}
        </button>
      {/if}
      
      {#if showBadge && unreadCount > 0}
        <Badge class="bg-yorha-accent-gold text-yorha-bg-primary text-xs">
          {unreadCount}
        </Badge>
      {/if}
    </div>
  </button>

<!-- Compact Variant with Tooltip -->
{:else if variant === 'compact'}
  {#if tooltipBuilder}
    <button
      use:melt={$trigger}
      class={buttonClasses}
      onclick={handleClick}
      {disabled}
      aria-label="AI Assistant"
    >
      <div class="relative p-2">
        <Brain class="w-5 h-5" />
        {#if showStatus}
          <div 
            class="absolute -top-0.5 -right-0.5 {StatusIndicator().class}"
            title={StatusIndicator().title}
          ></div>
        {/if}
      </div>
    </button>
  {:else}
    <button
      class={buttonClasses}
      onclick={handleClick}
      {disabled}
      aria-label="AI Assistant"
    >
      <div class="relative p-2">
        <Brain class="w-5 h-5" />
        {#if showStatus}
          <div 
            class="absolute -top-0.5 -right-0.5 {StatusIndicator().class}"
            title={StatusIndicator().title}
          ></div>
        {/if}
      </div>
    </button>
  {/if}

<!-- Full Variant -->
{:else if variant === 'full'}
  <button
    class={buttonClasses}
    onclick={handleClick}
    {disabled}
  >
    <div class="flex items-center justify-between w-full">
      <div class="flex items-center gap-4">
        <div class="relative">
          <div class="p-3 rounded-full bg-yorha-primary/10">
            <Brain class="w-6 h-6 text-yorha-primary" />
          </div>
          {#if showStatus}
            <div 
              class="absolute -bottom-1 -right-1 {StatusIndicator().class} w-3 h-3"
              title={StatusIndicator().title}
            ></div>
          {/if}
        </div>
        
        <div class="text-left">
          <h3 class="font-bold text-lg">AI Legal Assistant</h3>
          <p class="text-sm text-yorha-text-secondary">
            Get instant legal analysis, case insights, and research assistance
          </p>
          <div class="flex items-center gap-2 mt-2">
            <span class="text-xs px-2 py-1 bg-yorha-accent-gold/10 text-yorha-accent-gold rounded-full">
              Context7 Enhanced
            </span>
            {#if aiStatus === 'connected'}
              <span class="text-xs px-2 py-1 bg-green-500/10 text-green-400 rounded-full">
                Online
              </span>
            {/if}
          </div>
        </div>
      </div>
      
      <div class="flex items-center gap-3">
        {#if voiceEnabled}
          <button
            class="p-2 hover:bg-yorha-bg-hover rounded-lg"
            onclick|stopPropagation={toggleVoiceInput}
            aria-label={isListening ? 'Stop listening' : 'Start voice input'}
          >
            {#if isListening}
              <MicOff class="w-5 h-5 text-red-400" />
            {:else}
              <Mic class="w-5 h-5" />
            {/if}
          </button>
        {/if}
        
        <div class="flex flex-col items-end">
          {#if showBadge && unreadCount > 0}
            <Badge class="bg-yorha-accent-gold text-yorha-bg-primary mb-1">
              {unreadCount} new suggestions
            </Badge>
          {/if}
          <span class="text-xs text-yorha-text-secondary">Click to open</span>
        </div>
        
        <Sparkles class="w-5 h-5 text-yorha-accent-gold" />
      </div>
    </div>
  </button>
{/if}

<!-- Tooltip for compact variant -->
{#if variant === 'compact' && tooltipBuilder && $open}
  <div
    use:melt={$tooltipContent}
    class="z-50 rounded-lg bg-yorha-bg-primary border border-yorha-border-primary px-3 py-2 shadow-lg"
  >
    <div class="text-sm font-mono">
      <div class="font-semibold text-yorha-text-primary">AI Assistant</div>
      <div class="text-xs text-yorha-text-secondary mt-1">
        Status: {aiStatus}
        {#if unreadCount > 0}
          <br />{unreadCount} new suggestions
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .ai-assistant-btn {
    position: relative;
    overflow: hidden;
  }
  
  .ai-assistant-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(var(--yorha-accent-gold-rgb), 0.2),
      transparent
    );
    transition: left 0.5s ease;
  }
  
  .ai-assistant-btn:hover::before {
    left: 100%;
  }
  
  /* Pulse animation for processing state */
  @keyframes ai-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .ai-assistant-btn[data-status="processing"] {
    animation: ai-pulse 2s infinite;
  }
  
  /* Glowing effect for floating button */
  .ai-assistant-btn.fixed:hover {
    box-shadow: 0 0 30px rgba(var(--yorha-accent-gold-rgb), 0.3);
  }
</style>