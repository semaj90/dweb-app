<script lang="ts">
  // AI Button Component - TODO: Enhance with full AI integration
  //
  // 🚀 ENHANCEMENT ROADMAP (See: /ENHANCED_FEATURES_TODO.md)
  // ========================================================
  // 1. GEMMA3 INTEGRATION - Full LLM API integration with streaming
  // 2. CONTEXT AWARENESS - Inject current page/case context into prompts
  // 3. PROACTIVE AI - Smart suggestions based on user activity
  // 4. VOICE INTEGRATION - Speech-to-text and text-to-speech
  // 5. ANIMATION SYSTEM - Advanced micro-interactions and transitions
  // 6. ACCESSIBILITY - Screen reader support and keyboard navigation
  //
  // 📋 WIRING REQUIREMENTS:
  // - Services: Gemma3Service, ContextService, SpeechService
  // - APIs: /api/ai/chat, /api/ai/suggest, /api/ai/context
  // - Stores: Enhanced chatStore with typing indicators
  // - Components: VoiceRecorder, TypingIndicator, SuggestionBubble
  // - Dependencies: @microsoft/speech-sdk, framer-motion equivalent

  import { Button } from "$lib/components/ui/button";
  import {
    currentConversation,
    showProactivePrompt
  } from "$lib/stores/chatStore";
  import { Bot, Sparkles, X } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";
// Call Gemma3 LLM via SvelteKit API

  // Local store for chat open state
  import { writable } from "svelte/store";
  export const isChatOpen = writable(false);
  // Local store for user idle state (mock, replace with real logic if needed)
  export const isUserIdle = writable(false);

  const dispatch = createEventDispatcher();

  // TODO: IMPLEMENT FULL GEMMA3 INTEGRATION
  // ======================================
  // 1. Stream responses with proper typing indicators
  // 2. Context injection from current page/case data
  // 3. Error handling and retry logic
  // 4. Message history persistence
  // 5. Advanced prompt engineering for legal domain
  //
  // ENHANCEMENT: Replace with full service integration
  // ```typescript
  // import { Gemma3Service } from '$lib/services/gemma3';
  // import { ContextService } from '$lib/services/context';
  //
  // async function askGemma3Enhanced(message: string) {
  //   const context = await ContextService.getCurrentContext();
  //   const response = await Gemma3Service.streamChat({
  //     message,
  //     context,
  //     conversationId: $currentConversation?.id,
  //     temperature: 0.7,
  //     maxTokens: 1000
  //   });
  //
  //   // Handle streaming response with typing indicators
  //   for await (const chunk of response) {
  //     updateConversation(chunk);
  //   }
  // }
  // ```

  // --- ENHANCED: Real AI chat integration with context, streaming, and error handling ---
  import { get } from "svelte/store";
  import { onMount } from "svelte";
  let aiState: 'idle' | 'thinking' | 'speaking' = 'idle';
  let unreadCount = 0;
  let $isChatOpen, $isUserIdle, $currentConversation, $showProactivePrompt;

  // Fetch context from backend
  async function fetchContext() {
    try {
      const res = await fetch("/api/ai/context");
      const data = await res.json();
      return data.context;
    } catch (e) {
      return {};
    }
  }

  // Stream chat from backend
  async function askGemma3(message: string) {
    aiState = 'thinking';
    const context = await fetchContext();
    try {
      const res = await fetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          context,
          conversationId: get(currentConversation)?.id,
          model: "llama3",
          temperature: 0.7,
          maxTokens: 1000,
          systemPrompt: `You are an expert legal AI assistant. Context: ${context?.pageType || 'general'}`
        })
      });
      const data = await res.json();
      // Here you would update the chat store with the response
      aiState = 'speaking';
      // Example: updateConversationWithChunk(data.response)
      setTimeout(() => { aiState = 'idle'; }, 1200);
    } catch (e) {
      aiState = 'idle';
      // Handle error, show notification, etc.
    }
  }
  // TODO: IMPLEMENT ENHANCED CHAT TOGGLE WITH CONTEXT
  // ================================================
  // 1. Inject current page context into conversation
  // 2. Smart conversation resumption
  // 3. Proactive suggestions based on user activity
  // 4. Voice activation support
  // 5. Animation state management
  //
  // ENHANCEMENT: Add context awareness
  // ```typescript
  // async function toggleChatEnhanced() {
  //   const wasOpen = $isChatOpen;
  //   isChatOpen.update(open => !open);
  //
  //   if (!wasOpen) {
  //     // Inject current context when opening
  //     const context = await ContextService.getCurrentContext();
  //     const suggestion = await AIService.generateProactiveSuggestion(context);
  //
  //     if (suggestion) {
  //       dispatch('proactive-suggestion', { suggestion, context });
  //     }
  //
  //     // Smart greeting based on context
  //     const greeting = await AIService.generateContextualGreeting(context);
  //     askGemma3Enhanced(greeting);
  //   }
  //
  //   dispatch("toggle", { open: !wasOpen, context });
  // }
  // ```

  // Enhanced chat toggle with context and proactive suggestion
  async function toggleChat() {
    let wasOpen = get(isChatOpen);
    isChatOpen.update(open => !open);
    if (!wasOpen) {
      const context = await fetchContext();
      // Optionally fetch proactive suggestion
      // const suggestion = await fetchProactiveSuggestion(context);
      // if (suggestion) dispatch('proactive-suggestion', { suggestion, context });
      // Smart greeting
      await askGemma3("Hello, how can I help with your legal work?");
    }
    dispatch("toggle", { open: !wasOpen });
  }
  // TODO: IMPLEMENT ADVANCED STATE MANAGEMENT
  // ========================================
  // 1. Smart notification detection with ML
  // 2. Context-aware pulse behavior
  // 3. User activity tracking and idle detection
  // 4. Predictive UI states based on usage patterns
  //
  // ENHANCEMENT: Replace with intelligent state management
  // ```typescript
  // $: hasUnreadMessages = $currentConversation?.messages?.some(m =>
  //   !m.isRead && m.role === 'assistant'
  // );
  //
  // $: shouldPulse = $isUserIdle || $showProactivePrompt ||
  //   ($currentContext?.complexity > 0.7 && !$isChatOpen);
  //
  // $: pulseIntensity = calculatePulseIntensity($userActivity, $contextUrgency);
  // ```

  $: $isChatOpen = get(isChatOpen);
  $: $isUserIdle = get(isUserIdle);
  $: $currentConversation = get(currentConversation);
  $: $showProactivePrompt = get(showProactivePrompt);
  $: hasUnreadMessages = Boolean($currentConversation?.messages && $currentConversation.messages.some(m => !m.isRead && m.role === 'assistant'));
  $: unreadCount = $currentConversation?.messages?.filter(m => !m.isRead && m.role === 'assistant').length || 0;
  $: shouldPulse = $isUserIdle || $showProactivePrompt;
</script>

<!-- Floating AI Button -->
<!--
🔧 ENHANCEMENT OPPORTUNITIES:
- Add voice activation button
- Implement drag-and-drop repositioning
- Add context-aware color themes
- Implement smart hide/show based on page type
- Add keyboard shortcuts (Ctrl+/, Cmd+/)
- Implement gesture recognition for mobile
-->
<div class="container mx-auto px-4">
  <!-- Main Button -->
  <!-- TODO: Add enhanced accessibility and interaction states -->
  <Button
    variant="default"
    size="lg"
    class="relative h-14 w-14 rounded-full shadow-lg transition-all duration-300 ease-in-out"
    on:click={toggleChat}
    aria-label={getAccessibleLabel()}
    aria-expanded={$isChatOpen}
    aria-controls="ai-chat-panel"
    aria-describedby="ai-status-description"
    role="button"
    tabindex="0"
    on:keydown={handleKeyboardShortcuts}
  >
    <!-- Animated Background Rings -->
    <!-- TODO: ENHANCE ANIMATION SYSTEM
         - Add morphing ring patterns based on AI thinking state
         - Implement color-coded rings for different AI modes
         - Add sound wave visualization for voice interactions
         - Create breathing animation for idle state
    -->
    {#if shouldPulse}
      {#each Array(2) as _, i}
        <div
          class="absolute inset-0 rounded-full animate-ping"
          style="
            background: hsl(220, 70%, 60%);
            opacity: {0.3 - (i * 0.1)};
            animation-delay: {i * 0.2}s;
            transform: scale({1 + (i * 0.1)});
          "
        ></div>
      {/each}
    {/if}

    <!-- Icon with transition -->
    <!-- TODO: ENHANCE ICON SYSTEM
         - Add contextual icons (document, case, search modes)
         - Implement AI thinking/processing spinner
         - Add voice wave animation during speech
         - Create morphing transitions between states
         - Add status indicators (online/offline, thinking, error)
    -->
    <div class="flex items-center justify-center h-full w-full">
      {#if $isChatOpen}
        <X class="w-7 h-7" />
      {:else}
        <Bot class="w-7 h-7" />
      {/if}
    </div>

    <!-- Notification Badge -->
    <!-- TODO: ENHANCE NOTIFICATION SYSTEM
         - Add number count for multiple unread messages
         - Implement priority-based badge colors
         - Add smart notification grouping
         - Create contextual notification types (urgent, info, suggestion)
    -->
    {#if hasUnreadMessages && !$isChatOpen}
      <div class="absolute top-1 right-1 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs font-bold">
        {unreadCount}
      </div>
    {/if}

    <!-- Proactive Indicator -->
    <!-- TODO: ENHANCE PROACTIVE SUGGESTIONS
         - Add different suggestion types (tip, warning, insight, action)
         - Implement smart timing based on user activity
         - Create suggestion preview bubbles
         - Add dismissible suggestions with learning
    -->
    {#if $showProactivePrompt && !$isChatOpen}
      <div class="absolute left-1 bottom-1 animate-bounce text-yellow-400">
        <Sparkles class="w-5 h-5" />
      </div>
    {/if}
  </Button>

  <!-- Status Tooltip (appears on hover when closed) -->
  <!-- TODO: ENHANCE TOOLTIP SYSTEM
       - Add contextual tooltip content based on current page
       - Implement smart positioning to avoid screen edges
       - Add rich tooltip content with previews
       - Create keyboard shortcut hints
       - Add multilingual support
  -->
</div>


  // --- Accessibility and keyboard shortcuts ---
  function handleKeyboardShortcuts(event: KeyboardEvent) {
    switch (event.key) {
      case 'Enter':
      case ' ':
        event.preventDefault();
        toggleChat();
        break;
      case '/':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          // TODO: toggleVoiceMode();
        }
        break;
      case 'Escape':
        if ($isChatOpen) {
          toggleChat();
        }
        break;
    }
  }

  function getAccessibleLabel(): string {
    if ($isChatOpen) return 'Close AI Assistant';
    if (hasUnreadMessages) return `Open AI Assistant (${unreadCount} unread messages)`;
    if ($showProactivePrompt) return 'Open AI Assistant (has suggestion)';
    return 'Open AI Assistant';
  }

<!-- Status description for screen readers -->
<div id="ai-status-description" class="sr-only" aria-live="polite">
  {#if aiState === 'thinking'}
    AI is processing your request
  {:else if aiState === 'speaking'}
    AI is responding
  {:else if hasUnreadMessages}
    You have {unreadCount} unread AI messages
  {:else}
    AI assistant is ready to help
  {/if}
</div>
<div class="sr-only">
  Press Enter or Space to toggle. Press Ctrl+Slash for voice mode.
</div>
</div>

<style>
  /* @unocss-include */
  /* TODO: ENHANCE ANIMATION SYSTEM
     - Add CSS custom properties for dynamic theming
     - Implement advanced easing functions
     - Create component-level design tokens
     - Add dark/light mode support
     - Implement reduced motion preferences
  */

  @keyframes gentle-pulse {
    0%,
    100% {
      transform: scale(1);
      opacity: 1;
}
    50% {
      transform: scale(1.05);
      opacity: 0.8;
}}
  /* TODO: Add enhanced animations
  @keyframes ai-thinking {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
  @keyframes voice-wave {
    0%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(1.5); }
}
  @keyframes suggestion-bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-5px); }
    60% { transform: translateY(-3px); }
}
  */
</style>
