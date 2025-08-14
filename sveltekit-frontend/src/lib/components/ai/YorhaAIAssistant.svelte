<script lang="ts">
  import { createMachine, interpret } from 'xstate';
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { Button } from "$lib/components/ui/button";
  import { createDialog, melt } from '@melt-ui/svelte';
  import { X, Send, Minimize2, Maximize2, Bot, MessageSquare } from 'lucide-svelte'; lang="ts">
  import { createMachine, interpret } from 'xstate';
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { Button } from '$lib/components/ui/button';
  import { createDialog } from '@melt-ui/svelte';
  import { X, Send, Minimize2, Maximize2, Bot, MessageSquare } from 'lucide-svelte';

  interface Message {
    id: string;
    content: string;
    role: 'user' | 'assistant' | 'system';
    timestamp: Date;
    status?: 'sending' | 'sent' | 'error';
  }

  interface ChatContext {
    messages: Message[];
    currentMessage: string;
    isLoading: boolean;
    error: string | null;
    isMinimized: boolean;
    isExpanded: boolean;
  }

  // Props
  let { isOpen = $bindable(false) } = $props();

  // Chat machine with XState
  const chatMachine = createMachine<ChatContext>(
    {
      id: 'yorhaChat',
      initial: 'idle',
      context: {
        messages: [
          {
            id: 'welcome',
            content:
              'YoRHa Detective Interface activated. I am your AI assistant ready to help with legal analysis, document review, and case management.',
            role: 'system',
            timestamp: new Date(),
          },
        ],
        currentMessage: '',
        isLoading: false,
        error: null,
        isMinimized: false,
        isExpanded: false,
      },
      states: {
        idle: {
          on: {
            SEND_MESSAGE: 'sending',
            MINIMIZE: { actions: 'minimizeChat' },
            EXPAND: { actions: 'expandChat' },
          },
        },
        sending: {
          entry: 'setSending',
          invoke: {
            src: 'sendMessage',
            onDone: {
              target: 'idle',
              actions: 'onMessageSent',
            },
            onError: {
              target: 'idle',
              actions: 'onMessageError',
            },
          },
        },
      },
    },
    {
      actions: {
        setSending: (context) => {
          context.isLoading = true;
          if (context.currentMessage.trim()) {
            const userMessage: Message = {
              id: Date.now().toString(),
              content: context.currentMessage,
              role: 'user',
              timestamp: new Date(),
              status: 'sending',
            };
            context.messages = [...context.messages, userMessage];
          }
        },
        onMessageSent: (context, event) => {
          context.isLoading = false;
          const response: Message = {
            id: Date.now().toString() + '_response',
            content:
              event.data.content ||
              'I apologize, but I encountered an issue processing your request.',
            role: 'assistant',
            timestamp: new Date(),
          };
          context.messages = [...context.messages, response];
          context.currentMessage = '';
        },
        onMessageError: (context, event) => {
          context.isLoading = false;
          context.error = event.data?.message || 'Failed to send message';
          // Update the last message status to error
          context.messages = context.messages.map((msg) =>
            msg.status === 'sending' ? { ...msg, status: 'error' } : msg
          );
        },
        minimizeChat: (context) => {
          context.isMinimized = true;
          context.isExpanded = false;
        },
        expandChat: (context) => {
          context.isMinimized = false;
          context.isExpanded = true;
        },
      },
      services: {
        sendMessage: async (context) => {
          const response = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: context.currentMessage,
              context: 'legal_assistant',
              history: context.messages.slice(-10), // Last 10 messages for context
            }),
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }

          return response.json();
        },
      },
    }
  );

  let chatService: any;
  let chatState = writable(chatMachine.initialState);
  let currentMessage = '';
  let messagesContainer: HTMLElement;

  // Melt UI dialog
  const {
    elements: { trigger, overlay, content, title, close },
    states: { open },
  } = createDialog({
    forceVisible: true,
    closeOnOutsideClick: false,
  });

  onMount(() => {
    chatService = interpret(chatMachine);
    chatService.subscribe((state: any) => {
      chatState.set(state);
      currentMessage = state.context.currentMessage;
    });
    chatService.start();
  });

  onDestroy(() => {
    chatService?.stop();
  });

  function sendMessage() {
    if (currentMessage.trim() && chatService) {
      chatService.send({
        type: 'SEND_MESSAGE',
        currentMessage: currentMessage.trim(),
      });
    }
  }

  function minimizeChat() {
    chatService?.send('MINIMIZE');
  }

  function expandChat() {
    chatService?.send('EXPAND');
  }

  function handleKeypress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // Auto-scroll to bottom when new messages arrive
  $: if ($chatState.context.messages && messagesContainer) {
    setTimeout(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }, 100);
  }

  // Reactive variables from machine state
  $: messages = $chatState.context.messages || [];
  $: isLoading = $chatState.context.isLoading || false;
  $: error = $chatState.context.error;
  $: isMinimized = $chatState.context.isMinimized || false;
  $: isExpanded = $chatState.context.isExpanded || false;
</script>

<!-- YoRHa AI Assistant Toggle Button -->
<div class="fixed bottom-6 right-6 z-50">
  {#if !isOpen}
    <Button
      use:melt={$trigger}
      onclick={() => (isOpen = true)}
      class="bg-gray-800 hover:bg-gray-700 text-amber-300 border-2 border-amber-500/30 shadow-lg rounded-lg p-4 transition-all duration-300 hover:shadow-amber-500/20 hover:shadow-xl font-mono">
      <Bot class="w-6 h-6 mr-2" />
      YoRHa Detective
    </Button>
  {/if}
</div>

<!-- YoRHa AI Assistant Modal -->
{#if isOpen}
  <div use:melt={$overlay} class="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" />

  <div
    use:melt={$content}
    class="fixed z-50 transition-all duration-300 {isMinimized
      ? 'bottom-6 right-6 w-80 h-16'
      : isExpanded
        ? 'inset-4'
        : 'bottom-6 right-6 w-96 h-[600px]'}">
    <!-- YoRHa Themed Container -->
    <div
      class="w-full h-full bg-gray-900 border-2 border-amber-500/40 rounded-lg shadow-2xl shadow-amber-500/10 overflow-hidden font-mono">
      <!-- Header -->
      <div class="bg-gray-800 border-b border-amber-500/30 p-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="w-3 h-3 bg-amber-500 rounded-full animate-pulse"></div>
          <div class="text-amber-300 font-bold text-sm">YORHA DETECTIVE</div>
          <div class="text-gray-500 text-xs">AI ASSISTANT INTERFACE</div>
        </div>

        <div class="flex items-center space-x-2">
          {#if !isMinimized}
            <Button
              onclick={minimizeChat}
              variant="ghost"
              size="sm"
              class="text-amber-300 hover:text-amber-200 hover:bg-gray-700 p-1">
              <Minimize2 class="w-4 h-4" />
            </Button>

            <Button
              onclick={expandChat}
              variant="ghost"
              size="sm"
              class="text-amber-300 hover:text-amber-200 hover:bg-gray-700 p-1">
              <Maximize2 class="w-4 h-4" />
            </Button>
          {/if}

          <Button
            use:melt={$close}
            onclick={() => (isOpen = false)}
            variant="ghost"
            size="sm"
            class="text-amber-300 hover:text-amber-200 hover:bg-gray-700 p-1">
            <X class="w-4 h-4" />
          </Button>
        </div>
      </div>

      {#if !isMinimized}
        <!-- Messages Area -->
        <div
          bind:this={messagesContainer}
          class="flex-1 overflow-y-auto p-4 space-y-4 h-full {isExpanded
            ? 'max-h-[calc(100vh-200px)]'
            : 'max-h-[440px]'}">
          {#each messages as message (message.id)}
            <div class="message-bubble {message.role}">
              {#if message.role === 'system'}
                <div class="flex items-start space-x-3">
                  <div class="w-8 h-8 bg-amber-500 rounded-full flex items-center justify-center">
                    <Bot class="w-5 h-5 text-gray-900" />
                  </div>
                  <div class="flex-1">
                    <div class="text-amber-300 text-xs mb-1">SYSTEM STATUS</div>
                    <div class="text-gray-300 text-sm leading-relaxed">{message.content}</div>
                  </div>
                </div>
              {:else if message.role === 'assistant'}
                <div class="flex items-start space-x-3">
                  <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                    <Bot class="w-5 h-5 text-white" />
                  </div>
                  <div class="flex-1">
                    <div class="text-blue-300 text-xs mb-1">ASSISTANT RESPONSE</div>
                    <div class="text-gray-300 text-sm leading-relaxed">{message.content}</div>
                  </div>
                </div>
              {:else}
                <div class="flex items-start space-x-3 justify-end">
                  <div class="flex-1 text-right">
                    <div class="text-green-300 text-xs mb-1">USER INPUT</div>
                    <div
                      class="bg-green-800/30 border border-green-500/30 rounded-lg p-3 text-gray-300 text-sm inline-block">
                      {message.content}
                    </div>
                    {#if message.status === 'error'}
                      <div class="text-red-400 text-xs mt-1">Failed to send</div>
                    {/if}
                  </div>
                  <div class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                    <MessageSquare class="w-5 h-5 text-white" />
                  </div>
                </div>
              {/if}
            </div>
          {/each}

          {#if isLoading}
            <div class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <Bot class="w-5 h-5 text-white animate-pulse" />
              </div>
              <div class="flex space-x-1">
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                <div
                  class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                  style="animation-delay: 0.1s">
                </div>
                <div
                  class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"
                  style="animation-delay: 0.2s">
                </div>
              </div>
            </div>
          {/if}

          {#if error}
            <div class="bg-red-900/30 border border-red-500/30 rounded-lg p-3 text-red-300 text-sm">
              Error: {error}
            </div>
          {/if}
        </div>

        <!-- Input Area -->
        <div class="bg-gray-800 border-t border-amber-500/30 p-4">
          <div class="flex space-x-3">
            <textarea
              bind:value={currentMessage}
              onkeydown={handleKeypress}
              placeholder="Enter your query for YoRHa Detective AI..."
              class="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-gray-300 text-sm resize-none focus:border-amber-500 focus:ring-1 focus:ring-amber-500 font-mono"
              rows="2"
              disabled={isLoading}></textarea>

            <Button
              onclick={sendMessage}
              disabled={isLoading || !currentMessage.trim()}
              class="bg-amber-600 hover:bg-amber-500 text-gray-900 px-4 py-2 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center">
              <Send class="w-5 h-5" />
            </Button>
          </div>

          <div class="flex justify-between items-center mt-2 text-xs text-gray-500">
            <div>Press Enter to send, Shift+Enter for new line</div>
            <div class="flex items-center space-x-2">
              <div class="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>ONLINE</span>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  /* @unocss-include */
  .message-bubble {
    animation: slideInUp 0.3s ease-out;
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* Custom scrollbar for YoRHa theme */
  ::-webkit-scrollbar {
    width: 8px;
  }

  ::-webkit-scrollbar-track {
    background: rgb(31, 41, 55);
  }

  ::-webkit-scrollbar-thumb {
    background: rgb(217, 119, 6);
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: rgb(245, 158, 11);
  }
</style>
