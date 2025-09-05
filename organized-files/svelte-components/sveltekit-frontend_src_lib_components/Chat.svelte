<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { useMachine } from '@xstate/svelte';
	import { createMachine, assign } from 'xstate';
	import { Button } from '$lib/components/ui';
	import { chatStore } from '$lib/stores/chat';
	import { api } from '$lib/api/client';
	import type { ChatMessage, ChatSession } from '$lib/types/chat';
	
	// XState machine for chat functionality
	const chatMachine = createMachine({
		id: 'chat',
		initial: 'idle',
		context: {
			messages: [] as ChatMessage[],
			currentMessage: '',
			isTyping: false,
			session: null as ChatSession | null,
			error: null as string | null
		},
		states: {
			idle: {
				on: {
					TYPE: {
						target: 'typing',
						actions: assign({
							isTyping: true
						})
					},
					SEND: 'sending',
					CONNECT: 'connecting'
				}
			},
			connecting: {
				invoke: {
					src: 'connectWebSocket',
					onDone: {
						target: 'idle',
						actions: assign({
							session: ({ event }) => event.data
						})
					},
					onError: {
						target: 'error',
						actions: assign({
							error: ({ event }) => event.data.message
						})
					}
				}
			},
			typing: {
				after: {
					2000: {
						target: 'idle',
						actions: assign({
							isTyping: false
						})
					}
				},
				on: {
					STOP_TYPING: {
						target: 'idle',
						actions: assign({
							isTyping: false
						})
					},
					SEND: 'sending'
				}
			},
			sending: {
				invoke: {
					src: 'sendMessage',
					onDone: {
						target: 'idle',
						actions: [
							assign({
								messages: ({ context, event }) => [
									...context.messages,
									event.data.userMessage,
									event.data.aiResponse
								],
								currentMessage: ''
							}),
							'updateChatStore'
						]
					},
					onError: {
						target: 'error',
						actions: assign({
							error: ({ event }) => event.data.message
						})
					}
				}
			},
			error: {
				on: {
					RETRY: 'sending',
					CANCEL: {
						target: 'idle',
						actions: assign({
							error: null
						})
					}
				}
			}
		}
	}, {
		actions: {
			updateChatStore: ({ context }) => {
				chatStore.setMessages(context.messages);
			}
		},
		services: {
			connectWebSocket: async () => {
				const ws = new WebSocket('ws://localhost:8094/ws/chat');
				return new Promise((resolve, reject) => {
					ws.onopen = () => resolve({ id: crypto.randomUUID(), ws });
					ws.onerror = () => reject(new Error('WebSocket connection failed'));
				});
			},
			sendMessage: async ({ context, event }) => {
				const userMessage: ChatMessage = {
					id: crypto.randomUUID(),
					content: event.message,
					role: 'user',
					timestamp: new Date(),
					sessionId: context.session?.id
				};
				
				// Send to Enhanced RAG service
				const response = await api.post('/api/rag/chat', {
					message: event.message,
					context: context.messages.slice(-10), // Last 10 messages for context
					sessionId: context.session?.id
				});
				
				const aiResponse: ChatMessage = {
					id: crypto.randomUUID(),
					content: response.data.response,
					role: 'assistant',
					timestamp: new Date(),
					sessionId: context.session?.id,
					metadata: {
						confidence: response.data.confidence,
						sources: response.data.sources,
						processingTime: response.data.processingTime
					}
				};
				
				return { userMessage, aiResponse };
			}
		}
	});
	
	const { state, send } = useMachine(chatMachine);
	
	let messageInput = '';
	let chatContainer: HTMLDivElement
	let websocket: WebSocket | null = null;
	
	function handleSend() {
		if (messageInput.trim() && !$state.matches('sending')) {
			send({ type: 'SEND', message: messageInput.trim() });
			messageInput = '';
		}
	}
	
	function handleKeyPress(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSend();
		}
	}
	
	function scrollToBottom() {
		if (chatContainer) {
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	}
	
	function formatTime(date: Date): string {
		return new Intl.DateTimeFormat('en-US', {
			hour: '2-digit',
			minute: '2-digit'
		}).format(date);
	}
	
	function handleTyping() {
		send({ type: 'TYPE' });
		// Send typing indicator through WebSocket if connected
		if (websocket && websocket.readyState === WebSocket.OPEN) {
			websocket.send(JSON.stringify({
				type: 'typing',
				sessionId: $state.context.session?.id
			}));
		}
	}
	
	onMount(() => {
		// Connect to WebSocket for real-time features
		send({ type: 'CONNECT' });
		
		// Auto-scroll to bottom when messages change
		$: if ($state.context.messages.length > 0) {
			setTimeout(scrollToBottom, 100);
		}
	});
	
	onDestroy(() => {
		if (websocket) {
			websocket.close();
		}
	});
	
	// Reactive statements
	$: messages = $state.context.messages;
	$: isLoading = $state.matches('sending') || $state.matches('connecting');
	$: hasError = $state.matches('error');
	$: errorMessage = $state.context.error;
</script>

<div class="chat-container flex flex-col h-full bg-background rounded-lg border">
	<!-- Chat Header -->
	<div class="chat-header p-4 border-b bg-muted/50">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-2">
				<div class="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
				<h3 class="font-semibold text-lg">Legal AI Assistant</h3>
			</div>
			<div class="flex items-center space-x-2 text-sm text-muted-foreground">
				{#if $state.context.session}
					<span>Session: {$state.context.session.id.slice(-8)}</span>
				{/if}
				{#if $state.matches('connecting')}
					<span class="text-yellow-600">Connecting...</span>
				{:else if $state.context.session}
					<span class="text-green-600">Connected</span>
				{:else}
					<span class="text-red-600">Disconnected</span>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Chat Messages -->
	<div 
		bind:this={chatContainer}
		class="messages flex-1 overflow-y-auto p-4 space-y-4"
	>
		{#if messages.length === 0}
			<div class="text-center text-muted-foreground py-8">
				<div class="text-4xl mb-4">⚖️</div>
				<h4 class="text-lg font-medium mb-2">Welcome to Legal AI Assistant</h4>
				<p class="text-sm">Ask questions about legal documents, cases, or get AI-powered legal analysis.</p>
			</div>
		{/if}
		
		{#each messages as message}
			<div class="message flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
				<div class="message-content max-w-[80%] {message.role === 'user' ? 'order-2' : 'order-1'}">
					<div class="flex items-end space-x-2 {message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}">
						<!-- Avatar -->
						<div class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center {
							message.role === 'user' 
								? 'bg-primary text-primary-foreground' 
								: 'bg-secondary text-secondary-foreground'
						}">
							{message.role === 'user' ? '👤' : '🤖'}
						</div>
						
						<!-- Message bubble -->
						<div class="message-bubble rounded-lg px-4 py-3 {
							message.role === 'user'
								? 'bg-primary text-primary-foreground'
								: 'bg-muted text-foreground border'
						}">
							<div class="message-text whitespace-pre-wrap break-words">
								{message.content}
							</div>
							
							<!-- Metadata for AI responses -->
							{#if message.role === 'assistant' && message.metadata}
								<div class="mt-2 pt-2 border-t border-border/50 text-xs text-muted-foreground">
									{#if message.metadata.confidence}
										<div>Confidence: {Math.round(message.metadata.confidence * 100)}%</div>
									{/if}
									{#if message.metadata.sources}
										<div>Sources: {message.metadata.sources.length}</div>
									{/if}
									{#if message.metadata.processingTime}
										<div>Response time: {message.metadata.processingTime}ms</div>
									{/if}
								</div>
							{/if}
							
							<!-- Timestamp -->
							<div class="message-time text-xs opacity-70 mt-1">
								{formatTime(message.timestamp)}
							</div>
						</div>
					</div>
				</div>
			</div>
		{/each}
		
		<!-- Typing indicator -->
		{#if $state.matches('typing') || isLoading}
			<div class="message flex justify-start">
				<div class="message-content max-w-[80%]">
					<div class="flex items-end space-x-2">
						<div class="flex-shrink-0 w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center">
							🤖
						</div>
						<div class="message-bubble bg-muted border rounded-lg px-4 py-3">
							<div class="flex space-x-1">
								<div class="w-2 h-2 bg-foreground/60 rounded-full animate-pulse"></div>
								<div class="w-2 h-2 bg-foreground/60 rounded-full animate-pulse" style="animation-delay: 0.2s"></div>
								<div class="w-2 h-2 bg-foreground/60 rounded-full animate-pulse" style="animation-delay: 0.4s"></div>
							</div>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Error Display -->
	{#if hasError}
		<div class="error-banner bg-destructive/10 border-destructive/20 border-t p-3">
			<div class="flex items-center justify-between">
				<div class="flex items-center space-x-2 text-destructive">
					<span class="text-sm font-medium">Error:</span>
					<span class="text-sm">{errorMessage}</span>
				</div>
				<div class="flex space-x-2">
					<Button 
						variant="outline" 
						size="xs" 
						on:click={() => send({ type: 'RETRY' })}
					>
						Retry
					</Button>
					<Button 
						variant="ghost" 
						size="xs" 
						on:click={() => send({ type: 'CANCEL' })}
					>
						Dismiss
					</Button>
				</div>
			</div>
		</div>
	{/if}
	
	<!-- Input Area -->
	<div class="input-area p-4 border-t bg-muted/30">
		<form on:submit|preventDefault={handleSend} class="flex gap-2">
			<div class="flex-1 relative">
				<textarea
					bind:value={messageInput}
					on:input={handleTyping}
					on:keydown={handleKeyPress}
					placeholder="Ask a legal question..."
					class="w-full px-4 py-3 pr-12 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
					rows="1"
					disabled={isLoading}
					style="field-sizing: content max-height: 120px;"
				></textarea>
				
				<!-- Character count -->
				<div class="absolute bottom-2 right-2 text-xs text-muted-foreground">
					{messageInput.length}/2000
				</div>
			</div>
			
			<Button 
				type="submit"
				variant="legal"
				disabled={isLoading || !messageInput.trim() || messageInput.length > 2000}
				loading={isLoading}
				loadingText="Sending..."
				class="self-end"
			>
				{#if !isLoading}
					<svg 
						class="w-4 h-4" 
						fill="none" 
						stroke="currentColor" 
						viewBox="0 0 24 24"
						aria-hidden="true"
					>
						<path 
							stroke-linecap="round" 
							stroke-linejoin="round" 
							stroke-width="2" 
							d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
						/>
					</svg>
				{/if}
			</Button>
		</form>
		
		<!-- Quick actions -->
		<div class="flex flex-wrap gap-2 mt-3">
			<Button 
				variant="outline" 
				size="xs" 
				on:click={() => messageInput = "Analyze this legal document for key terms and clauses."}
			>
				📄 Analyze Document
			</Button>
			<Button 
				variant="outline" 
				size="xs" 
				on:click={() => messageInput = "What are the legal implications of this case?"}
			>
				⚖️ Legal Analysis
			</Button>
			<Button 
				variant="outline" 
				size="xs" 
				on:click={() => messageInput = "Search for similar cases and precedents."}
			>
				🔍 Find Precedents
			</Button>
			<Button 
				variant="outline" 
				size="xs" 
				on:click={() => messageInput = "Generate a case summary."}
			>
				📋 Case Summary
			</Button>
		</div>
	</div>
</div>

<style>
	.chat-container {
		min-height: 400px;
		max-height: 800px;
	}
	
	.messages {
		scroll-behavior: smooth
	}
	
	.message-bubble {
		word-wrap: break-word;
		overflow-wrap: break-word;
	}
	
	textarea {
		field-sizing: content
	}
</style>
