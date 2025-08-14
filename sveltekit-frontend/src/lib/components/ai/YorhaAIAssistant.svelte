<!-- YorhaAI Assistant - Advanced Chat Interface with SvelteKit 5 + Bits UI + Melt UI -->
<!-- Integrates with go-llama, MCP orchestrator, and tensor transport services -->
<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { browser } from '$app/environment';
	import { createDialog } from '@melt-ui/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Button from '$lib/components/ui/button';
	import * as Input from '$lib/components/ui/input';
	import { Badge } from '$lib/components/ui/badge';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Separator } from '$lib/components/ui/separator';
	import { createMachine, interpret } from 'xstate';
	import type { Interpreter } from 'xstate';
	import { writable, derived } from 'svelte/store';
	import { cn } from '$lib/utils';

	// Props and bindings
	interface YorhaAIAssistantProps {
		userID?: string;
		caseID?: string;
		initialOpen?: boolean;
		theme?: 'light' | 'dark' | 'yorha';
		enableGPUAcceleration?: boolean;
		enableMCPIntegration?: boolean;
	}

	let {
		userID = 'user_' + Date.now(),
		caseID = '',
		initialOpen = false,
		theme = 'yorha',
		enableGPUAcceleration = true,
		enableMCPIntegration = true
	}: YorhaAIAssistantProps = $props();

	// State management with Svelte 5 runes
	let isOpen = $state(initialOpen);
	let currentMessage = $state('');
	let isTyping = $state(false);
	let isConnected = $state(false);
	let streamingResponse = $state(false);
	let userActivity = $state<UserActivity>({
		isTyping: false,
		lastActivity: Date.now(),
		attentionLevel: 'medium',
		currentPage: browser ? window.location.pathname : ''
	});

	// Chat session state
	let chatSession = $state<ChatSession>({
		id: 'session_' + Date.now(),
		userID,
		messages: [],
		context: {
			caseID,
			userIntent: 'general',
			confidence: 0.8,
			recentActions: [],
			preferences: {}
		},
		createdAt: new Date(),
		updatedAt: new Date(),
		isActive: true
	});

	// Performance metrics
	let metrics = $state<PerformanceMetrics>({
		totalMessages: 0,
		averageResponseTime: 0,
		gpuUtilization: 0,
		cacheHitRate: 0,
		connectionLatency: 0
	});

	// WebSocket connections
	let chatSocket: WebSocket | null = $state(null);
	let activitySocket: WebSocket | null = $state(null);

	// XState machine for chat flow
	const chatMachine = createMachine({
		id: 'yorhaChat',
		initial: 'disconnected',
		context: {
			sessionID: chatSession.id,
			userID,
			retryCount: 0,
			lastError: null
		},
		states: {
			disconnected: {
				on: {
					CONNECT: 'connecting'
				}
			},
			connecting: {
				on: {
					CONNECTED: 'idle',
					ERROR: 'error'
				}
			},
			idle: {
				on: {
					SEND_MESSAGE: 'sending',
					START_STREAMING: 'streaming',
					DISCONNECT: 'disconnected'
				}
			},
			sending: {
				on: {
					MESSAGE_SENT: 'waiting_response',
					ERROR: 'error'
				}
			},
			waiting_response: {
				on: {
					RESPONSE_RECEIVED: 'idle',
					START_STREAMING: 'streaming',
					ERROR: 'error'
				}
			},
			streaming: {
				on: {
					STREAM_TOKEN: 'streaming',
					STREAM_COMPLETE: 'idle',
					ERROR: 'error'
				}
			},
			error: {
				on: {
					RETRY: 'connecting',
					RESET: 'disconnected'
				}
			}
		}
	});

	let chatService: Interpreter<any> | null = $state(null);
	let currentState = $derived(chatService?.getSnapshot()?.value || 'disconnected');

	// Types
	interface ChatMessage {
		id: string;
		role: 'user' | 'assistant' | 'system';
		content: string;
		timestamp: Date;
		metadata?: Record<string, any>;
		streaming?: boolean;
	}

	interface ChatSession {
		id: string;
		userID: string;
		messages: ChatMessage[];
		context: ChatContext;
		createdAt: Date;
		updatedAt: Date;
		isActive: boolean;
	}

	interface ChatContext {
		caseID?: string;
		userIntent: string;
		confidence: number;
		recentActions: any[];
		preferences: Record<string, any>;
	}

	interface UserActivity {
		isTyping: boolean;
		lastActivity: number;
		attentionLevel: 'low' | 'medium' | 'high';
		currentPage: string;
	}

	interface PerformanceMetrics {
		totalMessages: number;
		averageResponseTime: number;
		gpuUtilization: number;
		cacheHitRate: number;
		connectionLatency: number;
	}

	// Melt UI Dialog
	const {
		elements: { trigger, overlay, content, title, description, close },
		states: { open }
	} = createDialog({
		preventScroll: true,
		closeOnOutsideClick: false,
		forceVisible: true
	});

	// Reactive derived values
	let connectionStatus = $derived(
		isConnected
			? streamingResponse
				? 'streaming'
				: 'connected'
			: currentState === 'connecting'
				? 'connecting'
				: 'disconnected'
	);

	let canSendMessage = $derived(
		isConnected && 
		currentState === 'idle' && 
		currentMessage.trim() !== '' && 
		!streamingResponse
	);

	let hasMessages = $derived(chatSession.messages.length > 0);
	let lastAssistantMessage = $derived(
		chatSession.messages.filter(m => m.role === 'assistant').pop()
	);

	// Lifecycle
	onMount(() => {
		if (browser) {
			initializeAI();
			setupActivityTracking();
			if (initialOpen) {
				open.set(true);
				isOpen = true;
			}
		}

		return () => {
			cleanup();
		};
	});

	// Initialize AI services
	async function initializeAI() {
		try {
			// Start XState service
			chatService = interpret(chatMachine);
			chatService.start();
			
			// Connect to chat service
			await connectToChatService();
			
			// Connect to activity tracking
			if (enableMCPIntegration) {
				await connectToActivityService();
			}
			
			// Load existing session if available
			await loadChatSession();
			
			isConnected = true;
			chatService?.send('CONNECTED');
		} catch (error) {
			console.error('AI initialization failed:', error);
			chatService?.send('ERROR', { error });
		}
	}

	// Connect to go-llama chat service
	async function connectToChatService() {
		const wsUrl = `ws://localhost:8099/ws/chat?user_id=${userID}&session_id=${chatSession.id}`;
		
		chatSocket = new WebSocket(wsUrl);
		
		chatSocket.onopen = () => {
			console.log('= Connected to YorhaAI chat service');
			isConnected = true;
		};
		
		chatSocket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				handleChatResponse(data);
			} catch (error) {
				console.error('Chat message parse error:', error);
			}
		};
		
		chatSocket.onclose = () => {
			console.log('L Chat service disconnected');
			isConnected = false;
			chatService?.send('DISCONNECT');
		};
		
		chatSocket.onerror = (error) => {
			console.error('Chat service error:', error);
			chatService?.send('ERROR', { error });
		};
	}

	// Connect to user activity service
	async function connectToActivityService() {
		const wsUrl = `ws://localhost:8099/ws/activity?user_id=${userID}`;
		
		activitySocket = new WebSocket(wsUrl);
		
		activitySocket.onopen = () => {
			console.log('=� Connected to activity service');
		};
		
		activitySocket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				handleActivityResponse(data);
			} catch (error) {
				console.error('Activity message parse error:', error);
			}
		};
	}

	// Send message to AI
	async function sendMessage() {
		if (!canSendMessage) return;
		
		const message = currentMessage.trim();
		const timestamp = new Date();
		
		// Add user message to chat
		const userMessage: ChatMessage = {
			id: 'msg_' + Date.now(),
			role: 'user',
			content: message,
			timestamp,
			metadata: {
				userIntent: analyzeUserIntent(message),
				caseID: chatSession.context.caseID
			}
		};
		
		chatSession.messages = [...chatSession.messages, userMessage];
		currentMessage = '';
		
		// Update activity
		updateUserActivity({ action: 'send_message', content_length: message.length });
		
		// Send to chat service
		try {
			chatService?.send('SEND_MESSAGE');
			
			const chatRequest = {
				message,
				user_id: userID,
				session_id: chatSession.id,
				case_id: caseID,
				context: chatSession.context,
				stream: true,
				temperature: 0.7,
				max_tokens: 2000
			};
			
			chatSocket?.send(JSON.stringify(chatRequest));
			metrics.totalMessages++;
			
			chatService?.send('MESSAGE_SENT');
		} catch (error) {
			console.error('Send message error:', error);
			chatService?.send('ERROR', { error });
		}
	}

	// Handle chat service responses
	function handleChatResponse(data: any) {
		if (data.streaming) {
			handleStreamingResponse(data);
		} else {
			handleStandardResponse(data);
		}
	}

	// Handle streaming response
	function handleStreamingResponse(data: any) {
		if (data.token && !data.done) {
			// Update streaming message
			let streamingMessage = chatSession.messages.find(m => m.streaming);
			
			if (!streamingMessage) {
				streamingMessage = {
					id: 'streaming_' + Date.now(),
					role: 'assistant',
					content: data.token,
					timestamp: new Date(),
					streaming: true,
					metadata: {
						session_id: data.session_id,
						gpu_accelerated: enableGPUAcceleration
					}
				};
				chatSession.messages = [...chatSession.messages, streamingMessage];
			} else {
				streamingMessage.content += data.token;
				// Trigger reactivity
				chatSession.messages = [...chatSession.messages];
			}
			
			streamingResponse = true;
			chatService?.send('STREAM_TOKEN');
			
			// Auto-scroll to bottom
			tick().then(() => scrollToBottom());
			
		} else if (data.done) {
			// Finalize streaming message
			const streamingMessage = chatSession.messages.find(m => m.streaming);
			if (streamingMessage) {
				streamingMessage.streaming = false;
				streamingMessage.metadata = {
					...streamingMessage.metadata,
					token_count: data.token_count,
					processing_time: data.processing_time_ms,
					user_intent: data.user_intent
				};
				chatSession.messages = [...chatSession.messages];
			}
			
			streamingResponse = false;
			chatService?.send('STREAM_COMPLETE');
			
			// Update context and metrics
			updateChatContext(data);
			updateMetrics(data);
		}
	}

	// Handle standard response
	function handleStandardResponse(data: any) {
		const assistantMessage: ChatMessage = {
			id: 'msg_' + Date.now(),
			role: 'assistant',
			content: data.response || data.content,
			timestamp: new Date(),
			metadata: {
				session_id: data.session_id,
				token_count: data.token_count,
				processing_time: data.processing_time_ms,
				user_intent: data.user_intent,
				suggestions: data.suggestions || [],
				gpu_accelerated: enableGPUAcceleration
			}
		};
		
		chatSession.messages = [...chatSession.messages, assistantMessage];
		chatService?.send('RESPONSE_RECEIVED');
		
		updateChatContext(data);
		updateMetrics(data);
		
		tick().then(() => scrollToBottom());
	}

	// Handle activity service responses
	function handleActivityResponse(data: any) {
		if (data.attention_level) {
			userActivity.attentionLevel = data.attention_level;
		}
		
		if (data.suggestions) {
			// Handle AI-suggested actions based on user activity
			console.log('AI Activity Suggestions:', data.suggestions);
		}
	}

	// Update user activity
	function updateUserActivity(activity: Record<string, any>) {
		userActivity.lastActivity = Date.now();
		
		if (activity.action === 'typing') {
			userActivity.isTyping = true;
			isTyping = true;
		} else {
			userActivity.isTyping = false;
			isTyping = false;
		}
		
		// Send to activity service
		activitySocket?.send(JSON.stringify({
			...activity,
			user_id: userID,
			timestamp: userActivity.lastActivity,
			page: userActivity.currentPage
		}));
	}

	// Update chat context
	function updateChatContext(data: any) {
		if (data.context) {
			chatSession.context = { ...chatSession.context, ...data.context };
		}
		
		if (data.user_intent) {
			chatSession.context.userIntent = data.user_intent;
		}
		
		if (data.confidence !== undefined) {
			chatSession.context.confidence = data.confidence;
		}
		
		chatSession.updatedAt = new Date();
		
		// Persist session
		saveChatSession();
	}

	// Update performance metrics
	function updateMetrics(data: any) {
		if (data.processing_time_ms) {
			const responseTime = data.processing_time_ms;
			metrics.averageResponseTime = 
				(metrics.averageResponseTime * (metrics.totalMessages - 1) + responseTime) / metrics.totalMessages;
		}
		
		if (data.gpu_used !== undefined) {
			metrics.gpuUtilization = data.gpu_used ? 85 : 20; // Mock values
		}
		
		if (data.cache_hit !== undefined) {
			metrics.cacheHitRate = data.cache_hit ? 0.8 : 0.6; // Mock values
		}
	}

	// Analyze user intent
	function analyzeUserIntent(message: string): string {
		const lowerMsg = message.toLowerCase();
		
		if (lowerMsg.includes('search') || lowerMsg.includes('find')) return 'search';
		if (lowerMsg.includes('summarize') || lowerMsg.includes('summary')) return 'summarize';
		if (lowerMsg.includes('analyze') || lowerMsg.includes('analysis')) return 'analyze';
		if (lowerMsg.includes('draft') || lowerMsg.includes('write')) return 'draft';
		if (lowerMsg.includes('help') || lowerMsg.includes('explain')) return 'help';
		
		return 'general';
	}

	// Input handling
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		} else if (event.key === 'Escape') {
			closeDialog();
		}
		
		// Update typing activity
		updateUserActivity({ action: 'typing', key: event.key });
	}

	// Dialog controls
	function openDialog() {
		open.set(true);
		isOpen = true;
		
		tick().then(() => {
			const input = document.querySelector('[data-yorha-input]') as HTMLElement;
			input?.focus();
		});
	}

	function closeDialog() {
		open.set(false);
		isOpen = false;
	}

	// Utility functions
	function scrollToBottom() {
		const scrollArea = document.querySelector('[data-scroll-area]');
		if (scrollArea) {
			scrollArea.scrollTop = scrollArea.scrollHeight;
		}
	}

	function formatTimestamp(date: Date): string {
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}

	function getMessageIcon(role: string): string {
		switch (role) {
			case 'user': return '=d';
			case 'assistant': return '>';
			case 'system': return '�';
			default: return '=�';
		}
	}

	// Session management
	async function loadChatSession() {
		try {
			const response = await fetch(`/api/chat/sessions/${userID}/${chatSession.id}`);
			if (response.ok) {
				const sessionData = await response.json();
				chatSession = { ...chatSession, ...sessionData };
			}
		} catch (error) {
			console.error('Load session error:', error);
		}
	}

	async function saveChatSession() {
		try {
			await fetch(`/api/chat/sessions/${chatSession.id}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(chatSession)
			});
		} catch (error) {
			console.error('Save session error:', error);
		}
	}

	function clearChat() {
		chatSession.messages = [];
		chatSession.context.recentActions = [];
		chatSession.updatedAt = new Date();
		saveChatSession();
	}

	// Setup activity tracking
	function setupActivityTracking() {
		// Track typing in message input
		let typingTimeout: number;
		
		const updateTyping = () => {
			clearTimeout(typingTimeout);
			updateUserActivity({ action: 'typing' });
			
			typingTimeout = setTimeout(() => {
				updateUserActivity({ action: 'stopped_typing' });
			}, 1000) as any;
		};

		// Global activity tracking
		if (browser) {
			document.addEventListener('keydown', updateTyping);
			document.addEventListener('click', () => 
				updateUserActivity({ action: 'click', page: window.location.pathname })
			);
			
			// Page visibility tracking
			document.addEventListener('visibilitychange', () => {
				updateUserActivity({ 
					action: 'visibility_change', 
					visible: !document.hidden 
				});
			});
		}
	}

	// Cleanup
	function cleanup() {
		chatSocket?.close();
		activitySocket?.close();
		chatService?.stop();
	}

	// Effects
	$effect(() => {
		if (isOpen) {
			tick().then(() => scrollToBottom());
		}
	});

	$effect(() => {
		if (currentMessage) {
			updateUserActivity({ action: 'typing', message_length: currentMessage.length });
		}
	});
</script>

<!-- Main Chat Interface -->
<div class="yorha-ai-assistant">
	<!-- Trigger Button -->
	<Button.Root
		{...$trigger}
		onclick={openDialog}
		variant="default"
		size="lg"
		class={cn(
			"fixed bottom-6 right-6 z-50",
			"bg-gradient-to-r from-blue-500 to-purple-600",
			"hover:from-blue-600 hover:to-purple-700",
			"shadow-lg hover:shadow-xl",
			"transition-all duration-300",
			"rounded-full w-16 h-16",
			"flex items-center justify-center",
			theme === 'yorha' && "bg-gradient-to-r from-amber-400 to-amber-600 hover:from-amber-500 hover:to-amber-700"
		)}
	>
		<span class="text-2xl">></span>
		
		<!-- Status indicator -->
		<div class={cn(
			"absolute -top-1 -right-1 w-4 h-4 rounded-full",
			connectionStatus === 'connected' && "bg-green-400",
			connectionStatus === 'connecting' && "bg-yellow-400 animate-pulse",
			connectionStatus === 'streaming' && "bg-blue-400 animate-pulse",
			connectionStatus === 'disconnected' && "bg-red-400"
		)}></div>
	</Button.Root>

	<!-- Chat Dialog -->
	{#if $open}
		<div 
			{...overlay} 
			use:overlay
			class="fixed inset-0 z-[100] bg-black/50 backdrop-blur-sm"
		>
			<div
				{...content}
				use:content
				class={cn(
					"fixed right-6 bottom-24 z-[101]",
					"w-96 h-[600px]",
					"bg-background border border-border rounded-xl shadow-2xl",
					"flex flex-col overflow-hidden",
					theme === 'yorha' && "bg-gray-900 border-amber-500/30 shadow-amber-500/20"
				)}
			>
				<!-- Header -->
				<div class={cn(
					"flex items-center justify-between p-4 border-b",
					theme === 'yorha' && "border-amber-500/30"
				)}>
					<div class="flex items-center gap-3">
						<div class="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">
							<span class="text-white text-sm font-bold">YA</span>
						</div>
						<div>
							<h3 {...title} use:title class="font-semibold text-sm">
								YorhaAI Assistant
							</h3>
							<p class="text-xs text-muted-foreground">
								{connectionStatus === 'connected' ? 'Ready to assist' : 
								 connectionStatus === 'connecting' ? 'Connecting...' :
								 connectionStatus === 'streaming' ? 'Thinking...' : 'Disconnected'}
							</p>
						</div>
					</div>
					
					<div class="flex items-center gap-2">
						<!-- Metrics badge -->
						<Badge variant="secondary" class="text-xs">
							{metrics.totalMessages} msgs
						</Badge>
						
						<!-- Close button -->
						<Button.Root
							{...$close}
							variant="ghost"
							size="sm"
							onclick={closeDialog}
							class="h-6 w-6 p-0"
						>
							<span class="sr-only">Close</span>
							<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
							</svg>
						</Button.Root>
					</div>
				</div>

				<!-- Messages Area -->
				<ScrollArea class="flex-1 p-4" data-scroll-area>
					{#if !hasMessages}
						<div class="text-center text-muted-foreground py-8">
							<div class="text-4xl mb-4">></div>
							<p class="text-sm">
								Hello! I'm your YorhaAI assistant.<br/>
								How can I help with your legal case today?
							</p>
							
							{#if caseID}
								<Badge variant="outline" class="mt-2">
									Case: {caseID}
								</Badge>
							{/if}
						</div>
					{:else}
						<div class="space-y-4">
							{#each chatSession.messages as message (message.id)}
								<div class={cn(
									"flex gap-3",
									message.role === 'user' ? "justify-end" : "justify-start"
								)}>
									{#if message.role === 'assistant'}
										<div class="w-6 h-6 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
											<span class="text-white text-xs">AI</span>
										</div>
									{/if}
									
									<div class={cn(
										"max-w-[80%] p-3 rounded-lg text-sm",
										message.role === 'user' 
											? "bg-primary text-primary-foreground ml-auto" 
											: "bg-muted",
										message.streaming && "animate-pulse"
									)}>
										<div class="whitespace-pre-wrap">
											{message.content}
										</div>
										
										{#if message.metadata}
											<div class="mt-2 flex items-center gap-2 text-xs opacity-70">
												<span>{formatTimestamp(message.timestamp)}</span>
												
												{#if message.metadata.token_count}
													<Badge variant="secondary" class="text-xs">
														{message.metadata.token_count} tokens
													</Badge>
												{/if}
												
												{#if message.metadata.processing_time}
													<Badge variant="secondary" class="text-xs">
														{Math.round(message.metadata.processing_time)}ms
													</Badge>
												{/if}
												
												{#if message.metadata.gpu_accelerated}
													<Badge variant="secondary" class="text-xs bg-green-500/20 text-green-400">
														GPU
													</Badge>
												{/if}
											</div>
										{/if}
										
										<!-- AI Suggestions -->
										{#if message.metadata?.suggestions?.length > 0}
											<div class="mt-2 space-y-1">
												{#each message.metadata.suggestions as suggestion}
													<button
														class="text-xs text-blue-400 hover:text-blue-300 underline block"
														onclick={() => {
															currentMessage = suggestion;
															sendMessage();
														}}
													>
														=� {suggestion}
													</button>
												{/each}
											</div>
										{/if}
									</div>
									
									{#if message.role === 'user'}
										<div class="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center flex-shrink-0">
											<span class="text-white text-xs">U</span>
										</div>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				</ScrollArea>

				<!-- Input Area -->
				<div class={cn(
					"p-4 border-t space-y-3",
					theme === 'yorha' && "border-amber-500/30"
				)}>
					<!-- Context info -->
					{#if chatSession.context.userIntent !== 'general' || caseID}
						<div class="flex items-center gap-2 text-xs text-muted-foreground">
							{#if chatSession.context.userIntent !== 'general'}
								<Badge variant="outline" class="text-xs">
									Intent: {chatSession.context.userIntent}
								</Badge>
							{/if}
							
							{#if caseID}
								<Badge variant="outline" class="text-xs">
									Case: {caseID}
								</Badge>
							{/if}
							
							<Badge variant="outline" class="text-xs">
								Confidence: {Math.round(chatSession.context.confidence * 100)}%
							</Badge>
						</div>
					{/if}

					<!-- Message input -->
					<div class="flex gap-2">
						<Input.Root
							bind:value={currentMessage}
							placeholder={streamingResponse ? "AI is responding..." : "Type your message..."}
							disabled={!isConnected || streamingResponse}
							onkeydown={handleKeydown}
							data-yorha-input
							class="flex-1"
						/>
						
						<Button.Root
							onclick={sendMessage}
							disabled={!canSendMessage}
							variant="default"
							size="sm"
							class={cn(
								"px-3",
								theme === 'yorha' && "bg-amber-500 hover:bg-amber-600"
							)}
						>
							{#if streamingResponse}
								<div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
							{:else}
								<svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
								</svg>
							{/if}
						</Button.Root>
					</div>

					<!-- Quick actions -->
					<div class="flex justify-between items-center">
						<div class="flex gap-1">
							<Button.Root
								variant="ghost"
								size="sm"
								onclick={clearChat}
								disabled={!hasMessages}
								class="text-xs h-6 px-2"
							>
								Clear
							</Button.Root>
							
							{#if enableMCPIntegration}
								<Button.Root
									variant="ghost"
									size="sm"
									class="text-xs h-6 px-2"
								>
									MCP
								</Button.Root>
							{/if}
							
							{#if enableGPUAcceleration}
								<Badge 
									variant="secondary" 
									class={cn(
										"text-xs h-6",
										metrics.gpuUtilization > 50 && "bg-green-500/20 text-green-400"
									)}
								>
									GPU {Math.round(metrics.gpuUtilization)}%
								</Badge>
							{/if}
						</div>
						
						<div class="text-xs text-muted-foreground">
							{userActivity.attentionLevel} attention
						</div>
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.yorha-ai-assistant {
		/* YorhaUI theme variables */
		--yorha-primary: #d4af37;
		--yorha-secondary: #8b7355;
		--yorha-accent: #f4e4bc;
		--yorha-bg: #1a1a1a;
		--yorha-surface: #2a2a2a;
		--yorha-text: #f4e4bc;
		--yorha-border: rgba(212, 175, 55, 0.3);
		
		/* Animations */
		--animation-glow: 0 0 20px currentColor;
		--animation-pulse: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}

	/* Custom scrollbar for Yorha theme */
	:global([data-scroll-area]::-webkit-scrollbar) {
		width: 8px;
	}

	:global([data-scroll-area]::-webkit-scrollbar-track) {
		background: rgba(212, 175, 55, 0.1);
		border-radius: 4px;
	}

	:global([data-scroll-area]::-webkit-scrollbar-thumb) {
		background: rgba(212, 175, 55, 0.3);
		border-radius: 4px;
	}

	:global([data-scroll-area]::-webkit-scrollbar-thumb:hover) {
		background: rgba(212, 175, 55, 0.5);
	}

	/* Yorha-specific animations */
	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}

	/* Message typing animation */
	:global(.animate-pulse) {
		animation: pulse 1.5s ease-in-out infinite;
	}

	/* GPU utilization indicator */
	:global(.gpu-indicator) {
		position: relative;
	}

	:global(.gpu-indicator::after) {
		content: '';
		position: absolute;
		top: -2px;
		left: -2px;
		right: -2px;
		bottom: -2px;
		background: linear-gradient(45deg, transparent, rgba(34, 197, 94, 0.2), transparent);
		border-radius: inherit;
		z-index: -1;
		animation: var(--animation-pulse);
	}
</style>