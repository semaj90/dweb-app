<script lang="ts">
	import { onMount } from 'svelte';
	import { writable, derived } from 'svelte/store';
	import { Button } from 'bits-ui';
	import { Card } from 'bits-ui';
	import { Badge } from 'bits-ui';
	import { Textarea } from 'bits-ui';
	import { Separator } from 'bits-ui';
	import { ScrollArea } from 'bits-ui';
	import { Select } from 'bits-ui';
	import { Switch } from 'bits-ui';
	import { GEMMA3_CONFIG, LEGAL_AI_PROMPTS } from '$lib/config/gemma3-legal-config';
	import { ollamaService } from '$lib/services/ollama-service';
	import { createMachine } from 'xstate';
	import { useMachine } from '@xstate/svelte';
	import type { ChatMessage, AIModel } from '$lib/types/ai';

	// Component props
	export let modelName: string = GEMMA3_CONFIG.model.name;
	export let enableStreaming: boolean = true;
	export let enableContext: boolean = true;
	export let maxTokens: number = 2048;

	// Event dispatcher (use custom event dispatching to avoid deprecated createEventDispatcher)
	function dispatchEvent(name: string, detail: any) {
		const event = new CustomEvent(name, { detail });
		dispatchEvent(event);
	}

	// Stores
	const messages = writable<ChatMessage[]>([]);
	const isLoading = writable(false);
	const currentModel = writable(modelName);
	const streamingEnabled = writable(enableStreaming);
	const contextEnabled = writable(enableContext);

	// Available models
	const availableModels = writable<AIModel[]>([]);

	// Input state
	let inputText = '';
	let textareaElement: HTMLTextAreaElement;

	// XState machine for chat state management
	const chatMachine = createMachine({
		id: 'chat',
		initial: 'idle',
		context: {
			messages: [] as ChatMessage[],
			error: null as string | null,
			model: modelName
		},
		states: {
			idle: {
				on: {
					SEND_MESSAGE: 'processing',
					CHANGE_MODEL: { actions: 'updateModel' },
					CLEAR_CHAT: { actions: 'clearMessages' }
				}
			},
			processing: {
				on: {
					STREAM_TOKEN: { actions: 'addStreamToken' },
					MESSAGE_COMPLETE: { target: 'idle', actions: 'completeMessage' },
					ERROR: { target: 'error', actions: 'setError' }
				}
			},
			error: {
				on: {
					RETRY: 'processing',
					CLEAR_ERROR: 'idle'
				}
			}
		}
	}, {
		actions: {
			updateModel: (context: any, event: { model: string }) => {
				context.model = event.model;
			},
			clearMessages: (context: any) => {
				context.messages = [];
			},
			addStreamToken: (_context: any, _event: any) => {
				// Handle streaming tokens
			},
			completeMessage: (context: any, event: { message: ChatMessage }) => {
				context.messages.push(event.message);
			},
			setError: (context: any, event: { error: string }) => {
				context.error = event.error;
			}
		}
	});

	const { state, send } = useMachine(chatMachine);

	// Derived stores
	const canSend = derived(
		[isLoading],
		([$isLoading]) => !$isLoading && inputText.trim().length > 0
	);

	const messageCount = derived(messages, ($messages) => $messages.length);

	// Load available models on mount
	onMount(async () => {
		try {
			const models = await ollamaService.listModels();
			availableModels.set(models);
		} catch (error: any) {
			console.error('Failed to load models:', error);
			// Use fallback
			availableModels.set([
				{ name: GEMMA3_CONFIG.model.fallback, size: 'Unknown', modified: new Date() },
				{ name: GEMMA3_CONFIG.model.name, size: 'Unknown', modified: new Date() }
			]);
		}
	});

	// Send message function
	async function sendMessage() {
		let $canSend: boolean;
		let $currentModel: string;
		let $streamingEnabled: boolean;
		let $contextEnabled: boolean;
		let $messages: ChatMessage[];
		canSend.subscribe(v => $canSend = v)();
		currentModel.subscribe(v => $currentModel = v)();
		streamingEnabled.subscribe(v => $streamingEnabled = v)();
		contextEnabled.subscribe(v => $contextEnabled = v)();
		messages.subscribe(v => $messages = v)();

		if (!$canSend) return;

		const userMessage: ChatMessage = {
			id: crypto.randomUUID(),
			role: 'user',
			content: inputText.trim(),
			timestamp: new Date(),
			model: $currentModel
		};

		// Add user message
		messages.update(msgs => [...msgs, userMessage]);

		// Clear input
		const messageText = inputText;
		inputText = '';

		// Set loading state
		isLoading.set(true);
		send('SEND_MESSAGE');

		try {
			if ($streamingEnabled) {
				// Handle streaming response
				const assistantMessage: ChatMessage = {
					id: crypto.randomUUID(),
					role: 'assistant',
					content: '',
					timestamp: new Date(),
					model: $currentModel,
					streaming: true
				};

				messages.update(msgs => [...msgs, assistantMessage]);

				const stream = await ollamaService.streamChat($currentModel, messageText, {
					temperature: GEMMA3_CONFIG.parameters.temperature,
					top_p: GEMMA3_CONFIG.parameters.top_p,
					top_k: GEMMA3_CONFIG.parameters.top_k,
					context: $contextEnabled ? $messages : undefined
				});

				for await (const chunk of stream) {
					messages.update(msgs => {
						const lastMessage = msgs[msgs.length - 1];
						if (lastMessage.id === assistantMessage.id) {
							lastMessage.content += chunk.response || '';
						}
						return [...msgs];
					});
				}

				// Mark streaming complete
				messages.update(msgs => {
					const lastMessage = msgs[msgs.length - 1];
					if (lastMessage.id === assistantMessage.id) {
						lastMessage.streaming = false;
					}
					return [...msgs];
				});

			} else {
				// Handle single response
				const response = await ollamaService.chat($currentModel, messageText, {
					temperature: GEMMA3_CONFIG.parameters.temperature,
					context: $contextEnabled ? $messages : undefined
				});

				const assistantMessage: ChatMessage = {
					id: crypto.randomUUID(),
					role: 'assistant',
					content: response.response,
					timestamp: new Date(),
					model: $currentModel
				};

				messages.update(msgs => [...msgs, assistantMessage]);
			}

			send('MESSAGE_COMPLETE');
			// Use CustomEvent for dispatching
			dispatchEvent('messagesSent', { messages: $messages });

		} catch (error: any) {
			console.error('Chat error:', error);
			send('ERROR', { error: error.message });
			dispatchEvent('error', { error: error.message });

			// Add error message
			const errorMessage: ChatMessage = {
				id: crypto.randomUUID(),
				role: 'assistant',
				content: `❌ Error: ${error.message}. Falling back to ${GEMMA3_CONFIG.model.fallback}...`,
				timestamp: new Date(),
				model: $currentModel,
				error: true
			};

			messages.update(msgs => [...msgs, errorMessage]);
		} finally {
			isLoading.set(false);
		}
	}

	// Handle keyboard shortcuts
	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}

	// Clear chat
	function clearChat() {
		messages.set([]);
		send('CLEAR_CHAT');
	}

	// Change model
	function changeModel(newModel: string) {
		currentModel.set(newModel);
		send('CHANGE_MODEL', { model: newModel });
		dispatchEvent('modelChanged', { model: newModel });
	}

	// Apply legal prompt template
	function applyPromptTemplate(template: string) {
		inputText = template + ' ';
		textareaElement?.focus();
	}
</script>

<div class="legal-ai-chat">
	<!-- Header with model selection and controls -->
	<Card.Root class="mb-4">
		<Card.Header>
			<div class="flex items-center justify-between">
						onSelectedChange={(selected: { value: string }) => changeModel(selected.value)}
					<h3 class="text-lg font-semibold">Gemma3 Legal AI</h3>
					<Badge variant={$state.matches('error') ? 'destructive' : 'default'}>
						{$state.value}
					</Badge>
				</div>

				<div class="flex items-center gap-2">
					<!-- Model selection -->
					<Select.Root
						value={$currentModel}
						onSelectedChange={(selected) => changeModel(selected.value)}
					>
						<Select.Trigger class="w-48">
							<Select.Value />
						</Select.Trigger>
						<Select.Content>
							{#each $availableModels as model}
								<Select.Item value={model.name}>
									{model.name}
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>

					<!-- Streaming toggle -->
					<div class="flex items-center gap-2">
						<Switch.Root bind:checked={$streamingEnabled}>
							<Switch.Thumb />
						</Switch.Root>
						<span class="text-sm">Stream</span>
					</div>

					<!-- Context toggle -->
					<div class="flex items-center gap-2">
						<Switch.Root bind:checked={$contextEnabled}>
							<Switch.Thumb />
						</Switch.Root>
						<span class="text-sm">Context</span>
					</div>

					<!-- Clear chat -->
					<Button variant="outline" size="sm" on:click={clearChat}>
						Clear
					</Button>
				</div>
			</div>
		</Card.Header>
	</Card.Root>

						on:click={() => applyPromptTemplate(String(prompt))}
	<Card.Root class="mb-4">
		<Card.Header>
			<h4 class="text-sm font-medium">Legal AI Templates</h4>
		</Card.Header>
		<Card.Content>
			<div class="flex flex-wrap gap-2">
				{#each Object.entries(LEGAL_AI_PROMPTS) as [key, prompt]}
					<Button
						variant="outline"
						size="sm"
						on:click={() => applyPromptTemplate(prompt)}
					>
						{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
					</Button>
				{/each}
			</div>
		</Card.Content>
	</Card.Root>

	<!-- Chat messages -->
	<Card.Root class="flex-1 mb-4">
		<ScrollArea.Root class="h-96">
			<ScrollArea.Viewport class="p-4">
				{#if $messageCount === 0}
					<div class="text-center text-muted-foreground py-8">
						<p class="text-lg mb-2">👨‍⚖️ Gemma3 Legal AI Assistant</p>
						<p class="text-sm">
							Ask me about legal research, document analysis, case law, or contract review.
						</p>
					</div>
				{:else}
					{#each $messages as message (message.id)}
						<div class="message mb-4 {message.role === 'user' ? 'user-message' : 'assistant-message'}">
							<div class="flex items-start gap-3">
								<div class="avatar">
									{#if message.role === 'user'}
										<div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm">
											U
										</div>
									{:else}
										<div class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center text-white text-sm">
											🏛️
										</div>
									{/if}
								</div>

								<div class="flex-1">
									<div class="flex items-center gap-2 mb-1">
										<span class="font-medium text-sm">
											{message.role === 'user' ? 'You' : 'Legal AI'}
										</span>
										<span class="text-xs text-muted-foreground">
											{message.timestamp.toLocaleTimeString()}
										</span>
										{#if message.model}
											<Badge variant="outline" class="text-xs">
												{message.model}
											</Badge>
										{/if}
										{#if message.streaming}
											<div class="typing-indicator">
												<span></span>
												<span></span>
												<span></span>
											</div>
										{/if}
									</div>

									<div class="message-content {message.error ? 'text-red-600' : ''}">
										{@html message.content.replace(/\n/g, '<br>')}
									</div>
								</div>
							</div>
						</div>

						{#if message !== $messages[$messages.length - 1]}
							<Separator class="my-4" />
						{/if}
					{/each}
				{/if}
			</ScrollArea.Viewport>
			<ScrollArea.Scrollbar />
		</ScrollArea.Root>
	</Card.Root>

	<!-- Input area -->
	<Card.Root>
		<Card.Content class="pt-4">
			<div class="flex gap-2">
				<div class="flex-1">
					<Textarea
						bind:value={inputText}
						bind:element={textareaElement}
						placeholder="Ask your legal question here... (Shift+Enter for new line, Enter to send)"
						class="min-h-[80px] resize-none"
						on:keydown={handleKeydown}
						disabled={$isLoading}
					/>
				</div>

				<div class="flex flex-col gap-2">
					<Button
						on:click={sendMessage}
						disabled={!$canSend}
						class="px-6"
					>
						{#if $isLoading}
							<div class="loading-spinner mr-2"></div>
						{/if}
						Send
					</Button>

					<Button
						variant="outline"
						size="sm"
						on:click={() => inputText = ''}
					>
						Clear
					</Button>
<style>
	.legal-ai-chat {
		display: flex;
		flex-direction: column;
		height: 100%;
		max-width: 64rem; /* max-w-4xl */
		margin-left: auto;
		margin-right: auto;
		padding: 1rem;
	}

	.user-message {
		margin-left: 2rem; /* ml-8 */
	}

	.assistant-message {
		margin-right: 2rem; /* mr-8 */
	}

	.message-content {
		max-width: none;
	}
	.message-content :global(p) {
		margin: 0.5em 0;
		font-size: 0.95em;
	}

	.typing-indicator {
		display: flex;
		align-items: center;
		gap: 0.25rem; /* gap-1 */
	}

	.typing-indicator span {
		width: 0.25rem; /* w-1 */
		height: 0.25rem; /* h-1 */
		background-color: currentColor;
		border-radius: 9999px;
		animation: pulse 1s infinite;
		display: inline-block;
		animation-delay: calc(var(--i) * 0.2s);
	}

	.typing-indicator span:nth-child(1) { --i: 0; }
	.typing-indicator span:nth-child(2) { --i: 1; }
	.typing-indicator span:nth-child(3) { --i: 2; }

	@keyframes pulse {
		0%, 100% { opacity: 0.4; }
		50% { opacity: 1; }
	}

	.loading-spinner {
		width: 1rem; /* w-4 */
		height: 1rem; /* h-4 */
		border-width: 2px;
		border-style: solid;
		border-color: currentColor;
		border-top-color: transparent;
		border-radius: 9999px;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}
</style>
		animation-delay: calc(var(--i) * 0.2s);
	}

	.typing-indicator span:nth-child(1) { --i: 0; }
	.typing-indicator span:nth-child(2) { --i: 1; }
	.typing-indicator span:nth-child(3) { --i: 2; }

	.loading-spinner {
		@apply w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin;
	}
</style>
