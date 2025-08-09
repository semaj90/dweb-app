<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<!-- @ts-nocheck -->
<script>
	// @ts-nocheck
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	const cn = (...classes) => classes.filter(Boolean).join(' ');

	// Props
	export let caseId = null;
	export let sessionId = null;
	export let placeholder = 'Ask about legal matters, case analysis, or evidence review...';
	export let maxHeight = '600px';
	export let enableRAG = true;
	export let maxContextChunks = 5;

	// Reactive stores
	/** @type {import('svelte/store').Writable<Array<{id: string, role: string, content: string, timestamp: string, isError?: boolean, isStreaming?: boolean}>>} */
	const messages = writable([]);
	const isLoading = writable(false);
	const isStreaming = writable(false);
	const currentInput = writable('');
	const systemStatus = writable({ status: 'checking', message: 'Initializing AI system...' });
	/** @type {import('svelte/store').Writable<Array<any>>} */
	const contextSources = writable([]);

	// Component state
	let messageContainer;
	let inputElement;
	let currentSources = [];
	let chatSession = sessionId;
	let streamingMessage = '';
	let abortController = null;

	// Configuration
	const API_BASE = '/api/chat';
	const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

	onMount(async () => {
		await checkSystemHealth();
		// Set up periodic health checks
		const healthInterval = setInterval(checkSystemHealth, HEALTH_CHECK_INTERVAL);

		return () => {
			clearInterval(healthInterval);
		};
	});

	/**
	 * Check system health and update status
	 */
	async function checkSystemHealth() {
		try {
			const response = await fetch(`${API_BASE}?action=health`);
			const data = await response.json();

			if (data.status === 'healthy') {
				systemStatus.set({
					status: 'ready',
					message: 'AI system ready',
					details: data
				});
			} else {
				systemStatus.set({
					status: 'warning',
					message: 'System partially available',
					details: data
				});
			}
		} catch (error) {
			systemStatus.set({
				status: 'error',
				message: 'AI system unavailable',
				error: error.message
			});
		}
	}

	/**
	 * Send message and handle streaming response
	 */
	async function sendMessage() {
		const input = $currentInput.trim();
		if (!input || $isLoading || $isStreaming) return;

		// Add user message to chat
		messages.update(msgs => [...msgs, {
			id: Date.now().toString(),
			role: 'user',
			content: input,
			timestamp: new Date().toISOString()
		}]);

		// Clear input and set loading state
		currentInput.set('');
		isLoading.set(true);
		isStreaming.set(true);
		contextSources.set([]);
		streamingMessage = '';

		// Create abort controller for this request
		abortController = new AbortController();

		try {
			const response = await fetch(API_BASE, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					message: input,
					sessionId: chatSession,
					caseId,
					options: {
						stream: true,
						maxContextChunks,
						includeKnowledgeBase: enableRAG,
						includeDocuments: enableRAG
					}
				}),
				signal: abortController.signal
			});

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			// Handle streaming response
			const reader = response.body?.getReader();
			if (!reader) {
				throw new Error('No response body available');
			}

			// Add placeholder for assistant message
			const assistantMessageId = Date.now().toString() + '_assistant';
			messages.update(msgs => [...msgs, {
				id: assistantMessageId,
				role: 'assistant',
				content: '',
				timestamp: new Date().toISOString(),
				isStreaming: true
			}]);

			// Process streaming data
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				const chunk = new TextDecoder().decode(value);
				const lines = chunk.split('\n').filter(line => line.trim());

				for (const line of lines) {
					if (line.startsWith('data: ')) {
						try {
							const data = JSON.parse(line.slice(6));
							await handleStreamingData(data, assistantMessageId);
						} catch (parseError) {
							console.error('Error parsing streaming data:', parseError);
						}
					}
				}
			}

		} catch (error) {
			if (error.name !== 'AbortError') {
				console.error('Chat error:', error);

				// Add error message
				messages.update(msgs => [...msgs, {
					id: Date.now().toString(),
					role: 'system',
					content: `Error: ${error.message}. Please try again or check system status.`,
					timestamp: new Date().toISOString(),
					isError: true
				}]);
			}
		} finally {
			isLoading.set(false);
			isStreaming.set(false);
			abortController = null;
		}

		// Auto-scroll to bottom
		setTimeout(scrollToBottom, 100);
	}

	/**
	 * Handle streaming data from the server
	 */
	async function handleStreamingData(data, messageId) {
		switch (data.type) {
			case 'status':
				// Update system status or show progress
				systemStatus.update(status => ({
					...status,
					message: data.message
				}));
				break;

			case 'context':
				// Store context sources for display
				contextSources.set(data.sources || []);
				currentSources = data.sources || [];
				break;

			case 'token':
				// Append token to streaming message
				streamingMessage += data.content;
				messages.update(msgs =>
					msgs.map(msg =>
						msg.id === messageId
							? { ...msg, content: streamingMessage }
							: msg
					)
				);
				break;

			case 'complete':
				// Mark message as complete
				messages.update(msgs =>
					msgs.map(msg =>
						msg.id === messageId
							? {
								...msg,
								isStreaming: false,
								sources: data.sources,
								responseTime: data.responseTime,
								tokensUsed: data.tokensUsed
							}
							: msg
					)
				);

				// Update session ID if new
				if (data.sessionId && !chatSession) {
					chatSession = data.sessionId;
				}
				break;

			case 'error':
				// Handle streaming error
				messages.update(msgs =>
					msgs.map(msg =>
						msg.id === messageId
							? {
								...msg,
								content: `Error: ${data.error}`,
								isStreaming: false,
								isError: true
							}
							: msg
					)
				);
				break;
		}
	}

	/**
	 * Stop current streaming response
	 */
	function stopStreaming() {
		if (abortController) {
			abortController.abort();
		}
		isStreaming.set(false);
		isLoading.set(false);
	}

	/**
	 * Handle input keypress
	 */
	function handleKeyPress(event) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}

	/**
	 * Scroll chat to bottom
	 */
	function scrollToBottom() {
		if (messageContainer) {
			messageContainer.scrollTop = messageContainer.scrollHeight;
		}
	}

	/**
	 * Clear chat history
	 */
	function clearChat() {
		messages.set([]);
		contextSources.set([]);
		chatSession = null;
	}

	/**
	 * Format timestamp for display
	 */
	function formatTime(timestamp) {
		return new Date(timestamp).toLocaleTimeString([], {
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	/**
	 * Get status badge variant
	 */
	function getStatusVariant(status) {
		switch (status) {
			case 'ready': return 'success';
			case 'warning': return 'warning';
			case 'error': return 'destructive';
			default: return 'secondary';
		}
	}
</script>

<div class="flex flex-col h-full max-w-4xl mx-auto border rounded-lg shadow-lg bg-white">
	<!-- Header -->
	<div class="flex items-center justify-between p-4 border-b">
		<div class="flex items-center space-x-3">
			<h3 class="text-lg font-semibold text-amber-100">Legal AI Assistant</h3>
			{#if caseId}
				<span class="text-xs bg-gray-100 px-2 py-1 rounded border">Case: {caseId}</span>
			{/if}
		</div>

		<div class="flex items-center space-x-2">
			<span class="text-xs px-2 py-1 rounded" class:bg-green-100={$systemStatus.status === 'ready'} class:bg-yellow-100={$systemStatus.status === 'warning'} class:bg-red-100={$systemStatus.status === 'error'} class:bg-gray-100={$systemStatus.status === 'checking'}>
				{$systemStatus.message}
			</span>

			{#if $messages.length > 0}
				<button class="text-sm px-2 py-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded" on:click={clearChat}>
					Clear Chat
				</button>
			{/if}
		</div>
	</div>

	<!-- Messages Area -->
	<div class="flex-1 p-4 overflow-y-auto" style="height: {maxHeight};">
		<div bind:this={messageContainer} class="space-y-4">
			{#each $messages as message (message.id)}
				<div class={cn(
					"flex flex-col space-y-2",
					message.role === 'user' ? 'items-end' : 'items-start'
				)}>
					<!-- Message Content -->
					<div class={cn(
						"max-w-[80%] p-3 rounded-lg shadow-sm",
						message.role === 'user'
							? 'bg-amber-600 text-white ml-auto'
							: message.isError
								? 'bg-red-900/50 text-red-100 border border-red-700'
								: 'bg-gray-800 text-gray-100 border border-gray-700'
					)}>
						<div class="prose prose-sm max-w-none">
							{#if message.role === 'assistant' && message.isStreaming}
								<div class="flex items-center space-x-2">
									<span>{message.content}</span>
									<div class="animate-pulse">▋</div>
								</div>
							{:else}
								<div class="whitespace-pre-wrap">{message.content}</div>
							{/if}
						</div>

						<!-- Message Sources -->
						{#if message.sources && message.sources.length > 0}
							<div class="mt-3 pt-2 border-t border-gray-600">
								<p class="text-xs text-gray-400 mb-2">Sources used:</p>
								<div class="flex flex-wrap gap-1">
									{#each message.sources as source}
										<span class="text-xs bg-gray-200 px-2 py-1 rounded">
											{source.type}: {source.title.substring(0, 30)}...
											<span class="ml-1 text-green-400">
												({(source.similarity * 100).toFixed(1)}%)
											</span>
										</span>
									{/each}
								</div>
							</div>
						{/if}

						<!-- Message Metadata -->
						{#if message.responseTime || message.tokensUsed}
							<div class="mt-2 pt-2 border-t border-gray-600 text-xs text-gray-400">
								{#if message.responseTime}
									<span>Response: {message.responseTime}ms</span>
								{/if}
								{#if message.tokensUsed}
									<span class="ml-2">Tokens: {message.tokensUsed}</span>
								{/if}
							</div>
						{/if}
					</div>

					<!-- Timestamp -->
					<span class="text-xs text-gray-500">
						{formatTime(message.timestamp)}
					</span>
				</div>
			{/each}

			<!-- Loading Indicator -->
			{#if $isLoading && !$isStreaming}
				<div class="flex items-center space-x-2 text-gray-400">
					<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-amber-500"></div>
					<span class="text-sm">Processing your request...</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- Context Sources Display -->
	{#if $contextSources.length > 0}
		<div class="p-3 bg-gray-800/50 border-t border-gray-700">
			<p class="text-xs text-gray-400 mb-2">Using context from:</p>
			<div class="flex flex-wrap gap-1">
				{#each $contextSources as source}
					<span class="text-xs bg-gray-100 px-2 py-1 rounded border">
						{source.type}: {source.title.substring(0, 40)}...
						<span class="ml-1 text-green-400">
							({(source.similarity * 100).toFixed(1)}%)
						</span>
					</span>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Input Area -->
	<div class="p-4 border-t">
		<div class="flex space-x-2">
			<input
				bind:this={inputElement}
				bind:value={$currentInput}
				{placeholder}
				disabled={$isLoading || $systemStatus.status === 'error'}
				on:keypress={handleKeyPress}
				class="flex-1"
			/>

			{#if $isStreaming}
				<button class="text-sm px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600" on:click={stopStreaming}>
					Stop
				</button>
			{:else}
				<button class="px-4 py-2 bg-amber-600 text-white rounded hover:bg-amber-700"
					on:click={sendMessage}
					disabled={!$currentInput.trim() || $isLoading || $systemStatus.status === 'error'}
				>
					Send
				</button>
			{/if}
		</div>

		<!-- RAG Controls -->
		{#if enableRAG}
			<div class="mt-2 flex items-center justify-between text-xs text-gray-400">
				<span>RAG-enhanced responses with legal context retrieval</span>
				<span>Max context chunks: {maxContextChunks}</span>
			</div>
		{/if}
	</div>
</div>

<style>
	:global(.prose) {
		color: inherit;
	}

	:global(.prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6) {
		color: inherit;
		margin-top: 1em;
		margin-bottom: 0.5em;
	}

	:global(.prose p) {
		margin-bottom: 0.75em;
	}

	:global(.prose ul, .prose ol) {
		margin-left: 1em;
		margin-bottom: 0.75em;
	}

	:global(.prose strong) {
		font-weight: 600;
		color: #fbbf24;
	}

	:global(.prose em) {
		font-style: italic;
		color: #d1d5db;
	}
</style>
