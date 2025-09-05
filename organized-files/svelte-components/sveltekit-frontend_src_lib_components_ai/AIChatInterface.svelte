<script lang="ts">
  interface Props {
    visible?: any;
    minimized?: any;
    draggable?: any;
    width?: any;
    height?: any;
    apiEndpoint?: any;
    fallbackEndpoint?: any;
    modelName?: any;
    title?: any;
    subtitle?: any;
  }
  let {
    visible = false,
    minimized = false,
    draggable = true,
    width = 400,
    height = 600,
    apiEndpoint = 'http://localhost:11434/api/generate',
    fallbackEndpoint = 'http://localhost:8000/v1/chat/completions',
    modelName = 'gemma3-legal:latest',
    title = 'YoRHa Legal AI',
    subtitle = 'Powered by Gemma3'
  } = $props();



	import { onMount, createEventDispatcher, tick } from 'svelte';
	import { fade, fly, scale } from 'svelte/transition';
	import { quintOut, elasticOut } from 'svelte/easing';
	import { writable } from 'svelte/store';

	// Types
	interface Message {
		id: string
		role: 'user' | 'assistant' | 'system';
		content: string
		timestamp: Date
		streaming?: boolean;
		error?: boolean;
	}

	interface ChatSettings {
		model: string
		temperature: number
		maxTokens: number
		topP: number
		systemPrompt: string
	}

	// Props
	 // false;
	let {
    minimized = $bindable()
  } = $props(); // true;
	let {
    width = $bindable()
  } = $props(); // 600;
	let {
    apiEndpoint = $bindable()
  } = $props(); // 'http://localhost:8000/v1/chat/completions';
	let {
    modelName = $bindable()
  } = $props(); // 'YoRHa Legal AI';
	let { subtitle = $bindable() } = $props(); // 'Powered by Gemma3';

	// State
	let messages: Message[] = [];
	let inputValue = '';
	let isTyping = false;
	let isConnected = true;
	let isDragging = false;
	let dragOffset = { x: 0, y: 0 };
	let position = { x: 0, y: 0 };
	let settingsOpen = false;

	// Settings
	let settings: ChatSettings = {
		model: modelName,
		temperature: 0.1,
		maxTokens: 512,
		topP: 0.9,
		systemPrompt: 'You are a specialized Legal AI Assistant powered by Gemma 3. You excel at contract analysis, legal research, and providing professional legal guidance.'
	};

	// Elements
	let chatContainer: HTMLDivElement
	let messagesContainer: HTMLDivElement
	let inputElement: HTMLTextAreaElement
	let windowElement: HTMLDivElement

	// Event dispatcher
	const dispatch = createEventDispatcher<{
		close: void
		minimize: void
		maximize: void
		message: { message: Message };
		settingsChange: { settings: ChatSettings };
	}>();

	// Initialize welcome message
	onMount(() => {
		// Progressive enhancement: Only run client-side features if available
		if (typeof window !== 'undefined') {
			addMessage('system', `Hello! I'm your YoRHa Legal AI Assistant powered by ${modelName}. I can help you with:

• Contract analysis and review
• Legal document interpretation
• Liability and risk assessment
• Compliance guidance
• Legal terminology explanation

How can I assist you with your legal needs today?`);

			// Set initial position (bottom-right corner)
			position = {
				x: window.innerWidth - width - 20,
				y: window.innerHeight - height - 20
			};
		}
	});

	// Auto-scroll to bottom when new messages arrive (Svelte 5)
	$effect(() => {
		if (messages.length > 0) {
			tick().then(() => {
				if (messagesContainer) {
					messagesContainer.scrollTop = messagesContainer.scrollHeight;
				}
			});
		}
	});

	// Add message to chat
	function addMessage(role: Message['role'], content: string, options: Partial<Message> = {}) {
		const message: Message = {
			id: crypto.randomUUID(),
			role,
			content,
			timestamp: new Date(),
			...options
		};

		messages = [...messages, message];
		dispatch('message', { message });
		return message;
	}

	// Update message content (for streaming)
	function updateMessage(id: string, content: string) {
		messages = messages.map(msg =>
			msg.id === id ? { ...msg, content } : msg
		);
	}

	// Send message to AI
	async function sendMessage() {
		if (!inputValue.trim() || isTyping) return;

		const userMessage = inputValue.trim();
		inputValue = '';

		// Add user message
		addMessage('user', userMessage);

		// Show typing indicator
		isTyping = true;
		const typingMessage = addMessage('assistant', '', { streaming: true });

		try {
			// Try primary endpoint first
			let response = await callGemma3API(userMessage);

			if (!response) {
				// Fallback to enhanced API
				response = await callFallbackAPI(userMessage);
			}

			if (response) {
				// Remove typing message and add real response
				messages = messages.filter(msg => msg.id !== typingMessage.id);
				addMessage('assistant', response);
			} else {
				throw new Error('No response from AI service');
			}

		} catch (error) {
			console.error('Chat error:', error);
			messages = messages.filter(msg => msg.id !== typingMessage.id);
			addMessage('assistant', 'Sorry, I\'m having trouble connecting. Please try again.', { error: true });
		} finally {
			isTyping = false;
		}
	}

	// Call Gemma3 API directly
	async function callGemma3API(message: string): Promise<string | null> {
		// Progressive enhancement: Check if fetch is available
		if (typeof fetch === 'undefined') {
			console.warn('Fetch API not available');
			return null;
		}

		try {
			// Create AbortController for timeout (with fallback)
			const controller = new AbortController();
			const timeoutId = setTimeout(() => controller.abort(), 60000);

			const response = await fetch(apiEndpoint, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: settings.model,
					prompt: formatPromptForGemma3(message),
					stream: false,
					options: {
						temperature: settings.temperature,
						num_predict: settings.maxTokens,
						top_p: settings.topP,
						top_k: 40,
						repeat_penalty: 1.1,
						stop: ['<start_of_turn>', '<end_of_turn>']
					}
				}),
				signal: controller.signal
			});

			clearTimeout(timeoutId);

			if (response.ok) {
				const data = await response.json();
				return data.response?.trim() || null;
			}
		} catch (error) {
			console.warn('Primary API failed:', error);
			isConnected = false; // Update connection status
		}

		return null;
	}

	// Call fallback API
	async function callFallbackAPI(message: string): Promise<string | null> {
		try {
			const response = await fetch(fallbackEndpoint, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					messages: [
						{ role: 'system', content: settings.systemPrompt },
						{ role: 'user', content: message }
					],
					max_tokens: settings.maxTokens,
					temperature: settings.temperature,
					top_p: settings.topP
				}),
				signal: AbortSignal.timeout(60000)
			});

			if (response.ok) {
				const data = await response.json();
				return data.response || data.choices?.[0]?.message?.content || null;
			}
		} catch (error) {
			console.warn('Fallback API failed:', error);
		}

		return null;
	}

	// Format prompt for Gemma3 template
	function formatPromptForGemma3(message: string): string {
		const conversation = messages
			.filter(msg => msg.role !== 'system' && !msg.error)
			.map(msg => {
				if (msg.role === 'user') {
					return `<start_of_turn>user\n${msg.content}<end_of_turn>`;
				} else {
					return `<start_of_turn>model\n${msg.content}<end_of_turn>`;
				}
			})
			.join('\n');

		return `${settings.systemPrompt}\n\n${conversation}\n<start_of_turn>user\n${message}<end_of_turn>\n<start_of_turn>model\n`;
	}

	// Handle input keydown
	function handleKeyDown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}

	// Auto-resize textarea
	function autoResize() {
		if (inputElement) {
			inputElement.style.height = 'auto';
			inputElement.style.height = Math.min(inputElement.scrollHeight, 120) + 'px';
		}
	}

	// Dragging functionality
	function startDrag(event: MouseEvent) {
		if (!draggable || event.target instanceof HTMLButtonElement) return;

		isDragging = true;
		const rect = windowElement.getBoundingClientRect();
		dragOffset = {
			x: event.clientX - rect.left,
			y: event.clientY - rect.top
		};

		document.addEventListener('mousemove', handleDrag);
		document.addEventListener('mouseup', stopDrag);
	}

	function handleDrag(event: MouseEvent) {
		if (!isDragging) return;

		const newX = event.clientX - dragOffset.x;
		const newY = event.clientY - dragOffset.y;

		// Constrain to viewport
		const maxX = window.innerWidth - width;
		const maxY = window.innerHeight - height;

		position = {
			x: Math.max(0, Math.min(newX, maxX)),
			y: Math.max(0, Math.min(newY, maxY))
		};
	}

	function stopDrag() {
		isDragging = false;
		document.removeEventListener('mousemove', handleDrag);
		document.removeEventListener('mouseup', stopDrag);
	}

	// Window controls
	function closeWindow() {
		visible = false;
		dispatch('close');
	}

	function minimizeWindow() {
		minimized = !minimized;
		dispatch(minimized ? 'minimize' : 'maximize');
	}

	// Clear chat
	function clearChat() {
		messages = [];
		addMessage('system', 'Chat cleared. How can I help you?');
	}

	// Toggle settings
	function toggleSettings() {
		settingsOpen = !settingsOpen;
	}

	// Update settings
	function updateSettings() {
		dispatch('settingsChange', { settings });
		settingsOpen = false;
	}

	// Format timestamp
	function formatTime(date: Date): string {
		return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
	}
</script>

{#if visible}
	<!-- Chat Window -->
	<aside
		bind:this={windowElement}
		role="dialog"
		aria-labelledby="chat-window-title"
		aria-describedby="chat-window-description"
		class="fixed bg-yorha-bg-secondary border-2 border-yorha-primary shadow-2xl z-50 flex flex-col overflow-hidden font-mono focus-within:ring-2 focus-within:ring-yorha-primary/50"
		class:opacity-50={isDragging}
		style="
			width: {width}px;
			height: {minimized ? 60 : height}px;
			left: {position.x}px;
			top: {position.y}px;
			transform: scale({isDragging ? 1.02 : 1});
		"
		in:scale={{ duration: 300, easing: elasticOut }}
		out:scale={{ duration: 200 }}
	>
		<!-- Floating Particles Background -->
		<div class="absolute inset-0 overflow-hidden pointer-events-none">
			{#each Array(5) as _, i}
				<div
					class="absolute w-1 h-1 bg-yorha-accent rounded-full opacity-60 animate-float"
					style="
						left: {10 + (i * 20)}%;
						animation-delay: {i * 0.8}s;
						animation-duration: {6 + (i * 2)}s;
					"
				></div>
			{/each}
		</div>

		<!-- Header -->
		<header
			class="flex items-center justify-between p-4 bg-gradient-to-r from-yorha-bg-tertiary to-yorha-bg-secondary border-b border-yorha-border cursor-move select-none relative"
			onmousedown={startDrag}
			role="banner"
		>
			<!-- Scan line effect -->
			<div class="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-yorha-primary to-transparent animate-scan"></div>

			<div class="flex items-center space-x-3">
				<!-- Avatar -->
				<div class="w-8 h-8 bg-gradient-to-br from-yorha-primary to-yorha-secondary flex items-center justify-center" role="img" aria-label="Legal AI Assistant">
					<svg class="w-4 h-4 text-yorha-bg-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
					</svg>
				</div>

				<!-- Title -->
				<div>
					<h1 id="chat-window-title" class="text-sm font-semibold text-yorha-text-primary">{title}</h1>
					<div id="chat-window-description" class="flex items-center space-x-2">
						<span class="text-xs text-yorha-text-secondary">{subtitle}</span>
						{#if isTyping}
							<div class="flex space-x-1" role="status" aria-live="polite" aria-label="AI is thinking">
								{#each Array(3) as _, i}
									<div class="w-1 h-1 bg-yorha-accent rounded-full animate-pulse" style="animation-delay: {i * 0.2}s" aria-hidden="true"></div>
								{/each}
							</div>
						{:else}
							<div class="flex items-center space-x-1" role="status" aria-live="polite">
								<div class="w-2 h-2 bg-yorha-success rounded-full animate-pulse" aria-hidden="true"></div>
								<span class="text-xs text-yorha-success">READY</span>
							</div>
						{/if}
					</div>
				</div>
			</div>

			<!-- Controls -->
			<nav class="flex items-center space-x-1" role="toolbar" aria-label="Chat window controls">
				<!-- Settings -->
				<button
					type="button"
					class="w-8 h-8 flex items-center justify-center border border-yorha-border text-yorha-text-secondary hover:border-yorha-primary hover:text-yorha-primary focus:border-yorha-primary focus:outline-none focus:ring-2 focus:ring-yorha-primary/50 transition-colors"
					onclick={toggleSettings}
					aria-label="{settingsOpen ? 'Close settings' : 'Open settings'}"
					aria-expanded={settingsOpen}
				>
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
					</svg>
				</button>

				<!-- Minimize -->
				<button
					type="button"
					class="w-8 h-8 flex items-center justify-center border border-yorha-border text-yorha-text-secondary hover:border-yorha-primary hover:text-yorha-primary focus:border-yorha-primary focus:outline-none focus:ring-2 focus:ring-yorha-primary/50 transition-colors"
					onclick={minimizeWindow}
					aria-label={minimized ? 'Restore window' : 'Minimize window'}
				>
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
						{#if minimized}
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 14l9-9 3 3L9 18l-4-4z" />
						{:else}
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4" />
						{/if}
					</svg>
				</button>

				<!-- Close -->
				<button
					type="button"
					class="w-8 h-8 flex items-center justify-center border border-yorha-border text-yorha-text-secondary hover:border-red-500 hover:text-red-500 focus:border-red-500 focus:outline-none focus:ring-2 focus:ring-red-500/50 transition-colors"
					onclick={closeWindow}
					aria-label="Close chat window"
				>
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
					</svg>
				</button>
			</nav>
		</header>

		{#if !minimized}
			<!-- Settings Panel -->
			{#if settingsOpen}
				<div class="border-b border-yorha-border bg-yorha-bg-primary p-4" transitionfly={{ y: -50, duration: 200 }}>
					<div class="space-y-3">
						<!-- Model Selection -->
						<div>
							<label class="block text-xs text-yorha-text-secondary mb-1">Model</label>
							<select bind:value={settings.model} class="w-full bg-yorha-bg-tertiary border border-yorha-border text-yorha-text-primary text-xs p-2 focus:border-yorha-primary">
								<option value="gemma3-legal:latest">Gemma3 Legal (11.8B)</option>
								<option value="llama3.2:1b">Llama3.2 (1B)</option>
							</select>
						</div>

						<!-- Temperature -->
						<div>
							<label class="block text-xs text-yorha-text-secondary mb-1">Temperature: {settings.temperature}</label>
							<input type="range" min="0" max="1" step="0.1" bind:value={settings.temperature} class="w-full">
						</div>

						<!-- Max Tokens -->
						<div>
							<label class="block text-xs text-yorha-text-secondary mb-1">Max Tokens</label>
							<input type="number" min="100" max="2048" bind:value={settings.maxTokens} class="w-full bg-yorha-bg-tertiary border border-yorha-border text-yorha-text-primary text-xs p-2 focus:border-yorha-primary">
						</div>

						<!-- Buttons -->
						<div class="flex space-x-2">
							<button type="button" onclick={updateSettings} class="flex-1 bg-yorha-primary text-yorha-bg-primary text-xs p-2 hover:bg-yorha-secondary transition-colors">
								Apply
							</button>
							<button type="button" onclick={clearChat} class="flex-1 bg-yorha-error text-white text-xs p-2 hover:bg-red-600 transition-colors">
								Clear
							</button>
						</div>
					</div>
				</div>
			{/if}

			<!-- Messages Container -->
			<main bind:this={messagesContainer} class="flex-1 overflow-y-auto p-4 bg-yorha-bg-primary space-y-4" role="log" aria-live="polite" aria-label="Chat conversation">
				{#each messages as message (message.id)}
					<article class:flex class:justify-end={message.role === 'user'} class:justify-start={message.role !== 'user'} in:fly={{ y: 20, duration: 300 }}>
						<div
							class="max-w-[85%] border border-yorha-border p-3 relative shadow-sm"
							class:bg-yorha-bg-tertiary={message.role === 'user'}
							class:border-yorha-primary={message.role === 'user'}
							class:bg-yorha-bg-secondary={message.role !== 'user'}
							role={message.role === 'system' ? 'status' : 'article'}
						>
							{#if message.role === 'assistant'}
								<div class="absolute left-0 top-0 bottom-0 w-1 bg-yorha-accent"></div>
							{/if}

							<div class="text-sm text-yorha-text-primary whitespace-pre-wrap leading-relaxed">
								<span class="sr-only">{message.role === 'user' ? 'You said:' : 'AI responded:'}</span>
								{message.content}
							</div>

							{#if message.error}
								<div class="mt-2 text-xs text-red-400" role="alert">
									Failed to get response. <button onclick={sendMessage} class="underline hover:no-underline focus:outline-none focus:ring-2 focus:ring-red-400/50" aria-label="Retry sending message">Retry</button>
								</div>
							{/if}

							<time class="mt-2 text-xs text-yorha-text-muted" datetime={message.timestamp.toISOString()}>
								{formatTime(message.timestamp)}
							</time>
						</div>
					</article>
				{/each}

				{#if isTyping}
					<div class="flex justify-start" in:fade role="status" aria-live="polite">
						<div class="bg-yorha-bg-secondary border border-yorha-border p-3 relative shadow-sm">
							<div class="absolute left-0 top-0 bottom-0 w-1 bg-yorha-accent"></div>
							<div class="flex items-center space-x-2">
								<div class="flex space-x-1" aria-hidden="true">
									{#each Array(3) as _, i}
										<div class="w-2 h-2 bg-yorha-accent rounded-full animate-bounce" style="animation-delay: {i * 0.1}s"></div>
									{/each}
								</div>
								<span class="text-xs text-yorha-text-muted">AI is thinking...</span>
							</div>
						</div>
					</div>
				{/if}
			</main>

			<!-- Input Area -->
			<footer class="border-t border-yorha-border bg-yorha-bg-secondary p-4">
				<form class="flex space-x-3" on:submit|preventDefault={sendMessage} role="search" aria-label="Send message to AI">
					<textarea
						bind:this={inputElement}
						bind:value={inputValue}
						onkeydown={handleKeyDown}
						oninput={autoResize}
						placeholder="Ask me about contracts, liability, compliance, or any legal question..."
						class="flex-1 bg-yorha-bg-tertiary border border-yorha-border text-yorha-text-primary placeholder-yorha-text-muted p-3 text-sm resize-none focus:border-yorha-primary focus:outline-none focus:ring-2 focus:ring-yorha-primary/50 transition-colors"
						rows="1"
						style="min-height: 40px; max-height: 120px;"
						disabled={isTyping}
						aria-label="Type your legal question here"
						required
					></textarea>

					<button
						type="submit"
						disabled={!inputValue.trim() || isTyping}
						class="w-10 h-10 bg-yorha-primary text-yorha-bg-primary flex items-center justify-center hover:bg-yorha-secondary focus:outline-none focus:ring-2 focus:ring-yorha-primary/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
						aria-label={isTyping ? 'AI is responding' : 'Send message'}
					>
						<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
						</svg>
					</button>
				</form>
				<div class="flex justify-between items-center mt-2 text-xs text-yorha-text-muted" role="status">
					<span>Powered by {settings.model}</span>
					<div class="flex items-center space-x-1">
						<div class="w-2 h-2 rounded-full" class:bg-yorha-success={isConnected} class:bg-yorha-error={!isConnected} aria-hidden="true"></div>
						<span class="sr-only">Connection status:</span>
						<span>{isConnected ? 'Connected' : 'Disconnected'}</span>
					</div>
				</div>
			</footer>
		{/if}
	</aside>
{/if}

<style>
	@keyframes float {
		0%, 100% {
			transform: translateY(0) rotate(0deg);
			opacity: 0;
		}
		10% {
			opacity: 1;
		}
		90% {
			opacity: 1;
		}
		100% {
			transform: translateY(-100%) rotate(360deg);
			opacity: 0;
		}
	}

	@keyframes scan {
		0% {
			transform: translateX(-100%);
		}
		100% {
			transform: translateX(100%);
		}
	}

	.animate-float {
		animation: float linear infinite;
	}

	.animate-scan {
		animation: scan 3s linear infinite;
	}
</style>
