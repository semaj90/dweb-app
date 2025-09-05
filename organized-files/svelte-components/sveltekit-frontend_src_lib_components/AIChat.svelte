<script lang="ts">
	import { useMachine } from '@xstate/svelte';
	import { onMount } from 'svelte';
	import { chatMachine } from '$lib/machines/chatMachine';

	let chatContainer;
	let userInput = '';

	const { state, send } = useMachine(chatMachine, {
		actors: {
			streamChatActor: ({ input }) => (sendBack, receive) => {
				const controller = new AbortController();

				async function stream() {
					try {
						const response = await fetch('/api/chat', {
							method: 'POST',
							headers: { 'Content-Type': 'application/json' },
							body: JSON.stringify({ messages: input.messages }),
							signal: controller.signal
						});

						if (!response.ok || !response.body) {
							throw new Error('Failed to get response stream.');
						}

						const reader = response.body.getReader();
						const decoder = new TextDecoder();

						while (true) {
							const { done, value } = await reader.read();
							if (done) break;

							const chunk = decoder.decode(value, { stream: true });
							const lines = chunk.split('\n').filter(line => line.trim() !== '');

							for (const line of lines) {
								try {
									const jsonResponse = JSON.parse(line);
									if (jsonResponse.message && jsonResponse.message.content) {
										sendBack({ type: 'STREAM_CHUNK', chunk: jsonResponse.message.content });
									}
								} catch (e) {
									// Ignore parsing errors for incomplete chunks
								}
							}
						}
						sendBack({ type: 'STREAM_DONE' });
					} catch (error) {
						if (error.name !== 'AbortError') {
							console.error('Chat stream error:', error);
							sendBack({ type: 'error', data: error });
						}
					}
				}

				stream();

				return () => {
					controller.abort();
				};
			}
		}
	});

	function handleSubmit() {
		if (!userInput.trim()) return;
		send({ type: 'SUBMIT', message: userInput });
		userInput = '';
	}

	// Reactive statement to scroll down when messages change
	$effect(() => { if ($state.context.messages && typeof window !== 'undefined') {
		// Use a microtask to wait for the DOM to update
		Promise.resolve().then(() => {
			if (chatContainer) {
				chatContainer.scrollTop = chatContainer.scrollHeight;
			}
		});
	}
</script>

<div class="flex flex-col h-[calc(100vh-4rem)] bg-gray-50 dark:bg-gray-900">
	<div bind:this={chatContainer} class="flex-1 overflow-y-auto p-4 space-y-4">
		{#each $state.context.messages as message, i (i)}
			<div class="chat-message {message.role === 'user' ? 'user' : 'assistant'}">
				<div class="message-bubble">
					{@html message.content.replace(/\n/g, '<br>')}
					{#if $state.matches('loading') && i === $state.context.messages.length - 1}
						<span class="typing-indicator"></span>
					{/if}
				</div>
			</div>
		{/each}
		{#if $state.matches('error')}
			<div class="chat-message assistant">
				<div class="message-bubble error-bubble">
					<p>Sorry, an error occurred: {$state.context.error.message}</p>
					<p>Please try again.</p>
				</div>
			</div>
		{/if}
	</div>

	<div class="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
		<form on:submit|preventDefault={handleSubmit} class="flex items-center space-x-2">
			<input
				type="text"
				bind:value={userInput}
				placeholder="Ask about your case..."
				class="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				disabled={$state.matches('loading')}
			/>
			<button
				type="submit"
				class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
				disabled={$state.matches('loading') || !userInput.trim()}
			>
				Send
			</button>
		</form>
	</div>
</div>

<style>
	/* Styles from previous Chat.svelte component can be reused here */
	.chat-message { display: flex max-width: 80%; }
	.chat-message.user { margin-left: auto flex-direction: row-reverse; }
	.chat-message.assistant { margin-right: auto }
	.message-bubble { padding: 0.75rem 1rem; border-radius: 1.25rem; word-wrap: break-word; position: relative }
	.user .message-bubble { background-color: #2563eb; color: white border-bottom-right-radius: 0.25rem; }
	.assistant .message-bubble { background-color: #e5e7eb; color: #111827; border-bottom-left-radius: 0.25rem; }
	.dark .assistant .message-bubble { background-color: #374151; color: #f9fafb; }
	.error-bubble { background-color: #fef2f2; color: #991b1b; }
	.dark .error-bubble { background-color: #4c1d1d; color: #fca5a5; }
	.typing-indicator { display: inline-block; width: 5px; height: 5px; border-radius: 50%; background-color: currentColor animation: typing 1s infinite steps(4, end); margin-left: 8px; vertical-align: bottom }
	@keyframes typing { to { transform: translateY(-0.25rem); } }
</style>
