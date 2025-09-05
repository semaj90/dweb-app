<script lang="ts">
	// Web Speech API type declarations
	interface SpeechRecognition extends EventTarget {
		continuous: boolean
		interimResults: boolean
		lang: string
		start(): void;
		stop(): void;
		abort(): void;
		onresult: (event: SpeechRecognitionEvent) => void;
		onerror: (event: SpeechRecognitionErrorEvent) => void;
		onend: () => void;
	}

	interface SpeechRecognitionEvent extends Event {
		results: SpeechRecognitionResultList
	}

	interface SpeechRecognitionResultList {
		[index: number]: SpeechRecognitionResult;
		length: number
	}

	interface SpeechRecognitionResult {
		[index: number]: SpeechRecognitionAlternative;
		isFinal: boolean
	}

	interface SpeechRecognitionAlternative {
		transcript: string
		confidence: number
	}

	interface SpeechRecognitionErrorEvent extends Event {
		error: string
	}

	declare global {
		interface Window {
			SpeechRecognition?: typeof SpeechRecognition;
			webkitSpeechRecognition?: typeof SpeechRecognition;
		}
	}

	import { onMount, tick } from 'svelte';
	import { writable } from 'svelte/store';
	import { createMachine, interpret } from 'xstate';
	import { Dialog, Tabs, Resizable } from 'bits-ui';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import GoldenRatioLoader from '$lib/components/ui/enhanced-bits/GoldenRatioLoader.svelte';
	import {
		MessageSquare,
		FileText,
		BookOpen,
		Brain,
		Zap,
		Send,
		Mic,
		MicOff,
		Settings,
		Expand,
		Minimize,
		RefreshCw,
		Sparkles,
		Target,
		History,
		Save,
		Download
	} from 'lucide-svelte';

	// Type definitions for panel layout
	interface PanelConfig {
		width: number
		collapsed: boolean
	}

	interface PanelLayout {
		reports: PanelConfig
		summaries: PanelConfig
		citations: PanelConfig
		chat: PanelConfig
	}

	// 4-Panel Layout State
	let panelLayout = $state<PanelLayout>({
		reports: { width: 25, collapsed: false },
		summaries: { width: 25, collapsed: false },
		citations: { width: 25, collapsed: false },
		chat: { width: 25, collapsed: false }
	});

	// AI Assistant State
	let chatMessages = $state<ChatMessage[]>([]);
	let currentMessage = $state('');
	let isProcessing = $state(false);
	let voiceEnabled = $state(false);
	let isListening = $state(false);

	// XState Machine for AI Assistant
	const aiAssistantMachine = createMachine({
		id: 'aiAssistant',
		initial: 'idle',
		states: {
			idle: {
				on: {
					START_CHAT: 'processing',
					VOICE_INPUT: 'listening',
					CONTEXT_SWITCH: 'contextSwitching'
				}
			},
			listening: {
				on: {
					VOICE_COMPLETE: 'processing',
					VOICE_CANCEL: 'idle'
				}
			},
			processing: {
				on: {
					RESPONSE_READY: 'responding',
					ERROR: 'error'
				}
			},
			responding: {
				on: {
					RESPONSE_COMPLETE: 'idle',
					STREAM_UPDATE: 'responding'
				}
			},
			contextSwitching: {
				on: {
					CONTEXT_LOADED: 'idle',
					ERROR: 'error'
				}
			},
			error: {
				on: {
					RETRY: 'idle',
					RESET: 'idle'
				}
			}
		}
	});

	let aiService = interpret(aiAssistantMachine);

	interface ChatMessage {
		id: string
		type: 'user' | 'assistant' | 'system';
		content: string
		timestamp: Date
		context?: any;
		suggestions?: string[];
		metadata?: any;
	}

	interface Report {
		id: string
		title: string
		type: 'analysis' | 'summary' | 'research';
		date: Date
		summary: string
		status: 'draft' | 'completed' | 'archived';
	}

	interface Citation {
		id: string
		title: string
		source: string
		date: Date
		relevance: number
		tags: string[];
	}

	// Data stores
	let reports = $state<Report[]>([]);
	let summaries = $state<any[]>([]);
	let citations = $state<Citation[]>([]);
	let chatHistory = $state<any[]>([]);

	// AI Suggestions
	let aiSuggestions = $state<string[]>([]);
	let contextualSuggestions = $state<string[]>([]);

	// Melt UI Components
	const {
		elements: { root: tabsRoot, list: tabsList, content: tabsContent, trigger: tabsTrigger }
	} = createTabs({
		defaultValue: 'reports'
	});

	const {
		elements: { trigger: settingsDialogTrigger, overlay, content: dialogContent, close }
	} = createDialog();

	onMount(async () => {
		aiService.start();
		await loadInitialData();
		initializeWebSpeech();
		startRealTimeUpdates();
	});

	async function loadInitialData() {
		try {
			// Load reports
			const reportsResponse = await fetch('/api/reports');
			if (reportsResponse.ok) {
				reports = await reportsResponse.json();
			}

			// Load summaries
			const summariesResponse = await fetch('/api/summaries');
			if (summariesResponse.ok) {
				summaries = await summariesResponse.json();
			}

			// Load citations
			const citationsResponse = await fetch('/api/citations');
			if (citationsResponse.ok) {
				citations = await citationsResponse.json();
			}

			// Load chat history
			const historyResponse = await fetch('/api/chat/history');
			if (historyResponse.ok) {
				chatHistory = await historyResponse.json();
			}

		} catch (error) {
			console.error('Failed to load initial data:', error);
		}
	}

	function initializeWebSpeech() {
		if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
			voiceEnabled = true;
		}
	}

	async function sendMessage() {
		if (!currentMessage.trim()) return;

		const userMessage: ChatMessage = {
			id: `msg_${Date.now()}`,
			type: 'user',
			content: currentMessage,
			timestamp: new Date()
		};

		chatMessages = [...chatMessages, userMessage];
		const messageToSend = currentMessage;
		currentMessage = '';

		isProcessing = true;
		aiService.send('START_CHAT');

		try {
			const response = await fetch('/api/ai/chat', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					message: messageToSend,
					context: getRelevantContext(),
					chatHistory: chatMessages.slice(-10)
				})
			});

			if (response.ok) {
				const data = await response.json();

				const assistantMessage: ChatMessage = {
					id: `msg_${Date.now()}_ai`,
					type: 'assistant',
					content: data.response,
					timestamp: new Date(),
					suggestions: data.suggestions || [],
					metadata: data.metadata
				};

				chatMessages = [...chatMessages, assistantMessage];
				aiSuggestions = data.suggestions || [];

				aiService.send('RESPONSE_COMPLETE');

			} else {
				throw new Error('Chat request failed');
			}

		} catch (error) {
			console.error('Chat error:', error);

			const errorMessage: ChatMessage = {
				id: `msg_${Date.now()}_error`,
				type: 'system',
				content: 'Sorry, I encountered an error. Please try again.',
				timestamp: new Date()
			};

			chatMessages = [...chatMessages, errorMessage];
			aiService.send('ERROR');
		} finally {
			isProcessing = false;
		}
	}

	function getRelevantContext() {
		return {
			recentReports: reports.slice(0, 3),
			recentSummaries: summaries.slice(0, 3),
			topCitations: citations.sort((a, b) => b.relevance - a.relevance).slice(0, 5)
		};
	}

	async function startVoiceInput() {
		if (!voiceEnabled) return;

		try {
			const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
			if (!SpeechRecognitionAPI) {
				throw new Error('Speech recognition not supported');
			}
			const recognition = new SpeechRecognitionAPI();

			recognition.continuous = false;
			recognition.interimResults = false;
			recognition.lang = 'en-US';

			isListening = true;
			aiService.send('VOICE_INPUT');

			recognition.onresult = (event) => {
				const transcript = event.results[0][0].transcript;
				currentMessage = transcript;
				isListening = false;
				aiService.send('VOICE_COMPLETE');
				sendMessage();
			};

			recognition.onerror = () => {
				isListening = false;
				aiService.send('VOICE_CANCEL');
			};

			recognition.start();

		} catch (error) {
			console.error('Voice input error:', error);
			isListening = false;
		}
	}

	function startRealTimeUpdates() {
		// Generate contextual suggestions based on current context
		setInterval(() => {
			generateContextualSuggestions();
		}, 5000);
	}

	function generateContextualSuggestions() {
		const suggestions = [
			"Summarize the latest evidence",
			"Find similar legal cases",
			"Generate prosecution timeline",
			"Analyze contract vulnerabilities",
			"Compare case precedents"
		];

		contextualSuggestions = suggestions.slice(0, 3);
	}

	async function switchContext(contextId: string) {
		aiService.send('CONTEXT_SWITCH');

		try {
			const response = await fetch(`/api/context/${contextId}`);
			if (response.ok) {
				const contextData = await response.json();

				// Update relevant panels with new context
				// This would load specific reports, summaries, etc.

				aiService.send('CONTEXT_LOADED');
			}
		} catch (error) {
			console.error('Context switch error:', error);
			aiService.send('ERROR');
		}
	}

	function togglePanel(panelName: keyof PanelLayout) {
		panelLayout[panelName].collapsed = !panelLayout[panelName].collapsed;
	}

	function adjustPanelWidth(panelName: keyof PanelLayout, delta: number) {
		const current = panelLayout[panelName].width;
		const newWidth = Math.max(15, Math.min(50, current + delta));
		panelLayout[panelName].width = newWidth;

		// Redistribute remaining width
		const remaining = 100 - newWidth;
		const otherPanels = Object.keys(panelLayout).filter(p => p !== panelName);
		otherPanels.forEach(panel => {
			panelLayout[panel].width = remaining / otherPanels.length;
		});
	}

	async function exportChatHistory() {
		const data = {
			messages: chatMessages,
			timestamp: new Date(),
			context: getRelevantContext()
		};

		const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `chat-history-${Date.now()}.json`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function useSuggestion(suggestion: string) {
		currentMessage = suggestion;
		sendMessage();
	}
</script>

<svelte:head>
	<title>AI Assistant</title>
</svelte:head>

<div class="h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col">
	<!-- Header -->
	<header class="bg-white border-b border-slate-200 p-4">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-4">
				<h1 class="text-2xl font-bold text-slate-900">🤖 AI Assistant</h1>
				<Badge variant="outline" class="text-green-600 border-green-600">
					Context7 Active
				</Badge>
			</div>

			<div class="flex items-center space-x-2">			<Button variant="outline" size="sm" onclick={exportChatHistory}>
				<Download class="h-4 w-4 mr-1" />
				Export
			</Button>
			<button class="btn-outline btn-sm" use:settingsDialogTrigger>
				<Settings class="h-4 w-4" />
			</button>
			</div>
		</div>
	</header>

	<!-- 4-Panel Layout -->
	<div class="flex-1 flex">
		<!-- Reports Panel -->
		<div
			class="border-r border-slate-200 bg-white transition-all duration-300"
			style:width="{panelLayout.reports.collapsed ? '0px' : `${panelLayout.reports.width}%`}"
			style:min-width="{panelLayout.reports.collapsed ? '0px' : '250px'}"
		>
			{#if !panelLayout.reports.collapsed}
				<div class="h-full flex flex-col">
					<div class="p-4 border-b border-slate-200 flex items-center justify-between">
						<h3 class="font-semibold text-slate-900">Reports</h3>
						<div class="flex items-center space-x-1">
							<Button size="sm" variant="ghost" onclick={() => adjustPanelWidth('reports', -5)}>
								<Minimize class="h-3 w-3" />
							</Button>
							<Button size="sm" variant="ghost" onclick={() => togglePanel('reports')}>
								<Minimize class="h-3 w-3" />
							</Button>
						</div>
					</div>

					<div class="flex-1 overflow-y-auto p-4">
						{#if reports.length === 0}
							<div class="text-center py-8">
								<FileText class="h-12 w-12 text-slate-300 mx-auto mb-2" />
								<p class="text-slate-500 text-sm">No reports yet</p>
							</div>
						{:else}
							<div class="space-y-3">
								{#each reports as report}
									<div class="p-3 border border-slate-200 rounded-lg hover:shadow-sm cursor-pointer">
										<h4 class="font-medium text-slate-900 text-sm mb-1">{report.title}</h4>
										<p class="text-xs text-slate-600 mb-2">{report.type} • {new Date(report.date).toLocaleDateString()}</p>
										<p class="text-xs text-slate-700 line-clamp-2">{report.summary}</p>
										<Badge class="mt-2 text-xs" variant={report.status === 'completed' ? 'default' : 'outline'}>
											{report.status}
										</Badge>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				</div>
			{/if}
		</div>

		<!-- Summaries Panel -->
		<div
			class="border-r border-slate-200 bg-white transition-all duration-300"
			style:width="{panelLayout.summaries.collapsed ? '0px' : `${panelLayout.summaries.width}%`}"
			style:min-width="{panelLayout.summaries.collapsed ? '0px' : '250px'}"
		>
			{#if !panelLayout.summaries.collapsed}
				<div class="h-full flex flex-col">
					<div class="p-4 border-b border-slate-200 flex items-center justify-between">
						<h3 class="font-semibold text-slate-900">Summaries</h3>
						<Button size="sm" variant="ghost" onclick={() => togglePanel('summaries')}>
							<Minimize class="h-3 w-3" />
						</Button>
					</div>

					<div class="flex-1 overflow-y-auto p-4">
						<div class="space-y-3">
							{#each summaries as summary}
								<div class="p-3 border border-slate-200 rounded-lg">
									<h4 class="font-medium text-slate-900 text-sm mb-1">{summary.title}</h4>
									<p class="text-xs text-slate-700">{summary.content}</p>
								</div>
							{/each}
						</div>
					</div>
				</div>
			{/if}
		</div>

		<!-- Citations Panel -->
		<div
			class="border-r border-slate-200 bg-white transition-all duration-300"
			style:width="{panelLayout.citations.collapsed ? '0px' : `${panelLayout.citations.width}%`}"
			style:min-width="{panelLayout.citations.collapsed ? '0px' : '250px'}"
		>
			{#if !panelLayout.citations.collapsed}
				<div class="h-full flex flex-col">
					<div class="p-4 border-b border-slate-200 flex items-center justify-between">
						<h3 class="font-semibold text-slate-900">Citations</h3>
						<Button size="sm" variant="ghost" onclick={() => togglePanel('citations')}>
							<Minimize class="h-3 w-3" />
						</Button>
					</div>

					<div class="flex-1 overflow-y-auto p-4">
						<div class="space-y-3">
							{#each citations as citation}
								<div class="p-3 border border-slate-200 rounded-lg">
									<h4 class="font-medium text-slate-900 text-sm mb-1">{citation.title}</h4>
									<p class="text-xs text-slate-600 mb-1">{citation.source}</p>
									<div class="flex justify-between items-center">
										<span class="text-xs text-slate-500">{new Date(citation.date).toLocaleDateString()}</span>
										<Badge variant="outline" class="text-xs">
											{(citation.relevance * 100).toFixed(0)}% relevant
										</Badge>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			{/if}
		</div>

		<!-- AI Chat Panel -->
		<div
			class="bg-white transition-all duration-300 flex flex-col"
			style:width="{panelLayout.chat.collapsed ? '0px' : `${panelLayout.chat.width}%`}"
		>
			{#if !panelLayout.chat.collapsed}
				<!-- Chat Header -->
				<div class="p-4 border-b border-slate-200 flex items-center justify-between">
					<div class="flex items-center space-x-2">
						<Brain class="h-5 w-5 text-blue-500" />
						<h3 class="font-semibold text-slate-900">AI Assistant</h3>
						{#if isProcessing}
							<div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
						{/if}
					</div>
					<Button size="sm" variant="ghost" onclick={() => togglePanel('chat')}>
						<Expand class="h-3 w-3" />
					</Button>
				</div>

				<!-- AI Suggestions Bar -->
				{#if contextualSuggestions.length > 0}
					<div class="p-3 bg-blue-50 border-b border-blue-200">
						<p class="text-xs text-blue-700 mb-2">💡 Suggested actions:</p>
						<div class="flex flex-wrap gap-2">
							{#each contextualSuggestions as suggestion}
								<button
									class="text-xs px-2 py-1 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded transition-colors"
									onclick={() => useSuggestion(suggestion)}
								>
									{suggestion}
								</button>
							{/each}
						</div>
					</div>
				{/if}

				<!-- Chat Messages -->
				<div class="flex-1 overflow-y-auto p-4 space-y-4">
					{#if chatMessages.length === 0}
						<div class="text-center py-12">
							<MessageSquare class="h-16 w-16 text-slate-300 mx-auto mb-4" />
							<p class="text-slate-500">Start a conversation with your AI assistant</p>
							<p class="text-slate-400 text-sm mt-1">Ask questions about your cases, evidence, or legal research</p>
						</div>
					{:else}
						{#each chatMessages as message}
							<div class="flex {message.type === 'user' ? 'justify-end' : 'justify-start'}">
								<div class="max-w-[80%] {message.type === 'user' ? 'bg-blue-500 text-white' : message.type === 'assistant' ? 'bg-slate-100 text-slate-900' : 'bg-yellow-100 text-yellow-800'} rounded-lg p-3">
									<p class="text-sm">{message.content}</p>
									<p class="text-xs opacity-75 mt-1">
										{new Date(message.timestamp).toLocaleTimeString()}
									</p>

									{#if message.suggestions && message.suggestions.length > 0}
										<div class="mt-2 space-y-1">
											{#each message.suggestions as suggestion}
												<button
													class="block w-full text-left text-xs p-2 bg-white/20 hover:bg-white/30 rounded transition-colors"
													onclick={() => useSuggestion(suggestion)}
												>
													{suggestion}
												</button>
											{/each}
										</div>
									{/if}
								</div>
							</div>
						{/each}
					{/if}

					{#if isProcessing}
						<div class="flex justify-start">
							<div class="bg-slate-100 rounded-lg p-3">
								<div class="flex items-center space-x-2">
									<div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
									<div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
									<div class="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
									<span class="text-sm text-slate-600">AI is thinking...</span>
								</div>
							</div>
						</div>
					{/if}
				</div>

				<!-- Chat Input -->
				<div class="p-4 border-t border-slate-200">
					<div class="flex space-x-2">
						<div class="flex-1">
							<Input
								bind:value={currentMessage}
								placeholder="Ask your AI assistant..."
								onkeydown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
								disabled={isProcessing}
							/>
						</div>

						{#if voiceEnabled}
							<Button
								variant="outline"
								size="sm"
								onclick={startVoiceInput}
								disabled={isListening || isProcessing}
								class="{isListening ? 'bg-red-100 border-red-300' : ''}"
							>
								{#if isListening}
									<MicOff class="h-4 w-4" />
								{:else}
									<Mic class="h-4 w-4" />
								{/if}
							</Button>
						{/if}

						<Button
							onclick={sendMessage}
							disabled={!currentMessage.trim() || isProcessing}
						>
							<Send class="h-4 w-4" />
						</Button>
					</div>
				</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.line-clamp-2 {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical
		overflow: hidden
	}
</style>