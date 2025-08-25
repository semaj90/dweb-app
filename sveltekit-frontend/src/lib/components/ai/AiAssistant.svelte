<script lang="ts">
	import { getContext } from 'svelte';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	const AnyButton: any = Button;
	import { Badge } from '$lib/components/ui/badge';
	import {
		Bot,
		Sparkles,
		Save,
		RefreshCw,
		FileText,
		AlertCircle,
		User,
		Database
	} from 'lucide-svelte';

	// XState & Loki imports
	import Loki from 'lokijs';
	import type { Collection } from 'lokijs';
	import { createMachine, assign } from 'xstate';
	import { useMachine } from '@xstate/svelte';

	// Feedback Integration
	import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';

	// Get user from context (set in +layout.svelte)
	const getUser = getContext('user');
	const user = typeof getUser === 'function' ? getUser() : undefined;

	export let contextItems: any[] = [];
	export let caseId: string = '';

	// --- Client-Side Caching with Loki.js ---
	// Initializes a simple in-memory DB to cache summaries on the client.
	// Ensure Loki.js DB and collection are initialized only once (singleton pattern).
	let db: Loki;
	let lokiSummaryCache: Collection<any>;

	function getSummaryCache() {
		if (!db) {
			db = new Loki('ai-cache.db');
			lokiSummaryCache = db.addCollection('summaries', { indices: ['caseId'] });
		} else if (!lokiSummaryCache) {
			lokiSummaryCache = db.getCollection('summaries') || db.addCollection('summaries', { indices: ['caseId'] });
		}
		return lokiSummaryCache;
	}
	// initialize
	const summaryCacheCollection = getSummaryCache();

	// Component state (some are synced to XState below)
	let summary = '';
	let error = '';
	let isLoading = false;
	let isSaving = false;
	let retryCount = 0;
	let stream = '';
	let enableStreaming = false; // set true if you wire streaming
	let showSources = true;
	let sources: any[] = [];

	// Derived booleans used in template
	$: hasContent = contextItems.length > 0;
	$: canSummarize = hasContent && !!user && !isLoading;
	$: allowSave = true;
	$: canSave = !!summary && !!user && !isSaving;

	// Feedback integration variables
	let feedbackIntegration: any;
	let currentInteractionId: string | null = null;

	const getStatusInfo = () => {
		if (isLoading) {
			return { icon: Bot, text: 'Analyzing...', color: 'text-blue-600' };
		}
		if (error) {
			return { icon: AlertCircle, text: 'Error', color: 'text-red-600' };
		}
		if (summary) {
			return { icon: FileText, text: 'Summary ready', color: 'text-green-600' };
		}
		return null;
	};

	async function fetchSummaryFromServer(payload: { caseId: string; evidence: any[]; userId?: string }) {
		// Try Enhanced RAG service first, fallback to SvelteKit API endpoints
		try {
			// Try multiple Enhanced RAG endpoints
			let ragResult;
			
			// Try health check first
			const healthRes = await fetch('http://localhost:8094/health');
			if (!healthRes.ok) {
				throw new Error('Enhanced RAG service not available');
			}
			
			// Since direct GPU endpoint not available, use SvelteKit API proxy
			const res = await fetch('/api/ai/analyze-evidence', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					evidenceId: payload.caseId + '-batch',
					content: JSON.stringify(payload.evidence),
					forceReanalyze: true
				})
			});
			
			if (!res.ok) {
				// Fallback to enhanced processing endpoint
				const fallbackRes = await fetch('/api/ai/process-enhanced', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						evidence: payload.evidence,
						options: {
							analysisType: 'summary',
							model: 'gemma3-legal:latest',
							caseId: payload.caseId,
							userId: payload.userId || user?.id
						}
					})
				});
				
				if (!fallbackRes.ok) {
					throw new Error('All AI services unavailable');
				}
				ragResult = await fallbackRes.json();
			} else {
				ragResult = await res.json();
			}
			
			return {
				summary: ragResult.summary || ragResult.analysis || ragResult.result || 'Analysis completed',
				sources: ragResult.sources || [],
				confidence: ragResult.confidence || 0.85
			};
		} catch (error) {
			console.warn('AI service error:', error);
			throw error;
		}
	}

	// --- XState Machine for AI Processing ---
	const aiProcessingMachine = createMachine(
		{
			id: 'aiProcessing',
			initial: 'idle',
			context: {
				caseId: '',
				evidence: [] as any[],
				userId: user?.id || '',
				summary: '',
				error: ''
			},
			states: {
				idle: {
					on: {
						PROCESS: {
							target: 'checkingCache',
							actions: assign({
								caseId: (_, event: any) => event.caseId,
								evidence: (_, event: any) => event.evidence,
								userId: (_, event: any) => event.userId
							})
						}
					}
				},
				checkingCache: {
					always: [
						{
							target: 'success',
							cond: 'isSummaryInCache',
							actions: 'loadSummaryFromCache'
						},
						{ target: 'processing' }
					]
				},
				processing: {
					invoke: {
						src: 'processEvidenceOnServer',
						onDone: {
							target: 'success',
							actions: assign({
								summary: (_, event: any) => event.data.summary ?? ''
							})
						},
						onError: {
							target: 'failure',
							actions: assign({
								error: (_, event: any) =>
									event.data?.message || event.data?.message || 'An unexpected error occurred.'
							})
						}
					}
				},
				success: {
					entry: 'cacheSummary',
					on: { PROCESS: 'checkingCache' }
				},
				failure: {
					on: { PROCESS: 'checkingCache' }
				}
			}
		},
		{
			actions: {
				loadSummaryFromCache: assign({
					summary: (context) => {
						const cached = summaryCacheCollection.findOne({ caseId: context.caseId });
						return cached?.summary || '';
					}
				}),
				cacheSummary: (context) => {
					if (context.summary && !summaryCacheCollection.findOne({ caseId: context.caseId })) {
						summaryCacheCollection.insert({ caseId: context.caseId, summary: context.summary });
					}
				}
			},
			guards: {
				isSummaryInCache: () => {
					return !!summaryCacheCollection.findOne({ caseId });
				}
			},
			services: {
				processEvidenceOnServer: async (context) => {
					// Use the same resilient approach as fetchSummaryFromServer
					return await fetchSummaryFromServer({
						caseId: context.caseId,
						evidence: context.evidence,
						userId: context.userId
					});
				}
			}
		}
	);

	const { state, send } = useMachine(aiProcessingMachine);

	// --- Svelte Reactive Statements to sync state ---
	$: isLoading = state.matches('processing');
	$: summary = state.context.summary;
	$: error = state.context.error;

	// Track completion for feedback
	$: if (summary && currentInteractionId && feedbackIntegration) {
		feedbackIntegration.markCompleted({
			summary: summary.substring(0, 200) + '...',
			confidence: sources.length > 0 ? 0.9 : 0.7,
			processingTime: Date.now() - (state.context._startTime || Date.now())
		});
	}

	// Track errors for feedback
	$: if (error && currentInteractionId && feedbackIntegration) {
		feedbackIntegration.markFailed({
			errorMessage: error,
			retryCount,
			context: { caseId, evidenceCount: contextItems.length }
		});
	}

	function handleProcessEvidence() {
		if (!user) return;
		
		// Track AI interaction for feedback
		currentInteractionId = feedbackIntegration?.triggerFeedback({
			query: `Analyze ${contextItems.length} evidence items`,
			caseId,
			evidenceCount: contextItems.length,
			userId: user.id
		});

		send({
			type: 'PROCESS',
			caseId,
			evidence: contextItems,
			userId: user.id
		});
	}

	// alias used in template
	const handleSummarize = handleProcessEvidence;

	async function handleSave() {
		if (!canSave) return;
		isSaving = true;
		try {
			await fetch('/api/summary/save', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ caseId, summary, sources })
			});
			// no-op on success; you can emit events or update UI
		} catch (err: any) {
			// show save error
			error = err?.message ?? String(err);
		} finally {
			isSaving = false;
		}
	}

	function handleRetry() {
		if (retryCount >= 3) return;
		// clear error and retry
		error = '';
		send({
			type: 'PROCESS',
			caseId,
			evidence: contextItems,
			userId: user?.id
		});
	}

	function handleReset() {
		summary = '';
		error = '';
		stream = '';
		retryCount = 0;
		sources = [];
		// also clear machine context summary
		// send a PROCESS to re-check cache if needed
	}
</script>

<FeedbackIntegration
	bind:this={feedbackIntegration}
	interactionType="ai_response"
	ratingType="ai_accuracy"
	priority="high"
	context={{ caseId, component: 'AiAssistant' }}
	let:feedback
>
<Card class="ai-assistant-card border-l-4 border-l-blue-500 shadow-sm hover:shadow-md transition-shadow duration-200">
	<CardHeader>
		<CardTitle class="flex items-center gap-3">
			<Bot class="w-6 h-6 text-blue-600 dark:text-blue-400" />
			<span class="text-xl font-bold text-gray-900 dark:text-white">AI Evidence Summary</span>
			{#if contextItems.length > 0}
				<Badge variant="secondary" class="ml-auto">
					{contextItems.length} items
				</Badge>
			{/if}
		</CardTitle>
		<!-- Status Bar -->
		{#if getStatusInfo()}
			{@const status = getStatusInfo()}
			<div class="flex items-center gap-2 text-sm {status.color} bg-gray-50 dark:bg-gray-800 p-3 rounded-md">
				<svelte:component this={status.icon} class="w-4 h-4" />
				<span>{status.text}</span>
				{#if isLoading && enableStreaming}
					<div class="ml-auto animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full"></div>
				{/if}
			</div>
		{/if}
	</CardHeader>

	<CardContent class="space-y-6">
		<!-- Action Buttons -->
		<div class="flex flex-wrap gap-3">
			<AnyButton
				type="button"
				onclick={handleSummarize}
				aria-disabled={!canSummarize}
				disabled={!canSummarize}
				variant="default"
				class="flex-1 min-w-[140px] gap-2 transition-all duration-200 hover:scale-105"
			>
				<Sparkles class="w-4 h-4" />
				{#if !user}
					<User class="w-4 h-4" />
					Sign in to Summarize
				{:else if isLoading}
					Analyzing...
				{:else}
					Summarize Evidence
				{/if}
			</AnyButton>

			{#if allowSave}
				<AnyButton
					type="button"
					onclick={handleSave}
					aria-disabled={!canSave}
					disabled={!canSave}
					variant="outline"
					class="gap-2 transition-all duration-200"
				>
					{#if isSaving}
						<div class="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full"></div>
						Saving...
					{:else}
						<Save class="w-4 h-4" />
						Save to Case
					{/if}
				</AnyButton>
			{/if}

			{#if error && retryCount < 3}
				<AnyButton
					type="button"
					onclick={handleRetry}
					variant="outline"
					class="gap-2 text-orange-600 border-orange-600 hover:bg-orange-50"
				>
					<RefreshCw class="w-4 h-4" />
					Retry ({retryCount}/3)
				</AnyButton>
			{/if}

			{#if summary || error}
				<AnyButton
					type="button"
					onclick={handleReset}
					variant="ghost"
					size="sm"
					class="text-gray-500 hover:text-gray-700"
				>
					Reset
				</AnyButton>
			{/if}
		</div>

		<!-- Content Area -->
		<div class="min-h-[200px]">
			{#if isLoading}
				<div class="space-y-4" role="status" aria-live="polite">
					<div class="flex items-center gap-3 text-blue-600 dark:text-blue-400">
						<div class="animate-spin w-5 h-5 border-2 border-current border-t-transparent rounded-full"></div>
						<span class="font-medium">Analyzing evidence with AI...</span>
					</div>

					<!-- Streaming Output -->
					{#if enableStreaming && stream}
						<div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg border">
							<p class="text-sm text-gray-600 dark:text-gray-400 mb-2">Live output:</p>
							<pre class="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap font-mono">{stream}</pre>
						</div>
					{/if}
				</div>

			{:else if error}
				<div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4" role="alert">
					<div class="flex items-start gap-3">
						<AlertCircle class="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
						<div class="flex-1">
							<h4 class="font-medium text-red-800 dark:text-red-200 mb-1">Analysis Failed</h4>
							<p class="text-sm text-red-700 dark:text-red-300">{error}</p>
							{#if retryCount > 0}
								<p class="text-xs text-red-600 dark:text-red-400 mt-1">Attempt {retryCount} of 3</p>
							{/if}
						</div>
					</div>
				</div>

			{:else if summary}
				<div class="space-y-4">
					<!-- Summary Content -->
					<div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4" role="region" aria-label="Generated summary">
						<h4 class="font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
							<FileText class="w-4 h-4 text-blue-600" />
							Generated Summary
						</h4>
						<div class="prose prose-sm max-w-none dark:prose-invert">
							<pre class="whitespace-pre-wrap text-gray-800 dark:text-gray-200 leading-relaxed">{summary}</pre>
						</div>
					</div>

					<!-- Evidence Sources -->
					{#if showSources && sources.length > 0}
						<div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4" role="region" aria-label="Evidence sources">
							<h4 class="font-semibold text-blue-900 dark:text-blue-100 mb-3 flex items-center gap-2">
								<Database class="w-4 h-4" />
								Source Evidence ({sources.length})
							</h4>
							<div class="space-y-2">
								{#each sources.slice(0, 5) as source, i}
									<div class="flex items-start gap-3 p-2 bg-white dark:bg-blue-950/50 rounded border">
										<Badge variant="outline" class="flex-shrink-0 mt-0.5">
											{i + 1}
										</Badge>
										<div class="flex-1">
											<p class="text-sm font-medium text-blue-900 dark:text-blue-100">
												{source.title || source.name || source.id || `Evidence #${i + 1}`}
											</p>
											{#if source.description}
												<p class="text-xs text-blue-700 dark:text-blue-300 mt-1">
													{source.description}
												</p>
											{/if}
											{#if source.relevance}
												<Badge variant="secondary" class="text-xs mt-1">
													{Math.round(source.relevance * 100)}% relevant
												</Badge>
											{/if}
										</div>
									</div>
								{/each}
								{#if sources.length > 5}
									<p class="text-xs text-blue-600 dark:text-blue-400 text-center py-2">
										+{sources.length - 5} more sources used
									</p>
								{/if}
							</div>
						</div>
					{/if}
				</div>

			{:else}
				<div class="text-center py-12 text-gray-500 dark:text-gray-400">
					<Bot class="w-12 h-12 mx-auto mb-4 opacity-50" />
					{#if !hasContent}
						<h3 class="text-lg font-medium mb-2">No evidence to analyze</h3>
						<p class="text-sm">Select or upload evidence items to generate an AI summary</p>
					{:else if !user}
						<h3 class="text-lg font-medium mb-2">Sign in required</h3>
						<p class="text-sm">Please sign in to use AI analysis features</p>
					{:else}
						<h3 class="text-lg font-medium mb-2">Ready for AI analysis</h3>
						<p class="text-sm">Click "Summarize Evidence" to generate an AI-powered summary</p>
					{/if}
				</div>
			{/if}
		</div>
	</CardContent>
</Card>
</FeedbackIntegration>