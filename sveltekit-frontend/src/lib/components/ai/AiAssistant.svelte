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

	// TensorFlow.js AI Services Integration
	import { TensorFlowSynthesizer } from '$lib/middleware/tfjs-synthesizer';
	import { MultiLayerCache } from '$lib/services/multi-layer-cache';
	import { rabbitMQQueue, type DocumentMessage, type ChunkMessage } from '$lib/services/rabbitmq-queue-service';
	import type { SynthesizedAnalysis } from '$lib/middleware/tfjs-synthesizer';
	import type { NLPCacheOperations } from '$lib/services/multi-layer-cache';

	// Feedback Integration
	import FeedbackIntegration from '$lib/components/feedback/FeedbackIntegration.svelte';

	// Get user from context (set in +layout.svelte)
	const getUser = getContext('user');
	const user = typeof getUser === 'function' ? getUser() : undefined;

	interface Props {
		contextItems?: any[];
		caseId?: string;
	}

	let { contextItems = [], caseId = '' }: Props = $props();

// --- TensorFlow.js AI Services Initialization ---
let tensorFlowSynthesizer: TensorFlowSynthesizer | null = null;
let multiLayerCache: MultiLayerCache | null = null;
let synthesizedResults: SynthesizedAnalysis | null = null;

// Initialize AI services
async function initializeAIServices() {
	if (!multiLayerCache) {
		multiLayerCache = new MultiLayerCache({
			enableRedisCache: true,
			enableLokiCache: true,
			enableMemoryCache: true,
			redisTTL: 3600, // 1 hour
			lokiTTL: 1800,  // 30 minutes
			memoryTTL: 300, // 5 minutes
		});
		await multiLayerCache.initialize();
	}
	
	if (!tensorFlowSynthesizer) {
		tensorFlowSynthesizer = new TensorFlowSynthesizer({
			parallelProcessing: true,
			useGPUAcceleration: false, // Browser environment
			debugMode: false,
			legalBERTConfig: {
				modelPath: '/models/legal-bert',
				vocabularyPath: '/models/legal-bert-vocab.json'
			},
			cacheService: multiLayerCache
		});
		await tensorFlowSynthesizer.initialize();
	}

	// Initialize RabbitMQ queue service for NLP task queuing
	if (!rabbitMQQueue.connected) {
		try {
			await rabbitMQQueue.initialize();
			console.log('✅ RabbitMQ queue service initialized');
			
			// Set up event listeners for queue events
			rabbitMQQueue.on('documentQueued', (event) => {
				console.log('📄 Document queued for processing:', event);
			});
			
			rabbitMQQueue.on('chunkQueued', (event) => {
				console.log('🧩 Chunk queued for embedding:', event);
			});
			
			rabbitMQQueue.on('embeddingQueued', (event) => {
				console.log('🧠 Embedding queued for storage:', event);
			});
			
		} catch (error) {
			console.warn('⚠️ RabbitMQ initialization failed, continuing without queuing:', error);
		}
	}
}

// Initialize services when component mounts
let servicesInitialized = $state(false);
$effect(() => {
	if (!servicesInitialized) {
		initializeAIServices().then(() => {
			servicesInitialized = true;
		}).catch(err => {
			console.warn('AI services initialization failed:', err);
			servicesInitialized = true; // Allow fallback
		});
	}
});

// --- Client-Side Caching with Loki.js ---
// Initializes a simple in-memory DB to cache summaries on the client.
// Ensure Loki.js DB and collection are initialized only once (singleton pattern).
let db = $state<Loki | null>(null);
	let lokiSummaryCache: Collection<any> | null;

	function getSummaryCache() {
		if (!db) {
			db = new Loki('ai-cache.db');
			lokiSummaryCache = db.addCollection('summaries', { indices: ['caseId'] });
		} else if (!lokiSummaryCache) {
			lokiSummaryCache =
				db.getCollection('summaries') ||
				db.addCollection('summaries', { indices: ['caseId'] });
		}
		return lokiSummaryCache!;
	}
	// initialize
	const summaryCacheCollection = getSummaryCache();

	// Component state (some are synced to XState below)
	let summary = $state('');
	let error = $state('');
	let isSaving = $state(false);
	let retryCount = $state(0);
	let stream = $state('');
let enableStreaming = $state(false); // set true if you wire streaming
let showSources = $state(true);
	let sources = $state<any[]>([]);

	// Derived booleans used in template
	const hasContent = $derived(() => contextItems.length > 0);
	const isLoading = $derived(() => state.matches('processing'));
	const canSummarize = $derived(() => hasContent && !!user && !isLoading);
	const allowSave = true;
// Feedback integration variables
let feedbackIntegration = $state<any>();
	let currentInteractionId: string | null = null;
let feedbackIntegration = $state<any;
	let currentInteractionId: string | null >(null);

	const getStatusInfo = () => {
		if (isLoading) {
			const processingText = servicesInitialized && tensorFlowSynthesizer 
				? 'Analyzing with TensorFlow.js...' 
				: 'Analyzing...';
			return { icon: Bot, text: processingText, color: 'text-blue-600' };
		}
		if (error) {
			return { icon: AlertCircle, text: 'Error', color: 'text-red-600' };
		}
		if (summary) {
			const readyText = synthesizedResults 
				? 'TensorFlow.js Analysis Complete' 
				: 'Summary ready';
			return { icon: FileText, text: readyText, color: 'text-green-600' };
		}
		if (!servicesInitialized) {
			return { icon: RefreshCw, text: 'Initializing AI services...', color: 'text-orange-600' };
		}
		return null;
	};

	async function fetchSummaryFromServer(payload: { caseId: string; evidence: any[]; userId?: string }) {
		try {
			const evidenceText = payload.evidence
				.map(item => `${item.title || 'Evidence'}: ${item.content || item.description || ''}`)
				.join('\n\n');

			// Try TensorFlow.js synthesizer first with caching
			if (tensorFlowSynthesizer && multiLayerCache && servicesInitialized) {
				try {
					console.log('Using TensorFlow.js synthesizer with caching...');
					
					// Check cache first using multi-layer cache
					const cacheKey = `evidence_analysis_${payload.caseId}`;
					const cachedResult = await multiLayerCache.get('summary', cacheKey);
					
					if (cachedResult) {
						console.log('Cache hit for evidence analysis');
						synthesizedResults = cachedResult;
						return {
							summary: cachedResult.enhancedResponse.summary,
							sources: cachedResult.enhancedResponse.sources || [],
							confidence: cachedResult.qualityMetrics?.confidence || 0.85
						};
					}

					// Perform TensorFlow.js analysis with synthesizer
					const analysisResult = await tensorFlowSynthesizer.synthesizeAnalysis(
						evidenceText,
						`Analyze and summarize the following legal evidence for case ${payload.caseId}`,
						{
							caseId: payload.caseId,
							userId: payload.userId,
							evidenceCount: payload.evidence.length,
							requestType: 'evidence_summary'
						}
					);

					// Cache the results using multi-layer cache
					await multiLayerCache.set('summary', cacheKey, analysisResult, 3600); // 1 hour TTL
					
					synthesizedResults = analysisResult;
					
					return {
						summary: analysisResult.enhancedResponse.summary,
						sources: analysisResult.enhancedResponse.sources || [],
						confidence: analysisResult.qualityMetrics?.confidence || 0.9,
						processingTime: analysisResult.processingPipeline.totalProcessingTime,
						aiInsights: analysisResult.synthesizedInsights
					};
				} catch (tfError) {
					console.warn('TensorFlow.js synthesizer error:', tfError);
					// Fall through to API fallback
				}
			}

			// Fallback to existing Enhanced RAG service
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
					// Use the enhanced processing with queuing capability
					return await processWithQueuing(context.evidence);
				}
			}
		}
	);

	const { state, send } = useMachine(aiProcessingMachine);

	// --- Svelte Reactive Statements to sync state ---
	$effect(() => {
		summary = state.context.summary;
		error = state.context.error;
	});

	// Track completion for feedback
	$effect(() => {
		if (summary && currentInteractionId && feedbackIntegration) {
			feedbackIntegration.markCompleted({
				summary: summary.substring(0, 200) + '...',
				confidence: sources.length > 0 ? 0.9 : 0.7,
				processingTime: Date.now() - (state.context._startTime || Date.now())
			});
		}
	});

	// Track errors for feedback
	$effect(() => {
		if (error && currentInteractionId && feedbackIntegration) {
			feedbackIntegration.markFailed({
				errorMessage: error,
				retryCount,
				context: { caseId, evidenceCount: contextItems.length }
			});
		}
	});

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
		synthesizedResults = null;
		// also clear machine context summary
		// send a PROCESS to re-check cache if needed
	}

	/**
	 * Queue documents for background processing using RabbitMQ
	 * This implements the RAG ingestion pipeline described by the user
	 */
	async function queueDocumentProcessing(evidence: any[]): Promise<string[]> {
		if (!rabbitMQQueue.connected) {
			throw new Error('RabbitMQ queue service not available');
		}

		const jobIds: string[] = [];

		for (const item of evidence) {
			const documentMessage: DocumentMessage = {
				document_id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
				case_id: caseId,
				source_location: item.url || item.path || 'memory://evidence',
				metadata: {
					title: item.title || item.name || 'Evidence Item',
					file_type: item.type || 'text',
					upload_date: new Date().toISOString(),
					user_id: user?.id
				}
			};

			try {
				const jobId = await rabbitMQQueue.publishDocument(documentMessage);
				jobIds.push(jobId);
				
				console.log(`✅ Queued document ${documentMessage.document_id} for processing`);
			} catch (error) {
				console.error('❌ Failed to queue document:', error);
				throw error;
			}
		}

		return jobIds;
	}

	/**
	 * Enhanced processing function that uses queuing for expensive operations
	 */
	async function processWithQueuing(evidence: any[]): Promise<any> {
		try {
			// For small batches, process directly with TensorFlow.js
			if (evidence.length <= 3) {
				console.log('📊 Processing small batch directly with TensorFlow.js');
				return await fetchSummaryFromServer({
					caseId,
					evidence,
					userId: user?.id
				});
			}

			// For larger batches, use RabbitMQ queuing
			console.log('🚀 Using RabbitMQ queuing for large batch processing');
			
			const queuedJobIds = await queueDocumentProcessing(evidence);
			
			// Return immediate response indicating queued processing
			return {
				summary: `Processing ${evidence.length} evidence items in background. Job IDs: ${queuedJobIds.slice(0, 3).join(', ')}${queuedJobIds.length > 3 ? '...' : ''}`,
				sources: [],
				confidence: 0.0,
				processingMode: 'queued',
				jobIds: queuedJobIds,
				estimatedCompletionTime: evidence.length * 30 // 30 seconds per item estimate
			};

		} catch (error) {
			console.warn('⚠️ Queuing failed, falling back to direct processing:', error);
			
			// Fallback to direct processing
			return await fetchSummaryFromServer({
				caseId,
				evidence,
				userId: user?.id
			});
		}
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
				on:on:click={handleSummarize}
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
				<AnyButton
					type="button"
					on:click={handleSave}
					aria-disabled={!canSave}
					disabled={!canSave}
					variant="outline"
					class="gap-2 transition-all duration-200"
				>
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
				<AnyButton
					type="button"
					on:click={handleRetry}
					variant="outline"
					class="gap-2 text-orange-600 border-orange-600 hover:bg-orange-50"
				>
					class="gap-2 text-orange-600 border-orange-600 hover:bg-orange-50"
				>
					<RefreshCw class="w-4 h-4" />
					Retry ({retryCount}/3)
				</AnyButton>
			{/if}
				<AnyButton
					type="button"
					on:click={handleReset}
					variant="ghost"
					size="sm"
					class="text-gray-500 hover:text-gray-700"
				>
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

					<!-- AI Insights from TensorFlow.js Synthesizer -->
					{#if synthesizedResults?.synthesizedInsights}
						<div class="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg p-4" role="region" aria-label="AI insights">
							<h4 class="font-semibold text-emerald-900 dark:text-emerald-100 mb-3 flex items-center gap-2">
								<Sparkles class="w-4 h-4" />
								TensorFlow.js AI Insights
							</h4>
							<div class="space-y-3">
								{#if synthesizedResults.synthesizedInsights.riskAssessment}
									<div class="p-3 bg-white dark:bg-emerald-950/50 rounded border">
										<p class="text-sm font-medium text-emerald-900 dark:text-emerald-100 mb-1">Risk Assessment</p>
										<p class="text-xs text-emerald-700 dark:text-emerald-300">
											Level: {synthesizedResults.synthesizedInsights.riskAssessment.level} 
											(Score: {synthesizedResults.synthesizedInsights.riskAssessment.score})
										</p>
									</div>
								{/if}
								
								{#if synthesizedResults.synthesizedInsights.keyEntities?.length > 0}
									<div class="p-3 bg-white dark:bg-emerald-950/50 rounded border">
										<p class="text-sm font-medium text-emerald-900 dark:text-emerald-100 mb-2">Key Legal Entities</p>
										<div class="flex flex-wrap gap-1">
											{#each synthesizedResults.synthesizedInsights.keyEntities.slice(0, 8) as entity}
												<Badge variant="outline" class="text-xs">
													{entity.text} ({entity.type})
												</Badge>
											{/each}
										</div>
									</div>
								{/if}

								{#if synthesizedResults.synthesizedInsights.recommendedActions?.length > 0}
									<div class="p-3 bg-white dark:bg-emerald-950/50 rounded border">
										<p class="text-sm font-medium text-emerald-900 dark:text-emerald-100 mb-2">Recommended Actions</p>
										<div class="space-y-1">
											{#each synthesizedResults.synthesizedInsights.recommendedActions.slice(0, 3) as action}
												<p class="text-xs text-emerald-700 dark:text-emerald-300">• {action}</p>
											{/each}
										</div>
									</div>
								{/if}

								{#if synthesizedResults.qualityMetrics}
									<div class="flex gap-2 text-xs text-emerald-600 dark:text-emerald-400">
										<Badge variant="secondary">
											Confidence: {Math.round((synthesizedResults.qualityMetrics.confidence || 0) * 100)}%
										</Badge>
										{#if synthesizedResults.processingPipeline.totalProcessingTime}
											<Badge variant="secondary">
												Processing: {synthesizedResults.processingPipeline.totalProcessingTime}ms
											</Badge>
										{/if}
										<Badge variant="secondary">
											TensorFlow.js Enhanced
										</Badge>
									</div>
								{/if}
							</div>
						</div>
					{/if}

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