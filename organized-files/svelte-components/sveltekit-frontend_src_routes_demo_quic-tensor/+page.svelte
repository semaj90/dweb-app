<script lang="ts">
	// QUIC Tensor Demo - Interactive Testing Interface
	// Tests did-you-mean suggestions and tensor operations
	
	import { onMount, onDestroy } from 'svelte';
	import { QuicTensorClient, TensorUtils, type Tensor4DInfo } from '$lib/services/quic-tensor-client';
	import { DidYouMeanClient, SuggestionUtils, type SuggestionResponse } from '$lib/services/did-you-mean-client';
	import Button from '$lib/components/ui/Button.svelte';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';

	// Client instances
	const tensorClient = new QuicTensorClient();
	const suggestionClient = new DidYouMeanClient();

	// Reactive state
	let searchQuery = $state('');
	let suggestions = $state<SuggestionResponse | null>(null);
	let tensorInfo = $state<Tensor4DInfo | null>(null);
	let isLoading = $state(false);
	let error = $state<string | null>(null);
	let healthStatus = $state<any>(null);

	// Demo data
	let selectedDocument = $state('contract');
	const sampleDocuments = {
		contract: {
			text: 'This Service Agreement is entered into between the Client and the Service Provider. The Provider agrees to deliver legal services including contract review, liability assessment, and compliance verification.',
			type: 'contract',
			practiceArea: 'corporate'
		},
		litigation: {
			text: 'The plaintiff alleges that the defendant breached their fiduciary duty, resulting in damages of $500,000. Evidence includes email correspondence and financial records.',
			type: 'complaint',
			practiceArea: 'civil'
		},
		criminal: {
			text: 'The defendant is charged with fraud in the first degree. The prosecution will present evidence of financial misconduct and witness testimony.',
			type: 'indictment',
			practiceArea: 'criminal'
		}
	};

	// Suggestion demo queries
	const sampleQueries = [
		'contract liability',
		'defendant rights',
		'evidence admisibility',
		'plaintif damages',
		'legal precednt'
	];

	let selectedQuery = $state(0);
	let isStreaming = $state(false);

	// Auto-suggestion debounce timer
	let debounceTimer: NodeJS.Timeout;

	// Handle search query changes
	function handleQueryChange() {
		if (debounceTimer) clearTimeout(debounceTimer);
		
		debounceTimer = setTimeout(async () => {
			if (searchQuery.trim().length > 2) {
				await fetchSuggestions();
			} else {
				suggestions = null;
			}
		}, 150);
	}

	// Fetch suggestions
	async function fetchSuggestions() {
		if (!searchQuery.trim()) return;
		
		isLoading = true;
		error = null;
		
		try {
			const result = await suggestionClient.getSuggestions(searchQuery, 'legal_research');
			suggestions = result;
		} catch (err) {
			error = `Suggestion error: ${err instanceof Error ? err.message : 'Unknown error'}`;
			console.error('Suggestion error:', err);
		} finally {
			isLoading = false;
		}
	}

	// Process document into tensor
	async function processDocument() {
		const doc = sampleDocuments[selectedDocument as keyof typeof sampleDocuments];
		if (!doc) return;

		isLoading = true;
		error = null;

		try {
			const result = await tensorClient.processLegalDocument(
				`doc_${Date.now()}`,
				doc.text,
				doc.type,
				doc.practiceArea
			);
			tensorInfo = result;
		} catch (err) {
			error = `Tensor processing error: ${err instanceof Error ? err.message : 'Unknown error'}`;
			console.error('Tensor error:', err);
		} finally {
			isLoading = false;
		}
	}

	// Test tricubic interpolation
	async function testInterpolation() {
		if (!tensorInfo) {
			error = 'No tensor available for interpolation';
			return;
		}

		isLoading = true;
		error = null;

		try {
			const result = await tensorClient.tricubicInterpolation(
				tensorInfo.tensor_id,
				[0.5, 0.5, 0.5],
				{
					points: [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
					coordinates: [0.5, 0.5, 0.5],
					smoothness: 0.8
				}
			);
			console.log('Interpolation result:', result);
			alert(`Interpolation completed! Result dimension: ${result.dimension}, Processing time: ${result.quic_processing_time_ms}ms`);
		} catch (err) {
			error = `Interpolation error: ${err instanceof Error ? err.message : 'Unknown error'}`;
			console.error('Interpolation error:', err);
		} finally {
			isLoading = false;
		}
	}

	// Check system health
	async function checkHealth() {
		isLoading = true;
		error = null;

		try {
			const [tensorHealth, streamStatus] = await Promise.all([
				tensorClient.checkHealth(),
				tensorClient.getStreamStatus()
			]);
			
			healthStatus = {
				tensor: tensorHealth,
				streams: streamStatus
			};
		} catch (err) {
			error = `Health check error: ${err instanceof Error ? err.message : 'Unknown error'}`;
			console.error('Health check error:', err);
		} finally {
			isLoading = false;
		}
	}

	// Use sample query
	function useSampleQuery(index: number) {
		selectedQuery = index;
		searchQuery = sampleQueries[index];
		handleQueryChange();
	}

	// Clear cache
	async function clearCache() {
		try {
			await suggestionClient.clearCache();
			suggestions = null;
			alert('Cache cleared successfully');
		} catch (err) {
			error = `Cache clear error: ${err instanceof Error ? err.message : 'Unknown error'}`;
		}
	}

	// Cleanup on destroy
	onDestroy(() => {
		if (debounceTimer) clearTimeout(debounceTimer);
		suggestionClient.cleanup();
	});

	// Auto health check on mount
	onMount(() => {
		checkHealth();
	});
</script>

<svelte:head>
	<title>QUIC Tensor System Demo - Legal AI Platform</title>
	<meta name="description" content="Interactive demo of QUIC-based tensor operations and did-you-mean suggestions for legal AI" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
	<div class="max-w-7xl mx-auto space-y-8">
		<!-- Header -->
		<div class="text-center space-y-4">
			<h1 class="text-4xl font-bold text-white">🚀 QUIC Tensor System Demo</h1>
			<p class="text-xl text-purple-200">
				Ultra-low latency tensor operations and intelligent suggestions for Legal AI
			</p>
		</div>

		<!-- System Status -->
		<Card class="bg-slate-800/50 border-purple-500/20">
			<CardHeader>
				<CardTitle class="text-purple-300 flex items-center gap-2">
					📊 System Status
					<Button 
						on:click={checkHealth} 
						disabled={isLoading}
						variant="secondary" 
						size="sm"
					>
						Refresh
					</Button>
				</CardTitle>
			</CardHeader>
			<CardContent>
				{#if healthStatus}
					<div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
						<!-- Tensor Service Health -->
						<div class="space-y-2">
							<h4 class="font-semibold text-white">Tensor Service</h4>
							<div class="space-y-1 text-slate-300">
								<div>API: <span class="text-green-400">{healthStatus.tensor.quic_tensor_api}</span></div>
								<div>Go Backend: <span class={healthStatus.tensor.go_backend === 'healthy' ? 'text-green-400' : 'text-red-400'}>{healthStatus.tensor.go_backend}</span></div>
								<div>URL: <code class="text-purple-300">{healthStatus.tensor.go_backend_url}</code></div>
							</div>
						</div>

						<!-- Stream Status -->
						<div class="space-y-2">
							<h4 class="font-semibold text-white">QUIC Streams</h4>
							<div class="space-y-1 text-slate-300">
								<div>Active: <span class="text-blue-400">{healthStatus.streams.active_streams}/{healthStatus.streams.max_concurrent}</span></div>
								<div>Utilization: <span class="text-purple-400">{healthStatus.streams.utilization_percent.toFixed(1)}%</span></div>
								<div>Total: <span class="text-slate-400">{healthStatus.streams.total_streams}</span></div>
							</div>
						</div>
					</div>
				{:else}
					<div class="text-slate-400">Click Refresh to check system health</div>
				{/if}
			</CardContent>
		</Card>

		<div class="grid grid-cols-1 xl:grid-cols-2 gap-8">
			<!-- Did You Mean Suggestions -->
			<Card class="bg-slate-800/50 border-blue-500/20">
				<CardHeader>
					<CardTitle class="text-blue-300">🎯 Did You Mean - Legal Suggestions</CardTitle>
				</CardHeader>
				<CardContent class="space-y-4">
					<!-- Search Input -->
					<div class="space-y-2">
						<label class="text-white text-sm font-medium">Search Query</label>
						<input 
							type="text"
							bind:value={searchQuery}
							on:input={handleQueryChange}
							placeholder="Type a legal search query..."
							class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
						/>
					</div>

					<!-- Sample Queries -->
					<div class="space-y-2">
						<label class="text-white text-sm font-medium">Sample Queries (with typos)</label>
						<div class="flex flex-wrap gap-2">
							{#each sampleQueries as query, index}
								<Button
									on:click={() => useSampleQuery(index)}
									variant={selectedQuery === index ? "default" : "secondary"}
									size="sm"
								>
									{query}
								</Button>
							{/each}
						</div>
					</div>

					<!-- Suggestions Display -->
					{#if suggestions}
						<div class="space-y-3">
							<div class="flex justify-between items-center">
								<h4 class="text-white font-medium">
									Suggestions ({suggestions.suggestions.length})
								</h4>
								<div class="text-xs text-slate-400 space-x-4">
									<span>⚡ {suggestions.processing_time_ms}ms</span>
									<span>{suggestions.cache_hit ? '💾 Cached' : '🔄 Fresh'}</span>
									{#if suggestions.graph_traversal_ms}
										<span>🔗 Graph: {suggestions.graph_traversal_ms.toFixed(1)}ms</span>
									{/if}
								</div>
							</div>

							<div class="space-y-2 max-h-96 overflow-y-auto">
								{#each suggestions.suggestions as suggestion, index}
									<div class="p-3 bg-slate-700/50 rounded-lg border border-slate-600/50 hover:border-blue-500/30 transition-colors">
										<div class="flex items-center justify-between">
											<div class="flex items-center gap-2">
												<span class="text-lg">{suggestion.icon}</span>
												<span class="text-white">{suggestion.text}</span>
											</div>
											<div class="text-right space-y-1">
												<div class="text-xs text-slate-400">{suggestion.type}</div>
												<div class="text-xs font-mono" style="color: {SuggestionUtils.getSuggestionColor(suggestion.type)}">
													{(suggestion.confidence * 100).toFixed(0)}%
												</div>
											</div>
										</div>
										{#if suggestion.legal_context}
											<div class="text-xs text-slate-400 mt-1">{suggestion.legal_context}</div>
										{/if}
									</div>
								{/each}
							</div>
						</div>
					{:else if isLoading}
						<div class="text-center py-8">
							<div class="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2"></div>
							<div class="text-slate-400">Generating suggestions...</div>
						</div>
					{:else}
						<div class="text-center py-8 text-slate-400">
							Type a query to see intelligent legal suggestions
						</div>
					{/if}

					<!-- Actions -->
					<div class="flex gap-2">
						<Button on:click={fetchSuggestions} disabled={isLoading || !searchQuery.trim()} variant="default" size="sm">
							🔄 Refresh Suggestions
						</Button>
						<Button on:click={clearCache} variant="secondary" size="sm">
							🗑️ Clear Cache
						</Button>
					</div>
				</CardContent>
			</Card>

			<!-- Tensor Operations -->
			<Card class="bg-slate-800/50 border-purple-500/20">
				<CardHeader>
					<CardTitle class="text-purple-300">🧮 4D Tensor Operations</CardTitle>
				</CardHeader>
				<CardContent class="space-y-4">
					<!-- Document Selection -->
					<div class="space-y-2">
						<label class="text-white text-sm font-medium">Sample Legal Document</label>
						<select 
							bind:value={selectedDocument}
							class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20"
						>
							<option value="contract">📄 Service Agreement (Corporate)</option>
							<option value="litigation">⚖️ Civil Complaint (Litigation)</option>
							<option value="criminal">🏛️ Criminal Indictment</option>
						</select>
					</div>

					<!-- Document Preview -->
					<div class="space-y-2">
						<label class="text-white text-sm font-medium">Document Content</label>
						<div class="p-3 bg-slate-700/30 border border-slate-600/50 rounded-lg text-sm text-slate-300">
							{sampleDocuments[selectedDocument as keyof typeof sampleDocuments]?.text}
						</div>
					</div>

					<!-- Process Document -->
					<div class="space-y-3">
						<Button 
							on:click={processDocument} 
							disabled={isLoading}
							class="w-full"
							variant="default"
						>
							{isLoading ? '⏳ Processing...' : '🚀 Create 4D Tensor'}
						</Button>

						{#if tensorInfo}
							<div class="p-4 bg-slate-700/30 border border-purple-500/30 rounded-lg space-y-2">
								<h4 class="text-purple-300 font-medium">✅ Tensor Created</h4>
								<div class="text-sm text-slate-300 space-y-1">
									<div><strong>ID:</strong> <code class="text-purple-400">{tensorInfo.tensor_id}</code></div>
									<div><strong>Shape:</strong> <span class="text-blue-400">{TensorUtils.formatTensorShape(tensorInfo.shape)}</span></div>
									<div><strong>Tiles:</strong> <span class="text-green-400">{tensorInfo.tiles}</span></div>
									<div><strong>Memory:</strong> <span class="text-yellow-400">{(TensorUtils.estimateMemoryUsage(tensorInfo.shape) / 1024 / 1024).toFixed(2)}MB</span></div>
									<div><strong>Practice Area:</strong> <span class="text-cyan-400">{tensorInfo.metadata.practice_area}</span></div>
								</div>

								<div class="flex gap-2 pt-2">
									<Button on:click={testInterpolation} disabled={isLoading} variant="secondary" size="sm">
										🔄 Test Tricubic Interpolation
									</Button>
								</div>
							</div>
						{/if}
					</div>

					<!-- Performance Metrics -->
					{#if tensorInfo || suggestions}
						<div class="p-4 bg-slate-700/20 border border-slate-600/30 rounded-lg">
							<h4 class="text-white font-medium mb-2">📊 Performance Metrics</h4>
							<div class="text-xs text-slate-400 space-y-1">
								{#if suggestions}
									<div>Suggestion Cache: {suggestionClient.getSuggestionMetrics().cache_size} entries</div>
									<div>Avg Processing: {suggestionClient.getSuggestionMetrics().average_processing_time.toFixed(1)}ms</div>
									<div>Cache Hit Rate: {suggestionClient.getSuggestionMetrics().cache_hit_rate.toFixed(1)}%</div>
								{/if}
								{#if tensorInfo}
									<div>Tensor Operations: Active</div>
									<div>Backend: Go tensor-tiling service</div>
									<div>Protocol: QUIC/HTTP3 + REST fallback</div>
								{/if}
							</div>
						</div>
					{/if}
				</CardContent>
			</Card>
		</div>

		<!-- Error Display -->
		{#if error}
			<Card class="bg-red-900/20 border-red-500/30">
				<CardContent class="p-4">
					<div class="flex items-center gap-2 text-red-400">
						<span>❌</span>
						<span class="font-medium">Error:</span>
						<span>{error}</span>
						<Button on:click={() => error = null} variant="ghost" size="sm" class="ml-auto">
							✕
						</Button>
					</div>
				</CardContent>
			</Card>
		{/if}

		<!-- API Documentation -->
		<Card class="bg-slate-800/30 border-slate-700/50">
			<CardHeader>
				<CardTitle class="text-slate-300">📖 API Endpoints</CardTitle>
			</CardHeader>
			<CardContent>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
					<div class="space-y-3">
						<h4 class="text-white font-medium">Did You Mean API</h4>
						<div class="space-y-2 text-slate-400">
							<div><code class="text-green-400">GET</code> <code>/api/suggest/did-you-mean?q=query</code></div>
							<div><code class="text-blue-400">POST</code> <code>/api/suggest/did-you-mean</code></div>
							<div><code class="text-red-400">DELETE</code> <code>/api/suggest/did-you-mean</code></div>
						</div>
					</div>

					<div class="space-y-3">
						<h4 class="text-white font-medium">QUIC Tensor API</h4>
						<div class="space-y-2 text-slate-400">
							<div><code class="text-blue-400">POST</code> <code>/api/v1/quic/tensor?op=create</code></div>
							<div><code class="text-blue-400">POST</code> <code>/api/v1/quic/tensor?op=interpolate</code></div>
							<div><code class="text-green-400">GET</code> <code>/api/v1/quic/tensor?tensor_id=X</code></div>
							<div><code class="text-purple-400">OPTIONS</code> <code>/api/v1/quic/tensor</code></div>
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	</div>
</div>

<style>
	/* Custom scrollbar for suggestion list */
	.space-y-2::-webkit-scrollbar {
		width: 4px;
	}
	.space-y-2::-webkit-scrollbar-track {
		background: rgb(51 65 85 / 0.3);
		border-radius: 2px;
	}
	.space-y-2::-webkit-scrollbar-thumb {
		background: rgb(124 58 237 / 0.5);
		border-radius: 2px;
	}
	.space-y-2::-webkit-scrollbar-thumb:hover {
		background: rgb(124 58 237 / 0.8);
	}
</style>