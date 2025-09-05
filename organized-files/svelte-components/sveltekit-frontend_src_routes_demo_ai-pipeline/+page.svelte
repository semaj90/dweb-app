<!-- AI Pipeline Demo -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { enhance } from '$app/forms';
	import { writable } from 'svelte/store';
	import { ragQuery, ragResults, ragLoading } from '$lib/services/enhancedRAGPipeline';
	import { goMicroservice } from '$lib/services/goMicroservice';
	
	// Reactive state
	let demoText = `This is a sample legal document about employment contracts. 
It contains provisions for salary, benefits, termination clauses, and confidentiality agreements. 
The contract specifies a base salary of $75,000 per year with annual reviews. 
Termination requires 30 days notice from either party.`;
	
	let autoTagResult = writable(null);
	let isProcessing = false;
	let systemHealth = writable({});
	let benchmarkResults = writable({});
	
	// Demo functions
	async function runAutoTagging() {
		isProcessing = true;
		
		try {
			const response = await fetch('/api/ai/upload-auto-tag', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					documentId: 'demo-doc-' + Date.now(),
					content: demoText,
					documentType: 'contract',
					useGPUAcceleration: true
				})
			});
			
			const result = await response.json();
			autoTagResult.set(result);
			
		} catch (error) {
			console.error('Auto-tagging demo failed:', error);
			autoTagResult.set({ error: error.message });
		} finally {
			isProcessing = false;
		}
	}
	
	async function runRAGQuery() {
		ragLoading.set(true);
		
		const query = 'employment contract termination clauses';
		ragQuery.set(query);
		
		try {
			const response = await fetch('/api/ai/rag-query', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					query,
					options: {
						useSemanticSearch: true,
						useMemoryGraph: true,
						maxSources: 5,
						minConfidence: 0.7
					}
				})
			});
			
			const result = await response.json();
			ragResults.set(result);
			
		} catch (error) {
			console.error('RAG query demo failed:', error);
		} finally {
			ragLoading.set(false);
		}
	}
	
	async function checkSystemHealth() {
		try {
			const goHealth = await goMicroservice.healthCheck();
			const autoTagStatus = await fetch('/api/ai/upload-auto-tag');
			const autoTagData = await autoTagStatus.json();
			
			systemHealth.set({
				goMicroservice: goHealth,
				autoTagging: autoTagData,
				timestamp: new Date().toISOString()
			});
			
		} catch (error) {
			console.error('Health check failed:', error);
		}
	}
	
	async function runBenchmark() {
		try {
			const results = await goMicroservice.benchmark();
			benchmarkResults.set(results);
		} catch (error) {
			console.error('Benchmark failed:', error);
		}
	}
	
	onMount(() => {
		checkSystemHealth();
	});
	
	// Reactive derived values
	let tags = $derived($autoTagResult?.autoTags || [])
	let entities = $derived($autoTagResult?.entities || [])
	let confidence = $derived($autoTagResult?.confidence || 0)
	let similarDocs = $derived($autoTagResult?.similarDocuments || [])
</script>

<svelte:head>
	<title>AI Pipeline Demo - Legal AI System</title>
	<meta name="description" content="GPU-accelerated AI pipeline demonstration" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
	<div class="container mx-auto px-4 py-8">
		<!-- Header -->
		<header class="mb-8">
			<h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
				🚀 AI Pipeline Demo
			</h1>
			<p class="text-gray-400 mt-2">
				GPU-Accelerated Legal AI with Ollama, pgvector, Go microservices, and enhanced RAG
			</p>
		</header>

		<!-- System Health Dashboard -->
		<section class="mb-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700">
			<h2 class="text-2xl font-semibold mb-4 flex items-center">
				🔧 System Health
				<button 
					onclick={checkSystemHealth}
					class="ml-4 px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 rounded transition-colors"
				>
					Refresh
				</button>
			</h2>
			
			{#if $systemHealth.goMicroservice}
				<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
					<div class="p-4 bg-gray-700/50 rounded border">
						<h3 class="font-semibold text-green-400">Go Microservice</h3>
						<p class="text-sm">Status: {$systemHealth.goMicroservice.status}</p>
						<p class="text-sm">GPU: {$systemHealth.goMicroservice.gpu_available ? '✅' : '❌'}</p>
						<p class="text-sm">Uptime: {$systemHealth.goMicroservice.uptime || 0}ms</p>
					</div>
					
					<div class="p-4 bg-gray-700/50 rounded border">
						<h3 class="font-semibold text-blue-400">Auto-Tagging</h3>
						<p class="text-sm">Status: {$systemHealth.autoTagging?.services?.aiAutoTagging || 'Unknown'}</p>
						<p class="text-sm">RAG: {$systemHealth.autoTagging?.services?.enhancedRAG || 'Unknown'}</p>
						<p class="text-sm">Docs: {$systemHealth.autoTagging?.statistics?.totalDocuments || 0}</p>
					</div>
					
					<div class="p-4 bg-gray-700/50 rounded border">
						<h3 class="font-semibold text-purple-400">Capabilities</h3>
						{#each $systemHealth.autoTagging?.capabilities || [] as capability}
							<p class="text-xs">• {capability}</p>
						{/each}
					</div>
				</div>
			{:else}
				<p class="text-gray-400">Loading system health...</p>
			{/if}
		</section>

		<!-- Auto-Tagging Demo -->
		<section class="mb-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700">
			<h2 class="text-2xl font-semibold mb-4">🤖 AI Auto-Tagging Demo</h2>
			
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div>
					<label class="block text-sm font-medium mb-2">Sample Document Text:</label>
					<textarea 
						bind:value={demoText}
						class="w-full h-32 p-3 bg-gray-700 border border-gray-600 rounded resize-none focus:ring-2 focus:ring-blue-500 focus:outline-none"
						placeholder="Paste your legal document text here..."
					></textarea>
					
					<button 
						onclick={runAutoTagging}
						disabled={isProcessing}
						class="mt-4 px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 rounded transition-all duration-200 transform hover:scale-105"
					>
						{isProcessing ? '🔄 Processing...' : '🚀 Run AI Analysis'}
					</button>
				</div>
				
				<div>
					{#if $autoTagResult}
						<div class="space-y-4">
							{#if $autoTagResult.success}
								<!-- Auto-generated Tags -->
								<div>
									<h3 class="font-semibold text-green-400 mb-2">📌 Auto-Generated Tags</h3>
									<div class="flex flex-wrap gap-2">
										{#each tags as tag}
											<span class="px-2 py-1 bg-blue-600/30 border border-blue-500/50 rounded text-sm">
												{tag}
											</span>
										{/each}
									</div>
								</div>
								
								<!-- Extracted Entities -->
								{#if entities.length > 0}
									<div>
										<h3 class="font-semibold text-yellow-400 mb-2">🎯 Extracted Entities</h3>
										<div class="space-y-1">
											{#each entities as entity}
												<div class="text-sm">
													<span class="text-gray-300">{entity.type}:</span>
													<span class="text-white font-medium">{entity.text}</span>
													<span class="text-gray-500">({Math.round(entity.confidence * 100)}%)</span>
												</div>
											{/each}
										</div>
									</div>
								{/if}
								
								<!-- AI Summary -->
								{#if $autoTagResult.summary}
									<div>
										<h3 class="font-semibold text-purple-400 mb-2">📝 AI Summary</h3>
										<p class="text-sm text-gray-300 bg-gray-700/30 p-3 rounded border">
											{$autoTagResult.summary}
										</p>
									</div>
								{/if}
								
								<!-- Confidence Score -->
								<div>
									<h3 class="font-semibold text-orange-400 mb-2">🎯 Confidence Score</h3>
									<div class="flex items-center">
										<div class="flex-1 bg-gray-700 rounded-full h-2">
											<div 
												class="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-2 rounded-full transition-all duration-500"
												style="width: {confidence * 100}%"
											></div>
										</div>
										<span class="ml-3 text-sm font-medium">{Math.round(confidence * 100)}%</span>
									</div>
								</div>
								
								<!-- Similar Documents -->
								{#if similarDocs.length > 0}
									<div>
										<h3 class="font-semibold text-cyan-400 mb-2">🔗 Similar Documents</h3>
										<div class="space-y-2">
											{#each similarDocs as doc}
												<div class="text-sm p-2 bg-gray-700/30 rounded border">
													<span class="text-white font-medium">{doc.title}</span>
													<span class="text-gray-400 ml-2">({Math.round(doc.relevance * 100)}% match)</span>
												</div>
											{/each}
										</div>
									</div>
								{/if}
								
								<!-- Processing Info -->
								{#if $autoTagResult.processing}
									<div>
										<h3 class="font-semibold text-gray-400 mb-2">⚡ Processing Info</h3>
										<div class="text-sm space-y-1">
											<p>GPU Accelerated: {$autoTagResult.processing.gpuAccelerated ? '✅' : '❌'}</p>
											<p>Embedding Dimensions: {$autoTagResult.processing.embeddingDimensions}</p>
											<p>Processing Time: {$autoTagResult.processing.processingTime}ms</p>
										</div>
									</div>
								{/if}
							{:else}
								<div class="p-4 bg-red-900/30 border border-red-700 rounded">
									<p class="text-red-400">❌ Error: {$autoTagResult.error}</p>
								</div>
							{/if}
						</div>
					{:else}
						<div class="flex items-center justify-center h-48 text-gray-500">
							<p>👆 Click "Run AI Analysis" to see results here</p>
						</div>
					{/if}
				</div>
			</div>
		</section>

		<!-- RAG Query Demo -->
		<section class="mb-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700">
			<h2 class="text-2xl font-semibold mb-4">🧠 Enhanced RAG Pipeline Demo</h2>
			
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div>
					<label class="block text-sm font-medium mb-2">RAG Query:</label>
					<input 
						bind:value={$ragQuery}
						class="w-full p-3 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500 focus:outline-none"
						placeholder="Ask about legal documents..."
					/>
					
					<button 
						onclick={runRAGQuery}
						disabled={$ragLoading}
						class="mt-4 px-6 py-2 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 rounded transition-all duration-200 transform hover:scale-105"
					>
						{$ragLoading ? '🔄 Searching...' : '🔍 Query RAG System'}
					</button>
				</div>
				
				<div>
					{#if $ragResults}
						<div class="space-y-4">
							<div>
								<h3 class="font-semibold text-green-400 mb-2">💡 AI Answer</h3>
								<p class="text-sm text-gray-300 bg-gray-700/30 p-3 rounded border">
									{$ragResults.answer}
								</p>
							</div>
							
							<div>
								<h3 class="font-semibold text-blue-400 mb-2">📚 Sources</h3>
								<div class="space-y-2">
									{#each $ragResults.sources || [] as source}
										<div class="text-sm p-2 bg-gray-700/30 rounded border">
											<span class="text-white font-medium">{source.title}</span>
											<span class="text-gray-400 ml-2">({Math.round(source.relevance * 100)}%)</span>
										</div>
									{/each}
								</div>
							</div>
							
							{#if $ragResults.suggestedActions?.length > 0}
								<div>
									<h3 class="font-semibold text-purple-400 mb-2">🎯 Suggested Actions</h3>
									<ul class="text-sm space-y-1">
										{#each $ragResults.suggestedActions as action}
											<li class="text-gray-300">• {action}</li>
										{/each}
									</ul>
								</div>
							{/if}
						</div>
					{:else if $ragLoading}
						<div class="flex items-center justify-center h-32">
							<div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
						</div>
					{:else}
						<div class="flex items-center justify-center h-32 text-gray-500">
							<p>👆 Enter a query and click "Query RAG System"</p>
						</div>
					{/if}
				</div>
			</div>
		</section>

		<!-- Performance Benchmark -->
		<section class="mb-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700">
			<h2 class="text-2xl font-semibold mb-4 flex items-center">
				⚡ Performance Benchmark
				<button 
					onclick={runBenchmark}
					class="ml-4 px-3 py-1 text-sm bg-purple-600 hover:bg-purple-700 rounded transition-colors"
				>
					Run Benchmark
				</button>
			</h2>
			
			{#if $benchmarkResults && Object.keys($benchmarkResults).length > 0}
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
					<div class="p-4 bg-gray-700/50 rounded border text-center">
						<h3 class="font-semibold text-yellow-400">Embeddings/sec</h3>
						<p class="text-2xl font-bold">{Math.round($benchmarkResults.embeddings_per_second || 0)}</p>
					</div>
					
					<div class="p-4 bg-gray-700/50 rounded border text-center">
						<h3 class="font-semibold text-green-400">JSON Parse/sec</h3>
						<p class="text-2xl font-bold">{Math.round($benchmarkResults.json_parse_per_second || 0)}</p>
					</div>
					
					<div class="p-4 bg-gray-700/50 rounded border text-center">
						<h3 class="font-semibold text-blue-400">Matrix Ops/sec</h3>
						<p class="text-2xl font-bold">{Math.round($benchmarkResults.matrix_ops_per_second || 0)}</p>
					</div>
					
					<div class="p-4 bg-gray-700/50 rounded border text-center">
						<h3 class="font-semibold text-purple-400">GPU Speedup</h3>
						<p class="text-2xl font-bold">{$benchmarkResults.gpu_speedup || 0}x</p>
					</div>
				</div>
			{:else}
				<p class="text-gray-400">Click "Run Benchmark" to see performance metrics</p>
			{/if}
		</section>

		<!-- Footer -->
		<footer class="text-center text-gray-500 text-sm">
			<p>🚀 Powered by SvelteKit 2, Ollama GPU, PostgreSQL pgvector, Go microservices, and enhanced RAG</p>
		</footer>
	</div>
</div>

<style>
	:global(body) {
		font-family: 'Inter', system-ui, -apple-system, sans-serif;
	}
</style>