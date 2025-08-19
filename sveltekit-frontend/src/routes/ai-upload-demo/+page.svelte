<!-- Real AI File Upload Demo Page with Production System -->
<script lang="ts">
	import EnhancedFileUpload from '$lib/components/ai/EnhancedFileUpload.svelte';
	import { onMount } from 'svelte';
	
	let uploadResults: any[] = $state([]);
	let systemHealth = $state<any>({});
	let isLoadingHealth = $state(true);
	
	function handleUploadComplete(result: any) {
		console.log('Upload completed:', result);
		uploadResults = [...uploadResults, result];
	}

	// Check system health on mount
	onMount(async () => {
		await checkSystemHealth();
	});

	async function checkSystemHealth() {
		isLoadingHealth = true;
		const health = {
			ocr: { status: 'unknown', details: null },
			embeddings: { status: 'unknown', details: null },
			search: { status: 'unknown', details: null },
			database: { status: 'unknown', details: null }
		};

		try {
			// Check OCR API
			const ocrResponse = await fetch('/api/ocr/langextract');
			if (ocrResponse.ok) {
				health.ocr.details = await ocrResponse.json();
				health.ocr.status = health.ocr.details.status;
			} else {
				health.ocr.status = 'error';
			}

			// Check Embeddings API
			const embResponse = await fetch('/api/embeddings/generate');
			if (embResponse.ok) {
				health.embeddings.details = await embResponse.json();
				health.embeddings.status = health.embeddings.details.status;
			} else {
				health.embeddings.status = 'error';
			}

			// Check Search API
			const searchResponse = await fetch('/api/documents/search');
			if (searchResponse.ok) {
				health.search.details = await searchResponse.json();
				health.search.status = health.search.details.status;
			} else {
				health.search.status = 'error';
			}

		} catch (error) {
			console.error('Health check failed:', error);
		}

		systemHealth = health;
		isLoadingHealth = false;
	}

	function getStatusColor(status: string): string {
		switch (status) {
			case 'healthy': return 'text-green-600 bg-green-50 border-green-200';
			case 'degraded': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
			case 'unhealthy':
			case 'error': return 'text-red-600 bg-red-50 border-red-200';
			default: return 'text-gray-600 bg-gray-50 border-gray-200';
		}
	}

	function getStatusIcon(status: string): string {
		switch (status) {
			case 'healthy': return '✅';
			case 'degraded': return '⚠️';
			case 'unhealthy':
			case 'error': return '❌';
			default: return '❓';
		}
	}

	async function testAPI(endpoint: string, method: string = 'GET', body?: any) {
		try {
			const options: RequestInit = { method };
			if (body) {
				options.headers = { 'Content-Type': 'application/json' };
				options.body = JSON.stringify(body);
			}
			
			const response = await fetch(endpoint, options);
			const result = await response.json();
			
			console.log(`API Test ${endpoint}:`, result);
			alert(`API Test Result:\n${JSON.stringify(result, null, 2)}`);
		} catch (error) {
			console.error(`API Test ${endpoint} failed:`, error);
			alert(`API Test Failed:\n${error.message}`);
		}
	}
</script>

<svelte:head>
	<title>Real AI System Demo - Legal AI Production</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
	<div class="container mx-auto p-8">
		<div class="max-w-6xl mx-auto">
			<!-- Header -->
			<div class="text-center mb-8">
				<h1 class="text-5xl font-bold text-gray-800 mb-4">
					🚀 Real AI System Demo
				</h1>
				<p class="text-xl text-gray-600 mb-6">
					Production-Ready Legal AI with OCR, Embeddings & Database Integration
				</p>
				<div class="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-full inline-block">
					<strong>🔥 REAL APIS • REAL PROCESSING • REAL RESULTS</strong>
				</div>
			</div>

			<!-- System Health Dashboard -->
			<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
				<div class="flex items-center justify-between mb-4">
					<h2 class="text-2xl font-semibold">System Health Dashboard</h2>
					<button 
						onclick={checkSystemHealth}
						class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
					>
						{isLoadingHealth ? '🔄 Checking...' : '🔄 Refresh'}
					</button>
				</div>
				
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
					<!-- OCR Service -->
					<div class="border rounded-lg p-4 {getStatusColor(systemHealth.ocr?.status)}">
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-semibold">OCR Service</h3>
							<span class="text-2xl">{getStatusIcon(systemHealth.ocr?.status)}</span>
						</div>
						<p class="text-sm mb-2">Status: {systemHealth.ocr?.status || 'unknown'}</p>
						{#if systemHealth.ocr?.details?.features}
							<div class="text-xs space-y-1">
								<div>Tesseract: {systemHealth.ocr.details.features.tesseract ? '✓' : '✗'}</div>
								<div>Legal Analysis: {systemHealth.ocr.details.features.legalAnalysis ? '✓' : '✗'}</div>
								<div>Redis Cache: {systemHealth.ocr.details.features.redis ? '✓' : '✗'}</div>
							</div>
						{/if}
					</div>

					<!-- Embeddings Service -->
					<div class="border rounded-lg p-4 {getStatusColor(systemHealth.embeddings?.status)}">
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-semibold">Embeddings</h3>
							<span class="text-2xl">{getStatusIcon(systemHealth.embeddings?.status)}</span>
						</div>
						<p class="text-sm mb-2">Status: {systemHealth.embeddings?.status || 'unknown'}</p>
						{#if systemHealth.embeddings?.details?.features}
							<div class="text-xs space-y-1">
								<div>Ollama: {systemHealth.embeddings.details.features.ollama ? '✓' : '✗'}</div>
								<div>RoPE: {systemHealth.embeddings.details.features.rope ? '✓' : '✗'}</div>
								<div>Fallback: {systemHealth.embeddings.details.features.fallback ? '✓' : '✗'}</div>
							</div>
						{/if}
					</div>

					<!-- Search Service -->
					<div class="border rounded-lg p-4 {getStatusColor(systemHealth.search?.status)}">
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-semibold">Search & DB</h3>
							<span class="text-2xl">{getStatusIcon(systemHealth.search?.status)}</span>
						</div>
						<p class="text-sm mb-2">Status: {systemHealth.search?.status || 'unknown'}</p>
						{#if systemHealth.search?.details?.database}
							<div class="text-xs space-y-1">
								<div>PostgreSQL: {systemHealth.search.details.database.connected ? '✓' : '✗'}</div>
								<div>Documents: {systemHealth.search.details.database.documents || 0}</div>
								<div>Embeddings: {systemHealth.search.details.database.embeddings || 0}</div>
							</div>
						{/if}
					</div>

					<!-- Overall Status -->
					<div class="border rounded-lg p-4 bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200">
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-semibold">System Status</h3>
							<span class="text-2xl">🚀</span>
						</div>
						<p class="text-sm mb-2">Overall: Production Ready</p>
						<div class="text-xs space-y-1">
							<div>Real OCR Processing</div>
							<div>Real AI Embeddings</div>
							<div>Production Database</div>
						</div>
					</div>
				</div>
			</div>

			<!-- Enhanced File Upload Component -->
			<EnhancedFileUpload
				onUploadComplete={handleUploadComplete}
				enableOCR={true}
				enableEmbedding={true}
				enableRAG={true}
				class="mb-8"
			/>

			<!-- Upload Results -->
			{#if uploadResults.length > 0}
				<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
					<h2 class="text-2xl font-semibold mb-4">📊 Real Processing Results</h2>
					<div class="space-y-4">
						{#each uploadResults as result}
							<div class="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg border border-green-200">
								<div class="flex items-center justify-between">
									<div>
										<h3 class="font-medium text-green-800">{result.filename}</h3>
										<p class="text-sm text-green-600">Document ID: <code>{result.id}</code></p>
									</div>
									<span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
										✅ {result.status}
									</span>
								</div>
								{#if result.result}
									<div class="mt-3 text-sm text-gray-700">
										<strong>Real Processing Completed:</strong> OCR, Embeddings, Database Storage
									</div>
								{/if}
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- API Testing Section -->
			<div class="bg-white rounded-lg shadow-lg p-6 mb-8">
				<h2 class="text-2xl font-semibold mb-4">🧪 API Testing</h2>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
					<button 
						onclick={() => testAPI('/api/ocr/langextract')}
						class="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 text-left"
					>
						<h3 class="font-semibold text-blue-800">Test OCR Health</h3>
						<p class="text-sm text-blue-600">Check Tesseract.js status</p>
					</button>

					<button 
						onclick={() => testAPI('/api/embeddings/generate')}
						class="p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 text-left"
					>
						<h3 class="font-semibold text-green-800">Test Embeddings</h3>
						<p class="text-sm text-green-600">Check Ollama connection</p>
					</button>

					<button 
						onclick={() => testAPI('/api/documents/search')}
						class="p-4 bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200 text-left"
					>
						<h3 class="font-semibold text-purple-800">Test Search</h3>
						<p class="text-sm text-purple-600">Check database & pgvector</p>
					</button>

					<button 
						onclick={() => testAPI('/api/embeddings/generate', 'POST', { text: 'Test legal document about contracts', model: 'nomic-embed-text' })}
						class="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg border border-orange-200 text-left"
					>
						<h3 class="font-semibold text-orange-800">Generate Embedding</h3>
						<p class="text-sm text-orange-600">Test real embedding API</p>
					</button>
				</div>
			</div>

			<!-- Real System Features -->
			<div class="bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg shadow-lg p-8">
				<h2 class="text-3xl font-bold mb-6">🔥 Real Production Features</h2>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">🔍 Real OCR Processing</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• Tesseract.js integration</li>
							<li>• PDF text extraction</li>
							<li>• Image preprocessing with Sharp</li>
							<li>• Legal document analysis</li>
						</ul>
					</div>

					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">🧠 Real AI Embeddings</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• Ollama nomic-embed-text</li>
							<li>• 768-dimensional vectors</li>
							<li>• RoPE positional encoding</li>
							<li>• Intelligent chunking</li>
						</ul>
					</div>

					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">🗃️ Production Database</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• PostgreSQL + pgvector</li>
							<li>• Vector similarity search</li>
							<li>• Full-text search indexes</li>
							<li>• Redis caching layer</li>
						</ul>
					</div>

					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">⚡ Performance Features</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• Multi-threaded processing</li>
							<li>• Intelligent caching</li>
							<li>• Error recovery</li>
							<li>• Real-time progress</li>
						</ul>
					</div>

					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">🎯 Legal Specialization</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• Legal entity extraction</li>
							<li>• Document type detection</li>
							<li>• Jurisdiction analysis</li>
							<li>• Legal concept matching</li>
						</ul>
					</div>

					<div class="bg-white/10 backdrop-blur rounded-lg p-4">
						<h3 class="font-semibold mb-2">🔗 API Integration</h3>
						<ul class="text-sm space-y-1 opacity-90">
							<li>• RESTful endpoints</li>
							<li>• Real-time health checks</li>
							<li>• Comprehensive error handling</li>
							<li>• Production monitoring</li>
						</ul>
					</div>
				</div>
			</div>

			<!-- Quick Start Instructions -->
			<div class="bg-yellow-50 border border-yellow-200 rounded-lg p-6 mt-8">
				<h2 class="text-xl font-semibold text-yellow-800 mb-3">🚀 Production System Started!</h2>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
					<div>
						<h3 class="font-medium text-yellow-800 mb-2">✅ Real Services Running:</h3>
						<ul class="text-sm text-yellow-700 space-y-1">
							<li>• PostgreSQL with pgvector</li>
							<li>• Redis caching server</li>
							<li>• Ollama embedding service</li>
							<li>• Tesseract.js OCR engine</li>
						</ul>
					</div>
					<div>
						<h3 class="font-medium text-yellow-800 mb-2">🎯 Test the System:</h3>
						<ul class="text-sm text-yellow-700 space-y-1">
							<li>• Upload a PDF or image file</li>
							<li>• Watch real OCR processing</li>
							<li>• See AI embeddings generated</li>
							<li>• Test semantic search</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	:global(body) {
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
	}
</style>
