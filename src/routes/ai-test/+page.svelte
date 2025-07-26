<script>
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import AIChat from '$lib/components/AIChat.svelte';
	import { Card } from '$lib/components/ui';
	import { Button } from '$lib/components/ui';
	import { Badge } from '$lib/components/ui';

	// System status stores
	const systemHealth = writable({ status: 'checking', services: {} });
	const availableModels = writable([]);
	const testResults = writable({});

	// Test configuration
	let selectedCaseId = 'CASE-2024-001';
	let testSessionId = null;

	// Health check configuration
	const services = {
		ollama: { url: 'http://localhost:11434/api/tags', name: 'Ollama' },
		database: { url: '/api/chat?action=health', name: 'Database' },
		embedding: { url: '/api/chat?action=health', name: 'Embeddings' },
		rag: { url: '/api/chat?action=health', name: 'RAG System' }
	};

	onMount(async () => {
		await runHealthChecks();
		await loadAvailableModels();

		// Set up periodic health checks
		const healthInterval = setInterval(runHealthChecks, 30000);

		return () => clearInterval(healthInterval);
	});

	/**
	 * Run comprehensive health checks
	 */
	async function runHealthChecks() {
		const results = {};

		for (const [key, service] of Object.entries(services)) {
			try {
				const response = await fetch(service.url);
				const data = await response.json();

				results[key] = {
					status: response.ok ? 'healthy' : 'error',
					name: service.name,
					details: data,
					timestamp: new Date().toISOString()
				};
			} catch (error) {
				results[key] = {
					status: 'error',
					name: service.name,
					error: error.message,
					timestamp: new Date().toISOString()
				};
			}
		}

		const overallStatus = Object.values(results).every(r => r.status === 'healthy')
			? 'healthy'
			: 'partial';

		systemHealth.set({
			status: overallStatus,
			services: results,
			lastCheck: new Date().toISOString()
		});
	}

	/**
	 * Load available models from Ollama
	 */
	async function loadAvailableModels() {
		try {
			const response = await fetch('/api/chat?action=models');
			const data = await response.json();
			availableModels.set(data.models || []);
		} catch (error) {
			console.error('Error loading models:', error);
			availableModels.set([]);
		}
	}

	/**
	 * Run RAG system tests
	 */
	async function runRAGTests() {
		const tests = [
			{
				name: 'Basic Legal Query',
				query: 'What are the elements of embezzlement?',
				expected: 'legal knowledge'
			},
			{
				name: 'Case-Specific Query',
				query: 'What evidence do we have in the Anderson case?',
				expected: 'case documents'
			},
			{
				name: 'Procedural Question',
				query: 'How should I prepare witnesses for testimony?',
				expected: 'procedure guidance'
			},
			{
				name: 'Document Authentication',
				query: 'What are the requirements for authenticating digital evidence?',
				expected: 'authentication standards'
			}
		];

		const results = {};

		for (const test of tests) {
			try {
				const startTime = Date.now();

				const response = await fetch('/api/chat', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						message: test.query,
						caseId: selectedCaseId,
						options: { stream: false }
					})
				});

				const data = await response.json();
				const responseTime = Date.now() - startTime;

				results[test.name] = {
					status: response.ok ? 'pass' : 'fail',
					query: test.query,
					response: data.response?.substring(0, 200) + '...',
					sources: data.sources?.length || 0,
					responseTime,
					tokensUsed: data.tokensUsed || 0
				};
			} catch (error) {
				results[test.name] = {
					status: 'error',
					query: test.query,
					error: error.message
				};
			}
		}

		testResults.set(results);
	}

	/**
	 * Test streaming functionality
	 */
	async function testStreaming() {
		// This will be handled by the AIChat component
		console.log('Streaming test initiated through chat interface');
	}

	/**
	 * Get status badge variant
	 */
	function getStatusVariant(status) {
		switch (status) {
			case 'healthy':
			case 'pass':
				return 'success';
			case 'partial':
				return 'warning';
			case 'error':
			case 'fail':
				return 'destructive';
			default:
				return 'secondary';
		}
	}

	/**
	 * Format timestamp for display
	 */
	function formatTime(timestamp) {
		return new Date(timestamp).toLocaleTimeString();
	}
</script>

<svelte:head>
	<title>Legal AI Assistant - RAG System Test</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-gray-100 p-6">
	<div class="max-w-7xl mx-auto space-y-6">
		<!-- Header -->
		<div class="text-center space-y-2">
			<h1 class="text-4xl font-bold text-amber-400">
				Legal AI Assistant - RAG System Test
			</h1>
			<p class="text-gray-300">
				Comprehensive testing and validation of the RAG-powered legal AI system
			</p>
		</div>

		<!-- System Health Overview -->
		<Card class="p-6">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-2xl font-semibold text-amber-400">System Health</h2>
				<div class="flex items-center space-x-2">
					<Badge variant={getStatusVariant($systemHealth.status)}>
						{$systemHealth.status}
					</Badge>
					<Button variant="outline" size="sm" on:click={runHealthChecks}>
						Refresh
					</Button>
				</div>
			</div>

			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
				{#each Object.entries($systemHealth.services) as [key, service]}
					<Card class="p-4 bg-gray-800">
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-medium">{service.name}</h3>
							<Badge variant={getStatusVariant(service.status)} class="text-xs">
								{service.status}
							</Badge>
						</div>

						{#if service.error}
							<p class="text-sm text-red-400">{service.error}</p>
						{:else if service.details}
							<div class="text-xs text-gray-400 space-y-1">
								{#if service.details.models}
									<p>Models: {service.details.models.length}</p>
								{/if}
								{#if service.details.rag}
									<p>Document chunks: {service.details.rag.documentChunks}</p>
									<p>Knowledge entries: {service.details.rag.knowledgeBaseEntries}</p>
								{/if}
								{#if service.details.embedding}
									<p>Embedding model: {service.details.embedding.embeddingModel}</p>
								{/if}
							</div>
						{/if}

						<p class="text-xs text-gray-500 mt-2">
							Last check: {formatTime(service.timestamp)}
						</p>
					</Card>
				{/each}
			</div>
		</Card>

		<!-- Available Models -->
		{#if $availableModels.length > 0}
			<Card class="p-6">
				<h2 class="text-2xl font-semibold text-amber-400 mb-4">Available Models</h2>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{#each $availableModels as model}
						<Card class="p-4 bg-gray-800">
							<h3 class="font-medium mb-2">{model.name}</h3>
							<div class="text-xs text-gray-400 space-y-1">
								<p>Size: {(model.size / 1024 / 1024 / 1024).toFixed(2)} GB</p>
								<p>Modified: {new Date(model.modified_at).toLocaleDateString()}</p>
								{#if model.details?.family}
									<p>Family: {model.details.family}</p>
								{/if}
							</div>
						</Card>
					{/each}
				</div>
			</Card>
		{/if}

		<!-- RAG System Tests -->
		<Card class="p-6">
			<div class="flex items-center justify-between mb-4">
				<h2 class="text-2xl font-semibold text-amber-400">RAG System Tests</h2>
				<Button on:click={runRAGTests} class="bg-amber-600 hover:bg-amber-700">
					Run Tests
				</Button>
			</div>

			{#if Object.keys($testResults).length > 0}
				<div class="space-y-4">
					{#each Object.entries($testResults) as [testName, result]}
						<Card class="p-4 bg-gray-800">
							<div class="flex items-center justify-between mb-2">
								<h3 class="font-medium">{testName}</h3>
								<Badge variant={getStatusVariant(result.status)}>
									{result.status}
								</Badge>
							</div>

							<div class="space-y-2 text-sm">
								<p class="text-gray-300">
									<strong>Query:</strong> {result.query}
								</p>

								{#if result.response}
									<p class="text-gray-300">
										<strong>Response:</strong> {result.response}
									</p>
								{/if}

								{#if result.sources !== undefined}
									<p class="text-gray-400">
										<strong>Sources:</strong> {result.sources} context sources used
									</p>
								{/if}

								{#if result.responseTime}
									<p class="text-gray-400">
										<strong>Performance:</strong>
										{result.responseTime}ms response time,
										{result.tokensUsed} tokens used
									</p>
								{/if}

								{#if result.error}
									<p class="text-red-400">
										<strong>Error:</strong> {result.error}
									</p>
								{/if}
							</div>
						</Card>
					{/each}
				</div>
			{:else}
				<p class="text-gray-400 text-center py-8">
					Click "Run Tests" to validate RAG system functionality
				</p>
			{/if}
		</Card>

		<!-- Interactive Chat Interface -->
		<Card class="p-6">
			<h2 class="text-2xl font-semibold text-amber-400 mb-4">Interactive Chat Test</h2>

			<div class="mb-4 flex items-center space-x-4">
				<label class="text-sm text-gray-300">
					Test Case ID:
					<select
						bind:value={selectedCaseId}
						class="ml-2 bg-gray-800 border border-gray-600 rounded px-2 py-1"
					>
						<option value="CASE-2024-001">CASE-2024-001 (Anderson Embezzlement)</option>
						<option value={null}>No specific case</option>
					</select>
				</label>

				<Badge variant="outline" class="text-xs">
					RAG Enabled: Context retrieval + Legal knowledge
				</Badge>
			</div>

			<div class="h-[600px]">
				<AIChat
					caseId={selectedCaseId}
					sessionId={testSessionId}
					enableRAG={true}
					maxContextChunks={5}
					placeholder="Test RAG functionality: Ask about embezzlement elements, evidence authentication, witness preparation, or case-specific questions..."
				/>
			</div>
		</Card>

		<!-- Quick Test Queries -->
		<Card class="p-6">
			<h2 class="text-2xl font-semibold text-amber-400 mb-4">Quick Test Queries</h2>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div class="space-y-2">
					<h3 class="font-medium text-gray-300">Legal Knowledge Tests:</h3>
					<ul class="text-sm text-gray-400 space-y-1">
						<li>• "What are the elements of embezzlement?"</li>
						<li>• "How do I authenticate digital evidence?"</li>
						<li>• "What are Brady disclosure requirements?"</li>
						<li>• "Explain witness preparation guidelines"</li>
					</ul>
				</div>

				<div class="space-y-2">
					<h3 class="font-medium text-gray-300">RAG Context Tests:</h3>
					<ul class="text-sm text-gray-400 space-y-1">
						<li>• "Analyze the Anderson embezzlement case"</li>
						<li>• "What evidence do we have for financial fraud?"</li>
						<li>• "Timeline of the defendant's access to funds"</li>
						<li>• "Recommend prosecution strategy"</li>
					</ul>
				</div>
			</div>
		</Card>

		<!-- Footer -->
		<div class="text-center text-gray-500 text-sm">
			<p>Legal AI Assistant v2.0.0 - RAG-Powered System</p>
			<p class="mt-1">
				System Status:
				<span class="text-{$systemHealth.status === 'healthy' ? 'green' : 'amber'}-400">
					{$systemHealth.status}
				</span>
				{#if $systemHealth.lastCheck}
					| Last Check: {formatTime($systemHealth.lastCheck)}
				{/if}
			</p>
		</div>
	</div>
</div>

<style>
	:global(body) {
		background-color: #111827;
		font-family: 'Inter', sans-serif;
	}
</style>
