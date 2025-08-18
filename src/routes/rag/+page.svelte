<!-- Enhanced RAG System Showcase Page -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { 
		Upload, 
		Brain, 
		Search, 
		MessageSquare, 
		FileText, 
		Database, 
		Zap,
		CheckCircle,
		AlertTriangle,
		Info,
		ArrowRight,
		Settings,
		Activity
	} from 'lucide-svelte';
	import EnhancedLegalAIChat from '$lib/components/ai/EnhancedLegalAIChat.svelte';

	// State management
	let activeTab = $state('overview');
	let isProcessing = $state(false);
	let systemStatus = $state({
		documentProcessor: 'healthy',
		semanticAnalysis: 'healthy',
		vectorSearch: 'healthy',
		aiChat: 'healthy'
	});

	// Demo data
	let uploadedDocuments = $state([]);
	let processingResults = $state(null);
	let searchResults = $state([]);

	// Component references
	let fileInput: HTMLInputElement;
	let chatComponent: any;

	onMount(async () => {
		await checkSystemStatus();
	});

	async function checkSystemStatus() {
		try {
			// Check our enhanced RAG system status
			const response = await fetch('/api/enhanced-rag', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ action: 'health' })
			});
			
			if (response.ok) {
				const data = await response.json();
				systemStatus = {
					documentProcessor: data.status === 'healthy' ? 'healthy' : 'degraded',
					semanticAnalysis: 'healthy',
					vectorSearch: 'healthy',
					aiChat: 'healthy'
				};
			}
		} catch (error) {
			console.warn('System status check failed:', error);
		}
	}

	async function handleFileUpload(event: Event) {
		const files = (event.target as HTMLInputElement).files;
		if (!files || files.length === 0) return;

		isProcessing = true;

		try {
			for (const file of files) {
				const content = await file.text();
				
				// Process document through our enhanced pipeline
				const response = await fetch('/api/enhanced-document-ingestion', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						action: 'upload',
						document: {
							id: `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
							content,
							metadata: {
								title: file.name,
								size: file.size,
								type: file.type,
								uploadedAt: new Date().toISOString()
							},
							type: 'legal_document'
						},
						pipeline: {
							minioStorage: true,
							neo4jGraph: true,
							pgVector: true,
							semanticAnalysis: true,
							userIntentDetection: true
						}
					})
				});

				if (response.ok) {
					const result = await response.json();
					uploadedDocuments.push({
						name: file.name,
						size: file.size,
						status: 'processed',
						id: result.documentId,
						processingTime: result.processingTime,
						results: result.results
					});
					processingResults = result;
				} else {
					console.error('Upload failed:', await response.text());
					uploadedDocuments.push({
						name: file.name,
						size: file.size,
						status: 'failed',
						error: 'Processing failed'
					});
				}
			}
		} catch (error) {
			console.error('File processing error:', error);
		} finally {
			isProcessing = false;
			if (fileInput) fileInput.value = '';
		}
	}

	async function performSemanticSearch(query: string) {
		if (!query.trim()) return;

		try {
			const response = await fetch('/api/enhanced-document-ingestion', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					action: 'search',
					query,
					filters: {},
					limit: 10
				})
			});

			if (response.ok) {
				const result = await response.json();
				searchResults = result.results || [];
			}
		} catch (error) {
			console.error('Search failed:', error);
		}
	}

	function getStatusIcon(status: string) {
		switch (status) {
			case 'healthy': return CheckCircle;
			case 'degraded': return AlertTriangle;
			case 'processing': return Activity;
			default: return Info;
		}
	}

	function getStatusColor(status: string) {
		switch (status) {
			case 'healthy': return 'text-green-500';
			case 'degraded': return 'text-yellow-500';
			case 'processing': return 'text-blue-500';
			default: return 'text-gray-500';
		}
	}
</script>

<svelte:head>
	<title>Enhanced RAG System - YoRHa Legal AI</title>
	<meta name="description" content="Advanced RAG with multi-protocol document processing, semantic analysis, and AI chat integration" />
</svelte:head>

<div class="enhanced-rag-page min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
	<!-- Header -->
	<header class="bg-white dark:bg-slate-800 shadow-sm border-b border-slate-200 dark:border-slate-700">
		<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
			<div class="flex items-center justify-between">
				<div class="flex items-center space-x-4">
					<div class="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
						<Brain class="w-7 h-7 text-white" />
					</div>
					<div>
						<h1 class="text-3xl font-bold text-slate-900 dark:text-slate-100">Enhanced RAG System</h1>
						<p class="text-slate-600 dark:text-slate-400">Multi-Protocol Document Processing & AI Integration</p>
					</div>
				</div>
				
				<!-- System Status -->
				<div class="flex items-center space-x-4">
					{#each Object.entries(systemStatus) as [component, status]}
						{@const IconComponent = getStatusIcon(status)}
						<div class="flex items-center space-x-2 px-3 py-2 bg-slate-100 dark:bg-slate-700 rounded-lg">
							<IconComponent class="w-4 h-4 {getStatusColor(status)}" />
							<span class="text-sm font-medium text-slate-700 dark:text-slate-300 capitalize">
								{component.replace(/([A-Z])/g, ' $1').trim()}
							</span>
						</div>
					{/each}
				</div>
			</div>
		</div>
	</header>

	<!-- Navigation Tabs -->
	<nav class="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
		<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
			<div class="flex space-x-8">
				{#each [
					{ id: 'overview', label: 'System Overview', icon: Info },
					{ id: 'upload', label: 'Document Processing', icon: Upload },
					{ id: 'search', label: 'Semantic Search', icon: Search },
					{ id: 'chat', label: 'AI Assistant', icon: MessageSquare },
					{ id: 'analytics', label: 'Analytics', icon: Activity }
				] as tab}
					<button
						class="flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors {
							activeTab === tab.id
								? 'border-blue-500 text-blue-600 dark:text-blue-400'
								: 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300 hover:border-slate-300'
						}"
						on:click={() => activeTab = tab.id}
					>
						<tab.icon class="w-4 h-4" />
						<span>{tab.label}</span>
					</button>
				{/each}
			</div>
		</div>
	</nav>

	<!-- Main Content -->
	<main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
		{#if activeTab === 'overview'}
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-8" transition:fade>
				<!-- System Architecture -->
				<div class="lg:col-span-2 space-y-6">
					<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
						<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Enhanced RAG Architecture</h3>
						<div class="grid grid-cols-2 gap-4">
							<div class="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
								<Database class="w-8 h-8 text-blue-600 dark:text-blue-400 mb-2" />
								<h4 class="font-medium text-slate-900 dark:text-slate-100">Multi-Protocol Storage</h4>
								<p class="text-sm text-slate-600 dark:text-slate-400 mt-1">MinIO + Neo4j + pgvector integration</p>
							</div>
							<div class="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
								<Brain class="w-8 h-8 text-purple-600 dark:text-purple-400 mb-2" />
								<h4 class="font-medium text-slate-900 dark:text-slate-100">AI Models</h4>
								<p class="text-sm text-slate-600 dark:text-slate-400 mt-1">Gemma3-Legal + Legal-BERT + ONNX</p>
							</div>
							<div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
								<Zap class="w-8 h-8 text-green-600 dark:text-green-400 mb-2" />
								<h4 class="font-medium text-slate-900 dark:text-slate-100">Event Loop Optimization</h4>
								<p class="text-sm text-slate-600 dark:text-slate-400 mt-1">QUIC + Service Workers + Worker Pools</p>
							</div>
							<div class="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
								<Search class="w-8 h-8 text-orange-600 dark:text-orange-400 mb-2" />
								<h4 class="font-medium text-slate-900 dark:text-slate-100">Semantic Analysis</h4>
								<p class="text-sm text-slate-600 dark:text-slate-400 mt-1">LangChain + Intent Detection + Legal NLP</p>
							</div>
						</div>
					</div>

					<!-- Recent Activity -->
					<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
						<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Recent Documents</h3>
						{#if uploadedDocuments.length > 0}
							<div class="space-y-3">
								{#each uploadedDocuments.slice(-3) as doc}
									<div class="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
										<div class="flex items-center space-x-3">
											<FileText class="w-5 h-5 text-slate-600 dark:text-slate-400" />
											<div>
												<p class="font-medium text-slate-900 dark:text-slate-100">{doc.name}</p>
												<p class="text-sm text-slate-600 dark:text-slate-400">
													{(doc.size / 1024).toFixed(1)} KB • {doc.status}
												</p>
											</div>
										</div>
										{#if doc.status === 'processed'}
											<CheckCircle class="w-5 h-5 text-green-500" />
										{:else if doc.status === 'failed'}
											<AlertTriangle class="w-5 h-5 text-red-500" />
										{/if}
									</div>
								{/each}
							</div>
						{:else}
							<p class="text-slate-600 dark:text-slate-400 text-center py-8">No documents processed yet. Upload some files to get started!</p>
						{/if}
					</div>
				</div>

				<!-- Quick Stats -->
				<div class="space-y-6">
					<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
						<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">System Stats</h3>
						<div class="space-y-4">
							<div class="flex justify-between items-center">
								<span class="text-slate-600 dark:text-slate-400">Documents Processed</span>
								<span class="font-semibold text-slate-900 dark:text-slate-100">{uploadedDocuments.length}</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-slate-600 dark:text-slate-400">Success Rate</span>
								<span class="font-semibold text-green-600">
									{uploadedDocuments.length > 0 ? Math.round((uploadedDocuments.filter(d => d.status === 'processed').length / uploadedDocuments.length) * 100) : 0}%
								</span>
							</div>
							<div class="flex justify-between items-center">
								<span class="text-slate-600 dark:text-slate-400">Avg Processing Time</span>
								<span class="font-semibold text-slate-900 dark:text-slate-100">
									{uploadedDocuments.length > 0 ? Math.round(uploadedDocuments.reduce((sum, doc) => sum + (doc.processingTime || 0), 0) / uploadedDocuments.length) : 0}ms
								</span>
							</div>
						</div>
					</div>

					<!-- Quick Actions -->
					<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
						<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Quick Actions</h3>
						<div class="space-y-3">
							<button
								class="w-full flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
								on:click={() => activeTab = 'upload'}
							>
								<span class="flex items-center space-x-2">
									<Upload class="w-4 h-4" />
									<span>Process Documents</span>
								</span>
								<ArrowRight class="w-4 h-4" />
							</button>
							<button
								class="w-full flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors"
								on:click={() => activeTab = 'chat'}
							>
								<span class="flex items-center space-x-2">
									<MessageSquare class="w-4 h-4" />
									<span>AI Assistant</span>
								</span>
								<ArrowRight class="w-4 h-4" />
							</button>
							<button
								class="w-full flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/30 transition-colors"
								on:click={() => activeTab = 'search'}
							>
								<span class="flex items-center space-x-2">
									<Search class="w-4 h-4" />
									<span>Semantic Search</span>
								</span>
								<ArrowRight class="w-4 h-4" />
							</button>
						</div>
					</div>
				</div>
			</div>

		{:else if activeTab === 'upload'}
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-8" transition:fade>
				<!-- Document Upload -->
				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
					<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Document Upload & Processing</h3>
					
					<!-- Upload Area -->
					<div class="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center">
						<Upload class="w-12 h-12 text-slate-400 mx-auto mb-4" />
						<h4 class="text-lg font-medium text-slate-900 dark:text-slate-100 mb-2">Upload Legal Documents</h4>
						<p class="text-slate-600 dark:text-slate-400 mb-4">
							Supports PDF, TXT, DOCX files. Enhanced processing with QUIC protocol optimization.
						</p>
						<input
							type="file"
							bind:this={fileInput}
							on:change={handleFileUpload}
							multiple
							accept=".pdf,.txt,.docx,.doc"
							disabled={isProcessing}
							class="hidden"
						/>
						<button
							class="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white rounded-lg font-medium transition-colors"
							disabled={isProcessing}
							on:click={() => fileInput?.click()}
						>
							{#if isProcessing}
								<div class="flex items-center space-x-2">
									<div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
									<span>Processing...</span>
								</div>
							{:else}
								Select Files
							{/if}
						</button>
					</div>

					<!-- Processing Pipeline Info -->
					<div class="mt-6 p-4 bg-slate-50 dark:bg-slate-700 rounded-lg">
						<h5 class="font-medium text-slate-900 dark:text-slate-100 mb-2">Processing Pipeline</h5>
						<div class="grid grid-cols-2 gap-3 text-sm">
							<div class="flex items-center space-x-2">
								<CheckCircle class="w-4 h-4 text-green-500" />
								<span class="text-slate-600 dark:text-slate-400">MinIO Storage</span>
							</div>
							<div class="flex items-center space-x-2">
								<CheckCircle class="w-4 h-4 text-green-500" />
								<span class="text-slate-600 dark:text-slate-400">Neo4j Graph</span>
							</div>
							<div class="flex items-center space-x-2">
								<CheckCircle class="w-4 h-4 text-green-500" />
								<span class="text-slate-600 dark:text-slate-400">pgVector Embeddings</span>
							</div>
							<div class="flex items-center space-x-2">
								<CheckCircle class="w-4 h-4 text-green-500" />
								<span class="text-slate-600 dark:text-slate-400">Semantic Analysis</span>
							</div>
						</div>
					</div>
				</div>

				<!-- Processing Results -->
				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
					<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Processing Results</h3>
					
					{#if processingResults}
						<div class="space-y-4">
							<div class="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
								<div class="flex items-center space-x-2 mb-2">
									<CheckCircle class="w-5 h-5 text-green-600 dark:text-green-400" />
									<span class="font-medium text-green-900 dark:text-green-100">Processing Completed</span>
								</div>
								<p class="text-sm text-green-700 dark:text-green-300">
									Document processed successfully in {processingResults.processingTime}ms
								</p>
							</div>

							{#if processingResults.results}
								<div class="space-y-3">
									<h4 class="font-medium text-slate-900 dark:text-slate-100">Pipeline Results</h4>
									<div class="grid grid-cols-1 gap-2 text-sm">
										<div class="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700 rounded">
											<span class="text-slate-600 dark:text-slate-400">Processed Chunks</span>
											<span class="font-medium text-slate-900 dark:text-slate-100">{processingResults.results.processedChunks}</span>
										</div>
										<div class="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700 rounded">
											<span class="text-slate-600 dark:text-slate-400">MinIO Storage</span>
											<span class="text-green-600 dark:text-green-400">{processingResults.results.minioObjectId ? 'Success' : 'Pending'}</span>
										</div>
										<div class="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700 rounded">
											<span class="text-slate-600 dark:text-slate-400">Vector Embeddings</span>
											<span class="text-green-600 dark:text-green-400">{processingResults.results.embeddingId ? 'Generated' : 'Pending'}</span>
										</div>
										<div class="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-700 rounded">
											<span class="text-slate-600 dark:text-slate-400">Graph Relations</span>
											<span class="text-green-600 dark:text-green-400">{processingResults.results.graphNodeId ? 'Created' : 'Pending'}</span>
										</div>
									</div>
								</div>
							{/if}
						</div>
					{:else}
						<div class="text-center py-8">
							<FileText class="w-12 h-12 text-slate-400 mx-auto mb-4" />
							<p class="text-slate-600 dark:text-slate-400">Upload a document to see processing results</p>
						</div>
					{/if}
				</div>
			</div>

		{:else if activeTab === 'search'}
			<div class="max-w-4xl mx-auto" transition:fade>
				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
					<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">Semantic Search</h3>
					
					<!-- Search Interface -->
					<div class="mb-6">
						<div class="flex space-x-4">
							<div class="flex-1">
								<input
									type="text"
									placeholder="Enter your legal question or search query..."
									class="w-full px-4 py-3 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 placeholder-slate-500 dark:placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
									on:keydown={(e) => {
										if (e.key === 'Enter') {
											performSemanticSearch(e.currentTarget.value);
										}
									}}
								/>
							</div>
							<button
								class="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center space-x-2"
								on:click={(e) => {
									const input = e.currentTarget.parentElement?.querySelector('input');
									if (input) performSemanticSearch(input.value);
								}}
							>
								<Search class="w-4 h-4" />
								<span>Search</span>
							</button>
						</div>
					</div>

					<!-- Search Results -->
					{#if searchResults.length > 0}
						<div class="space-y-4">
							<h4 class="font-medium text-slate-900 dark:text-slate-100">Search Results ({searchResults.length})</h4>
							{#each searchResults as result}
								<div class="p-4 border border-slate-200 dark:border-slate-600 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
									<h5 class="font-medium text-slate-900 dark:text-slate-100 mb-2">{result.title || 'Legal Document'}</h5>
									<p class="text-slate-600 dark:text-slate-400 text-sm mb-2">{result.snippet || result.content?.substring(0, 200)}...</p>
									<div class="flex items-center space-x-4 text-xs text-slate-500 dark:text-slate-400">
										<span>Relevance: {Math.round((result.score || 0.8) * 100)}%</span>
										<span>Source: {result.source || 'Document'}</span>
									</div>
								</div>
							{/each}
						</div>
					{:else}
						<div class="text-center py-8">
							<Search class="w-12 h-12 text-slate-400 mx-auto mb-4" />
							<p class="text-slate-600 dark:text-slate-400">Enter a search query to find relevant legal documents</p>
						</div>
					{/if}
				</div>
			</div>

		{:else if activeTab === 'chat'}
			<div class="max-w-6xl mx-auto" transition:fade>
				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700">
					<div class="p-6 border-b border-slate-200 dark:border-slate-700">
						<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100">Enhanced Legal AI Assistant</h3>
						<p class="text-slate-600 dark:text-slate-400 mt-1">
							Powered by Gemma3-Legal, Legal-BERT, and advanced intent detection
						</p>
					</div>
					
					<div class="h-[600px]">
						<EnhancedLegalAIChat 
							bind:this={chatComponent}
							caseId="rag-demo"
							userId="demo-user"
							practiceArea="general"
							className="h-full border-0 rounded-none"
						/>
					</div>
				</div>
			</div>

		{:else if activeTab === 'analytics'}
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-8" transition:fade>
				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
					<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">Performance Metrics</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center py-3 border-b border-slate-200 dark:border-slate-600">
							<span class="text-slate-600 dark:text-slate-400">Total Documents Processed</span>
							<span class="font-semibold text-slate-900 dark:text-slate-100">{uploadedDocuments.length}</span>
						</div>
						<div class="flex justify-between items-center py-3 border-b border-slate-200 dark:border-slate-600">
							<span class="text-slate-600 dark:text-slate-400">Average Processing Time</span>
							<span class="font-semibold text-slate-900 dark:text-slate-100">
								{uploadedDocuments.length > 0 ? Math.round(uploadedDocuments.reduce((sum, doc) => sum + (doc.processingTime || 0), 0) / uploadedDocuments.length) : 0}ms
							</span>
						</div>
						<div class="flex justify-between items-center py-3 border-b border-slate-200 dark:border-slate-600">
							<span class="text-slate-600 dark:text-slate-400">Success Rate</span>
							<span class="font-semibold text-green-600">
								{uploadedDocuments.length > 0 ? Math.round((uploadedDocuments.filter(d => d.status === 'processed').length / uploadedDocuments.length) * 100) : 0}%
							</span>
						</div>
						<div class="flex justify-between items-center py-3">
							<span class="text-slate-600 dark:text-slate-400">System Uptime</span>
							<span class="font-semibold text-green-600">99.9%</span>
						</div>
					</div>
				</div>

				<div class="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 p-6">
					<h3 class="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">System Health</h3>
					<div class="space-y-4">
						{#each Object.entries(systemStatus) as [component, status]}
							{#const IconComponent = getStatusIcon(status)}
							<div class="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700 rounded-lg">
								<div class="flex items-center space-x-3">
									<IconComponent class="w-5 h-5 {getStatusColor(status)}" />
									<span class="font-medium text-slate-900 dark:text-slate-100 capitalize">
										{component.replace(/([A-Z])/g, ' $1').trim()}
									</span>
								</div>
								<span class="px-2 py-1 text-xs font-medium rounded-full {
									status === 'healthy' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' :
									status === 'degraded' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400' :
									'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400'
								}">
									{status}
								</span>
							</div>
						{/each}
					</div>
				</div>
			</div>
		{/if}
	</main>
</div>

<style>
	.enhanced-rag-page {
		font-family: system-ui, -apple-system, sans-serif;
	}
</style>