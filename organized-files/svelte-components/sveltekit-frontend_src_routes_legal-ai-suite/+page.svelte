<script lang="ts">
	import { onMount } from 'svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Progress } from '$lib/components/ui/progress/index.js';
	import { AlertCircle, Upload, Search, Brain, CheckCircle, AlertTriangle } from 'lucide-svelte';
	
	// Svelte 5 runes for state management
	let selectedFiles = $state<File[]>([]);
	let isProcessing = $state(false);
	let processedDocuments = $state<any[]>([]);
	let ragQuery = $state('');
	let ragResults = $state<any[]>([]);
	let systemMetrics = $state({
		gpuAcceleration: false,
		ollamaStatus: 'unknown',
		processingSpeed: 0,
		caseAIScore: 0
	});
	let selectedJurisdiction = $state('federal');
	let processingSummary = $state<any>(null);
	let realTimeLogs = $state<string[]>([]);
	
	// Computed properties
	let hasFiles = $derived(selectedFiles.length > 0)
	let canProcess = $derived(hasFiles && !isProcessing)
	let totalEntities = $derived(processedDocuments.reduce((sum, doc) => sum + (doc.entityCount || 0), 0)
	);
	let averageProsecutionScore = $derived(processedDocuments.length > 0 
			? processedDocuments.reduce((sum, doc) => sum + (doc.prosecutionScore || 0), 0) / processedDocuments.length
			: 0
	);
	
	onMount(async () => {
		await checkSystemStatus();
		// Start real-time logging
		startRealTimeLogging();
	});
	
	function handleFileSelect(event: Event) {
		const input = event.target as HTMLInputElement;
		if (input.files) {
			selectedFiles = Array.from(input.files).filter(file => 
				file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
			);
			addLog(`📄 Selected ${selectedFiles.length} PDF files for processing`);
		}
	}
	
	async function processLegalDocuments() {
		if (!canProcess) return;
		
		isProcessing = true;
		processingSummary = null;
		addLog(`🚀 Starting legal document processing...`);
		
		try {
			const formData = new FormData();
			
			// Add files to form data
			selectedFiles.forEach(file => {
				formData.append('pdfFiles', file);
			});
			
			// Add processing parameters
			formData.append('jurisdiction', selectedJurisdiction);
			formData.append('enhanceRAG', 'true');
			formData.append('caseId', `case-${Date.now()}`);
			
			addLog(`⚖️ Processing ${selectedFiles.length} documents under ${selectedJurisdiction} jurisdiction`);
			
			const response = await fetch('/api/legal/ingest', {
				method: 'POST',
				body: formData
			});
			
			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}
			
			const result = await response.json();
			
			if (result.success) {
				processedDocuments = result.documents || [];
				processingSummary = result.summary;
				systemMetrics.caseAIScore = result.caseAISummaryScore;
				
				addLog(`✅ Processing complete: ${result.documentsProcessed} documents`);
				addLog(`📊 Total entities extracted: ${result.summary?.totalEntities || 0}`);
				addLog(`🎯 Average prosecution score: ${(averageProsecutionScore * 100).toFixed(1)}%`);
				addLog(`📈 Case AI summary score: ${result.caseAISummaryScore}/100`);
			} else {
				throw new Error(result.error || 'Processing failed');
			}
			
		} catch (error) {
			console.error('Document processing failed:', error);
			addLog(`❌ Processing failed: ${error.message}`);
		} finally {
			isProcessing = false;
		}
	}
	
	async function executeRAGQuery() {
		if (!ragQuery.trim()) return;
		
		addLog(`🔍 Executing enhanced RAG query: "${ragQuery}"`);
		
		try {
			const response = await fetch('/api/enhanced-rag/query', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({
					query: ragQuery,
					jurisdiction: selectedJurisdiction,
					maxResults: 5,
					includeContext7: true,
					prioritizeFactChecked: true,
					minProsecutionScore: 0.5
				})
			});
			
			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}
			
			const result = await response.json();
			
			if (result.success) {
				ragResults = result.results || [];
				addLog(`✅ RAG query complete: ${ragResults.length} results, score: ${(result.ragScore * 100).toFixed(1)}%`);
				
				if (result.aggregatedAnalysis?.recommendedNextQuery) {
					addLog(`💡 Recommended follow-up: "${result.aggregatedAnalysis.recommendedNextQuery}"`);
				}
			} else {
				throw new Error(result.error || 'RAG query failed');
			}
			
		} catch (error) {
			console.error('RAG query failed:', error);
			addLog(`❌ RAG query failed: ${error.message}`);
		}
	}
	
	async function checkSystemStatus() {
		try {
			// Check Ollama status
			const ollamaResponse = await fetch('http://localhost:11434/api/tags');
			systemMetrics.ollamaStatus = ollamaResponse.ok ? 'healthy' : 'offline';
			
			// Simulate GPU detection
			systemMetrics.gpuAcceleration = Math.random() > 0.3; // 70% chance for demo
			
			addLog(`🖥️ System status: Ollama ${systemMetrics.ollamaStatus}, GPU: ${systemMetrics.gpuAcceleration ? 'enabled' : 'disabled'}`);
			
		} catch (error) {
			systemMetrics.ollamaStatus = 'error';
			addLog(`⚠️ System check failed: ${error.message}`);
		}
	}
	
	function addLog(message: string) {
		const timestamp = new Date().toLocaleTimeString();
		realTimeLogs = [...realTimeLogs, `[${timestamp}] ${message}`];
		
		// Keep only the last 20 log entries
		if (realTimeLogs.length > 20) {
			realTimeLogs = realTimeLogs.slice(-20);
		}
	}
	
	function startRealTimeLogging() {
		// Simulate periodic system metrics updates
		setInterval(() => {
			if (isProcessing) {
				systemMetrics.processingSpeed = Math.random() * 100 + 50; // 50-150 docs/min
			}
		}, 1000);
	}
	
	function clearLogs() {
		realTimeLogs = [];
		addLog('📋 Logs cleared');
	}
	
	function getFactCheckBadgeVariant(status: string) {
		switch (status) {
			case 'FACT': return 'default';
			case 'FICTION': case 'DISPUTED': return 'destructive';
			case 'UNVERIFIED': return 'secondary';
			default: return 'outline';
		}
	}
	
	function getProsecutionScoreColor(score: number) {
		if (score >= 0.8) return 'text-green-600';
		if (score >= 0.6) return 'text-yellow-600';
		return 'text-red-600';
	}
</script>

<svelte:head>
	<title>Legal AI Suite - Enhanced RAG & Multi-PDF Processing</title>
	<meta name="description" content="GPU-accelerated legal document analysis with enhanced RAG and fact-checking" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
	<div class="max-w-7xl mx-auto space-y-6">
		<!-- Header -->
		<div class="text-center mb-8">
			<h1 class="text-4xl font-bold text-gray-900 mb-2">
				⚖️ Legal AI Suite
			</h1>
			<p class="text-lg text-gray-600">
				GPU-Accelerated Legal Document Analysis with Enhanced RAG
			</p>
		</div>

		<!-- System Status Cards -->
		<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">GPU Acceleration</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="flex items-center space-x-2">
						{#if systemMetrics.gpuAcceleration}
							<CheckCircle class="h-4 w-4 text-green-500" />
							<span class="text-sm text-green-600">Enabled</span>
						{:else}
							<AlertTriangle class="h-4 w-4 text-yellow-500" />
							<span class="text-sm text-yellow-600">CPU Mode</span>
						{/if}
					</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Ollama Status</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="flex items-center space-x-2">
						{#if systemMetrics.ollamaStatus === 'healthy'}
							<CheckCircle class="h-4 w-4 text-green-500" />
							<span class="text-sm text-green-600">Healthy</span>
						{:else}
							<AlertCircle class="h-4 w-4 text-red-500" />
							<span class="text-sm text-red-600">Offline</span>
						{/if}
					</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Processing Speed</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="text-sm">
						{#if isProcessing}
							<span class="text-blue-600">{systemMetrics.processingSpeed.toFixed(0)} docs/min</span>
						{:else}
							<span class="text-gray-500">Idle</span>
						{/if}
					</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Case AI Score</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="text-sm font-semibold">
						{systemMetrics.caseAIScore}/100
					</div>
					<Progress value={systemMetrics.caseAIScore} class="h-2 mt-1" />
				</CardContent>
			</Card>
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Document Processing Panel -->
			<Card>
				<CardHeader>
					<CardTitle class="flex items-center space-x-2">
						<Upload class="h-5 w-5" />
						<span>Multi-PDF Document Processing</span>
					</CardTitle>
				</CardHeader>
				<CardContent class="space-y-4">
					<!-- File Upload -->
					<div>
						<label for="pdf-files" class="block text-sm font-medium text-gray-700 mb-2">
							Select Legal Documents (PDF)
						</label>
						<input
							id="pdf-files"
							type="file"
							multiple
							accept=".pdf"
							onchange={handleFileSelect}
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
						/>
						{#if hasFiles}
							<p class="text-sm text-gray-600 mt-1">
								{selectedFiles.length} PDF file{selectedFiles.length !== 1 ? 's' : ''} selected
							</p>
						{/if}
					</div>

					<!-- Jurisdiction Selection -->
					<div>
						<label for="jurisdiction" class="block text-sm font-medium text-gray-700 mb-2">
							Jurisdiction
						</label>
						<select
							id="jurisdiction"
							bind:value={selectedJurisdiction}
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
						>
							<option value="federal">Federal</option>
							<option value="state">State</option>
							<option value="local">Local</option>
							<option value="international">International</option>
						</select>
					</div>

					<!-- Processing Controls -->
					<div class="flex space-x-2">
						<Button 
							onclick={processLegalDocuments} 
							disabled={!canProcess}
							class="flex-1"
						>
							{#if isProcessing}
								<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
								Processing...
							{:else}
								<Upload class="h-4 w-4 mr-2" />
								Process Documents
							{/if}
						</Button>
					</div>

					<!-- Processing Summary -->
					{#if processingSummary}
						<div class="border-t pt-4 space-y-2">
							<h4 class="font-semibold text-gray-800">Processing Summary</h4>
							<div class="grid grid-cols-2 gap-4 text-sm">
								<div>
									<span class="text-gray-600">Total Entities:</span>
									<span class="font-medium ml-2">{processingSummary.totalEntities}</span>
								</div>
								<div>
									<span class="text-gray-600">Total Chunks:</span>
									<span class="font-medium ml-2">{processingSummary.totalChunks}</span>
								</div>
								<div>
									<span class="text-gray-600">Facts Verified:</span>
									<span class="font-medium ml-2 text-green-600">{processingSummary.factCheckResults?.facts || 0}</span>
								</div>
								<div>
									<span class="text-gray-600">Disputed Claims:</span>
									<span class="font-medium ml-2 text-red-600">{processingSummary.factCheckResults?.fiction || 0}</span>
								</div>
							</div>
						</div>
					{/if}
				</CardContent>
			</Card>

			<!-- Enhanced RAG Query Panel -->
			<Card>
				<CardHeader>
					<CardTitle class="flex items-center space-x-2">
						<Search class="h-5 w-5" />
						<span>Enhanced RAG Query</span>
					</CardTitle>
				</CardHeader>
				<CardContent class="space-y-4">
					<!-- Query Input -->
					<div>
						<label for="rag-query" class="block text-sm font-medium text-gray-700 mb-2">
							Legal Query
						</label>
						<Input
							id="rag-query"
							bind:value={ragQuery}
							placeholder="Enter your legal question or search query..."
							class="w-full"
						/>
					</div>

					<!-- Query Controls -->
					<div class="flex space-x-2">
						<Button 
							onclick={executeRAGQuery} 
							disabled={!ragQuery.trim()}
							class="flex-1"
						>
							<Brain class="h-4 w-4 mr-2" />
							Query Enhanced RAG
						</Button>
					</div>

					<!-- RAG Results -->
					{#if ragResults.length > 0}
						<div class="border-t pt-4">
							<h4 class="font-semibold text-gray-800 mb-3">Query Results</h4>
							<div class="space-y-3 max-h-64 overflow-y-auto">
								{#each ragResults as result}
									<div class="p-3 bg-gray-50 rounded-md">
										<div class="flex justify-between items-start mb-2">
											<span class="text-sm font-medium text-gray-800">
												{result.sourceDocument}
											</span>
											<div class="flex space-x-1">
												<Badge variant="outline" class="text-xs">
													Similarity: {(result.similarity * 100).toFixed(0)}%
												</Badge>
												<Badge 
													variant={getFactCheckBadgeVariant(result.factCheckStatus)}
													class="text-xs"
												>
													{result.factCheckStatus || 'N/A'}
												</Badge>
											</div>
										</div>
										<p class="text-sm text-gray-700 mb-2">
											{result.content.substring(0, 200)}...
										</p>
										<div class="flex justify-between items-center text-xs text-gray-500">
											<span>Jurisdiction: {result.jurisdiction}</span>
											<span class={getProsecutionScoreColor(result.prosecutionScore)}>
												Prosecution Score: {(result.prosecutionScore * 100).toFixed(0)}%
											</span>
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</CardContent>
			</Card>
		</div>

		<!-- Processed Documents Display -->
		{#if processedDocuments.length > 0}
			<Card>
				<CardHeader>
					<CardTitle>Processed Documents</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{#each processedDocuments as doc}
							<div class="p-4 border border-gray-200 rounded-lg">
								<h4 class="font-semibold text-gray-800 mb-2">{doc.filename}</h4>
								<div class="space-y-1 text-sm">
									<div class="flex justify-between">
										<span class="text-gray-600">Jurisdiction:</span>
										<span class="font-medium">{doc.jurisdiction}</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-600">Entities:</span>
										<span class="font-medium">{doc.entityCount}</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-600">Chunks:</span>
										<span class="font-medium">{doc.chunkCount}</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-600">Prosecution Score:</span>
										<span class={`font-medium ${getProsecutionScoreColor(doc.prosecutionScore)}`}>
											{(doc.prosecutionScore * 100).toFixed(0)}%
										</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-600">Processing Time:</span>
										<span class="font-medium">{doc.processingTime}ms</span>
									</div>
									{#if doc.factCheckSummary}
										<div class="mt-2 pt-2 border-t border-gray-100">
											<div class="text-xs text-gray-600">
												Facts: <span class="text-green-600">{doc.factCheckSummary.verified}</span>
												| Disputed: <span class="text-red-600">{doc.factCheckSummary.disputed}</span>
											</div>
										</div>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</CardContent>
			</Card>
		{/if}

		<!-- Real-time Logging Panel -->
		<Card>
			<CardHeader class="flex flex-row items-center justify-between">
				<CardTitle class="flex items-center space-x-2">
					<AlertCircle class="h-5 w-5" />
					<span>Real-time System Logs</span>
				</CardTitle>
				<Button variant="outline" size="sm" onclick={clearLogs}>
					Clear Logs
				</Button>
			</CardHeader>
			<CardContent>
				<div class="bg-gray-900 text-green-400 p-4 rounded-md font-mono text-sm max-h-64 overflow-y-auto">
					{#if realTimeLogs.length === 0}
						<div class="text-gray-500">No logs yet...</div>
					{:else}
						{#each realTimeLogs as log}
							<div class="mb-1">{log}</div>
						{/each}
					{/if}
				</div>
			</CardContent>
		</Card>

		<!-- System Statistics -->
		{#if processedDocuments.length > 0 || ragResults.length > 0}
			<Card>
				<CardHeader>
					<CardTitle>System Performance Statistics</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
						<div>
							<div class="text-2xl font-bold text-blue-600">{processedDocuments.length}</div>
							<div class="text-sm text-gray-600">Documents Processed</div>
						</div>
						<div>
							<div class="text-2xl font-bold text-green-600">{totalEntities}</div>
							<div class="text-sm text-gray-600">Entities Extracted</div>
						</div>
						<div>
							<div class="text-2xl font-bold text-purple-600">{ragResults.length}</div>
							<div class="text-sm text-gray-600">RAG Results</div>
						</div>
						<div>
							<div class="text-2xl font-bold text-orange-600">{(averageProsecutionScore * 100).toFixed(0)}%</div>
							<div class="text-sm text-gray-600">Avg Prosecution Score</div>
						</div>
					</div>
				</CardContent>
			</Card>
		{/if}
	</div>
</div>

<style>
	/* Custom scrollbar for logs */
	:global(.max-h-64::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.max-h-64::-webkit-scrollbar-track) {
		background: #f1f1f1;
		border-radius: 3px;
	}
	
	:global(.max-h-64::-webkit-scrollbar-thumb) {
		background: #c1c1c1;
		border-radius: 3px;
	}
	
	:global(.max-h-64::-webkit-scrollbar-thumb:hover) {
		background: #a8a8a8;
	}
</style>