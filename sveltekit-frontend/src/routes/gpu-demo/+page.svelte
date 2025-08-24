<!-- GPU Integration Demo for Legal AI Platform -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { gpuServiceClient, isGPUAvailable, getGPUUtilization, processLegalDocument } from '$lib/services/gpu-service-client';
	import type { GPUStatus, GPUMetrics, WorkerStatus, ServiceRegistry, GPUResult } from '$lib/types/gpu-services';

	// Component state
	let gpuAvailable = false;
	let loading = true;
	let error = '';
	
	// GPU Status & Metrics
	let gpuStatus: GPUStatus | null = null;
	let gpuMetrics: GPUMetrics | null = null;
	let workers: WorkerStatus[] = [];
	let services: ServiceRegistry | null = null;
	let utilization = 0;

	// Demo forms
	let documentText = `EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on [DATE] between [COMPANY NAME], a [STATE] corporation ("Company"), and [EMPLOYEE NAME] ("Employee").

1. POSITION AND DUTIES
Employee will serve as [POSITION TITLE] and will perform duties and responsibilities typically associated with such position, including but not limited to:
- Managing day-to-day operations
- Overseeing team performance
- Reporting to senior management

2. COMPENSATION
Employee will receive an annual base salary of $[AMOUNT], payable in accordance with Company's standard payroll practices.

3. BENEFITS
Employee will be eligible for Company's standard benefits package including health insurance, dental insurance, and retirement plans.

4. TERMINATION
This agreement may be terminated by either party with thirty (30) days written notice.`;

	let queryText = 'What are the termination provisions in employment contracts?';
	let processingResults: any[] = [];
	let isProcessing = false;

	// Real-time updates
	let metricsInterval: NodeJS.Timeout;
	let statusInterval: NodeJS.Timeout;

	onMount(async () => {
		await loadGPUData();
		startRealtimeUpdates();
		
		return () => {
			if (metricsInterval) clearInterval(metricsInterval);
			if (statusInterval) clearInterval(statusInterval);
		};
	});

	async function loadGPUData() {
		try {
			loading = true;
			error = '';

			// Load all GPU data in parallel
			const [available, status, metrics, workerStatus, serviceRegistry, util] = await Promise.all([
				isGPUAvailable(),
				gpuServiceClient.getStatus(),
				gpuServiceClient.getMetrics(),
				gpuServiceClient.getWorkers(),
				gpuServiceClient.getServices(),
				getGPUUtilization()
			]);

			gpuAvailable = available;
			gpuStatus = status;
			gpuMetrics = metrics;
			workers = workerStatus;
			services = serviceRegistry;
			utilization = util;

		} catch (err: any) {
			error = `Failed to load GPU data: ${err.message}`;
			console.error('GPU data loading error:', err);
		} finally {
			loading = false;
		}
	}

	function startRealtimeUpdates() {
		// Update metrics every 5 seconds
		metricsInterval = setInterval(async () => {
			try {
				const [newMetrics, newUtilization] = await Promise.all([
					gpuServiceClient.getMetrics(),
					getGPUUtilization()
				]);
				gpuMetrics = newMetrics;
				utilization = newUtilization;
			} catch (err) {
				console.warn('Metrics update failed:', err);
			}
		}, 5000);

		// Update status every 10 seconds
		statusInterval = setInterval(async () => {
			try {
				const [newStatus, newWorkers] = await Promise.all([
					gpuServiceClient.getStatus(),
					gpuServiceClient.getWorkers()
				]);
				gpuStatus = newStatus;
				workers = newWorkers;
			} catch (err) {
				console.warn('Status update failed:', err);
			}
		}, 10000);
	}

	async function processDocument() {
		if (!documentText.trim()) return;
		
		try {
			isProcessing = true;
			
			const result = await processLegalDocument(documentText, {
				documentId: `demo-doc-${Date.now()}`,
				documentType: 'contract',
				practiceArea: 'employment-law',
				jurisdiction: 'US'
			});

			processingResults = [...processingResults, {
				timestamp: new Date().toLocaleTimeString(),
				type: 'Document Processing',
				input: documentText.substring(0, 100) + '...',
				result: result,
				success: true
			}];

		} catch (err: any) {
			processingResults = [...processingResults, {
				timestamp: new Date().toLocaleTimeString(),
				type: 'Document Processing',
				input: documentText.substring(0, 100) + '...',
				error: err.message,
				success: false
			}];
		} finally {
			isProcessing = false;
		}
	}

	async function processQuery() {
		if (!queryText.trim()) return;
		
		try {
			isProcessing = true;
			
			const result = await gpuServiceClient.submitTask({
				type: 'embedding',
				data: textToFloatArray(queryText),
				metadata: {
					query_type: 'legal_search',
					practice_area: 'general'
				},
				priority: 8,
				service_origin: 'demo-interface'
			});

			processingResults = [...processingResults, {
				timestamp: new Date().toLocaleTimeString(),
				type: 'Query Processing',
				input: queryText,
				result: result,
				success: result.status === 'success'
			}];

		} catch (err: any) {
			processingResults = [...processingResults, {
				timestamp: new Date().toLocaleTimeString(),
				type: 'Query Processing',
				input: queryText,
				error: err.message,
				success: false
			}];
		} finally {
			isProcessing = false;
		}
	}

	// Helper function for simple text to float array conversion
	function textToFloatArray(text: string): number[] {
		// Simplified version for demo - in production use proper tokenization
		const normalized = text.toLowerCase().replace(/[^\w\s]/g, '');
		const words = normalized.split(/\s+/).filter(w => w.length > 0);
		
		// Create 384-dimensional embedding
		const embedding = new Array(384).fill(0);
		
		for (let i = 0; i < words.length; i++) {
			const word = words[i];
			let hash = 0;
			for (let j = 0; j < word.length; j++) {
				hash = ((hash << 5) - hash) + word.charCodeAt(j);
				hash = hash & hash;
			}
			
			const index = Math.abs(hash) % embedding.length;
			embedding[index] += 1 / Math.sqrt(words.length);
		}
		
		// Normalize
		const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
		if (magnitude > 0) {
			for (let i = 0; i < embedding.length; i++) {
				embedding[i] /= magnitude;
			}
		}
		
		return embedding;
	}

	function clearResults() {
		processingResults = [];
	}

	function getStatusColor(status: string): string {
		switch (status) {
			case 'healthy': 
			case 'running': 
			case 'success': return 'text-green-600';
			case 'degraded': 
			case 'processing': return 'text-yellow-600';
			case 'unhealthy': 
			case 'error': 
			case 'down': return 'text-red-600';
			default: return 'text-gray-600';
		}
	}
</script>

<svelte:head>
	<title>GPU Integration Demo - Legal AI Platform</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 p-6">
	<div class="max-w-7xl mx-auto">
		<!-- Header -->
		<div class="bg-white shadow rounded-lg p-6 mb-6">
			<h1 class="text-3xl font-bold text-gray-900 mb-2">
				GPU Integration Demo
			</h1>
			<p class="text-gray-600">
				Legal AI Platform with CUDA GPU Acceleration - Real-time monitoring and processing
			</p>
			
			{#if error}
				<div class="mt-4 bg-red-50 border border-red-200 rounded-md p-4">
					<div class="flex">
						<div class="flex-shrink-0">
							<svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
								<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
							</svg>
						</div>
						<div class="ml-3">
							<h3 class="text-sm font-medium text-red-800">Error</h3>
							<p class="mt-1 text-sm text-red-700">{error}</p>
							<button 
								on:click={loadGPUData}
								class="mt-2 text-sm bg-red-100 text-red-800 px-3 py-1 rounded hover:bg-red-200"
							>
								Retry
							</button>
						</div>
					</div>
				</div>
			{/if}
		</div>

		{#if loading}
			<div class="bg-white shadow rounded-lg p-6 text-center">
				<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
				<p class="text-gray-600">Loading GPU integration data...</p>
			</div>
		{:else}
			<!-- GPU Status Dashboard -->
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
				<!-- GPU Status -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-2">GPU Status</h3>
					<div class="flex items-center">
						<div class="flex-shrink-0">
							<div class={`w-3 h-3 rounded-full ${gpuAvailable ? 'bg-green-500' : 'bg-red-500'}`}></div>
						</div>
						<div class="ml-3">
							<p class={`text-sm font-medium ${gpuAvailable ? 'text-green-800' : 'text-red-800'}`}>
								{gpuAvailable ? 'Available' : 'Unavailable'}
							</p>
							<p class="text-xs text-gray-500">CUDA GPU</p>
						</div>
					</div>
				</div>

				<!-- Active Workers -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-2">Active Workers</h3>
					<div class="text-2xl font-bold text-blue-600">
						{gpuStatus?.workers_active || 0}
					</div>
					<p class="text-xs text-gray-500">of {gpuStatus?.total_workers || 0} workers</p>
				</div>

				<!-- Queue Length -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-2">Queue Length</h3>
					<div class="text-2xl font-bold text-yellow-600">
						{gpuStatus?.queue_length || 0}
					</div>
					<p class="text-xs text-gray-500">pending tasks</p>
				</div>

				<!-- GPU Utilization -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-2">GPU Utilization</h3>
					<div class="text-2xl font-bold text-purple-600">
						{utilization.toFixed(1)}%
					</div>
					<p class="text-xs text-gray-500">current usage</p>
				</div>
			</div>

			<!-- Processing Interface -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
				<!-- Document Processing -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-4">Document Processing</h3>
					
					<div class="mb-4">
						<label for="document-text" class="block text-sm font-medium text-gray-700 mb-2">
							Legal Document Text
						</label>
						<textarea
							id="document-text"
							bind:value={documentText}
							rows="10"
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
							placeholder="Enter legal document text for processing..."
						></textarea>
					</div>

					<button
						on:click={processDocument}
						disabled={isProcessing || !documentText.trim()}
						class="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
					>
						{#if isProcessing}
							<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
							Processing...
						{:else}
							Process Document with GPU
						{/if}
					</button>
				</div>

				<!-- Query Processing -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-4">Query Processing</h3>
					
					<div class="mb-4">
						<label for="query-text" class="block text-sm font-medium text-gray-700 mb-2">
							Legal Query
						</label>
						<textarea
							id="query-text"
							bind:value={queryText}
							rows="4"
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
							placeholder="Enter your legal query..."
						></textarea>
					</div>

					<button
						on:click={processQuery}
						disabled={isProcessing || !queryText.trim()}
						class="w-full bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
					>
						{#if isProcessing}
							<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
							Processing...
						{:else}
							Process Query with GPU
						{/if}
					</button>
				</div>
			</div>

			<!-- Processing Results -->
			{#if processingResults.length > 0}
				<div class="bg-white shadow rounded-lg p-6 mb-6">
					<div class="flex justify-between items-center mb-4">
						<h3 class="text-lg font-semibold text-gray-900">Processing Results</h3>
						<button
							on:click={clearResults}
							class="text-sm text-gray-600 hover:text-gray-800"
						>
							Clear Results
						</button>
					</div>
					
					<div class="space-y-4 max-h-96 overflow-y-auto">
						{#each processingResults.slice().reverse() as result}
							<div class={`border-l-4 pl-4 py-2 ${result.success ? 'border-green-400 bg-green-50' : 'border-red-400 bg-red-50'}`}>
								<div class="flex justify-between items-start">
									<div class="flex-1">
										<p class="font-medium text-sm">
											{result.type} - {result.timestamp}
										</p>
										<p class="text-xs text-gray-600 mt-1">
											Input: {result.input}
										</p>
										{#if result.success && result.result}
											<div class="mt-2">
												<p class="text-xs text-green-800">
													Status: {result.result.status}
												</p>
												{#if result.result.result}
													<p class="text-xs text-gray-600">
														Result: {result.result.result.length} values generated
													</p>
												{/if}
												{#if result.result.process_time}
													<p class="text-xs text-gray-600">
														Process time: {result.result.process_time}ms
													</p>
												{/if}
											</div>
										{/if}
										{#if !result.success}
											<p class="text-xs text-red-800 mt-1">
												Error: {result.error}
											</p>
										{/if}
									</div>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- System Metrics -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Worker Status -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-4">Worker Status</h3>
					
					{#if workers.length > 0}
						<div class="space-y-3">
							{#each workers as worker}
								<div class="flex justify-between items-center p-3 bg-gray-50 rounded">
									<div>
										<p class="font-medium">Worker {worker.id}</p>
										<p class="text-sm text-gray-600">
											Tasks processed: {worker.tasks_processed}
										</p>
										{#if worker.current_task}
											<p class="text-xs text-blue-600">
												Current: {worker.current_task}
											</p>
										{/if}
									</div>
									<div class={`px-2 py-1 rounded text-xs ${worker.busy ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
										{worker.busy ? 'Busy' : 'Available'}
									</div>
								</div>
							{/each}
						</div>
					{:else}
						<p class="text-gray-600">No worker information available</p>
					{/if}
				</div>

				<!-- Service Registry -->
				<div class="bg-white shadow rounded-lg p-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-4">Service Registry</h3>
					
					{#if services?.services}
						<div class="space-y-2">
							{#each Object.entries(services.services) as [serviceName, serviceInfo]}
								<div class="flex justify-between items-center p-2 bg-gray-50 rounded">
									<div>
										<p class="font-medium text-sm">{serviceName}</p>
										<p class="text-xs text-gray-600">
											Port: {serviceInfo.port} | Type: {serviceInfo.type}
										</p>
									</div>
									<div class={`px-2 py-1 rounded text-xs ${getStatusColor(serviceInfo.status)}`}>
										{serviceInfo.status}
									</div>
								</div>
							{/each}
						</div>
					{:else}
						<p class="text-gray-600">No service information available</p>
					{/if}
				</div>
			</div>

			<!-- Performance Metrics -->
			{#if gpuMetrics}
				<div class="bg-white shadow rounded-lg p-6 mt-6">
					<h3 class="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
					
					<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
						<div>
							<p class="text-sm text-gray-600">Total Tasks</p>
							<p class="text-xl font-bold text-blue-600">{gpuMetrics.total_tasks}</p>
						</div>
						<div>
							<p class="text-sm text-gray-600">Completed</p>
							<p class="text-xl font-bold text-green-600">{gpuMetrics.completed_tasks}</p>
						</div>
						<div>
							<p class="text-sm text-gray-600">Failed</p>
							<p class="text-xl font-bold text-red-600">{gpuMetrics.failed_tasks}</p>
						</div>
						<div>
							<p class="text-sm text-gray-600">Avg Process Time</p>
							<p class="text-xl font-bold text-purple-600">{gpuMetrics.average_process_time}ms</p>
						</div>
					</div>
				</div>
			{/if}
		{/if}
	</div>
</div>

<style>
	/* Custom scrollbar for processing results */
	:global(.space-y-4::-webkit-scrollbar) {
		width: 6px;
	}

	:global(.space-y-4::-webkit-scrollbar-track) {
		background: #f1f5f9;
		border-radius: 3px;
	}

	:global(.space-y-4::-webkit-scrollbar-thumb) {
		background: #cbd5e1;
		border-radius: 3px;
	}

	:global(.space-y-4::-webkit-scrollbar-thumb:hover) {
		background: #94a3b8;
	}
</style>