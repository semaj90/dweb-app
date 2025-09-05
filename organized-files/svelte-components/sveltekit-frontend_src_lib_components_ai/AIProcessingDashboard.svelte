<!-- AI Processing Dashboard - Integration Demo -->
<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Badge } from 'bits-ui';
	import { Card, CardContent, CardHeader, CardTitle } from 'bits-ui';
	import { Button } from 'bits-ui';
	import { Progress } from 'bits-ui';
	import LLMProviderSelector from './LLMProviderSelector.svelte';
	import { aiServiceWorkerManager, type AITask, type AITaskResult } from '$lib/services/aiServiceWorkerManager';
	import type { LLMProvider } from '$lib/types/llm';
	import { fade, fly } from 'svelte/transition';

	// Reactive stores from the service worker manager
	let taskQueue = $derived(aiServiceWorkerManager.taskQueue$)
	let workerStatus = $derived(aiServiceWorkerManager.workerStatus$)
	let systemMetrics = $derived(aiServiceWorkerManager.systemMetrics$)

	// Component state
	let selectedProvider: LLMProvider | null = null;
	let isProcessing = false;
	let processingResults: AITaskResult[] = [];
	let testInput = "Analyze this legal document for key compliance issues and regulatory requirements.";

	// Demo task examples
	const demoTasks = [
		{
			name: "Document Embedding",
			type: "embedding" as const,
			description: "Generate vector embeddings for document search",
			payload: { text: testInput, model: "nomic-embed-text" }
		},
		{
			name: "Legal Analysis",
			type: "analysis" as const,
			description: "Analyze document for legal compliance",
			payload: { content: testInput, analysisType: "legal-document" }
		},
		{
			name: "Text Generation",
			type: "generation" as const,
			description: "Generate legal summary and recommendations",
			payload: { prompt: `Create a legal summary for: ${testInput}`, model: "gemma3-legal" }
		},
		{
			name: "Vector Search",
			type: "vector-search" as const,
			description: "Search similar documents in database",
			payload: { query: testInput, collection: "legal_docs", limit: 5 }
		}
	];

	// Event handlers
	const handleProviderSelected = (event: CustomEvent<{ provider: LLMProvider }>) => {
		selectedProvider = event.detail.provider;
	};

	const handleStatusChanged = (event: CustomEvent<{ provider: LLMProvider status: string }>) => {
		console.log(`Provider ${event.detail.provider.name} status changed to ${event.detail.status}`);
	};

	// Process single task
	const processTask = async (taskTemplate: typeof demoTasks[0]) => {
		if (!selectedProvider) {
			alert('Please select an LLM provider first');
			return;
		}

		if (selectedProvider.status !== 'online') {
			alert(`Provider ${selectedProvider.name} is ${selectedProvider.status}. Please select an online provider.`);
			return;
		}

		try {
			isProcessing = true;
			
			const taskId = await aiServiceWorkerManager.queueTask({
				type: taskTemplate.type,
				priority: 'medium',
				provider: selectedProvider,
				payload: taskTemplate.payload,
				metadata: {
					userId: 'demo-user',
					sessionId: 'demo-session',
					timestamp: Date.now()
				}
			});

			console.log(`✅ Task ${taskId} queued successfully`);
			
			// For demo purposes, simulate task completion
			// In a real app, you would listen to worker events
			setTimeout(() => {
				const mockResult: AITaskResult = {
					taskId,
					success: true,
					result: generateMockResult(taskTemplate.type),
					duration: Math.random() * 2000 + 500,
					metrics: {
						tokensProcessed: Math.floor(Math.random() * 1000) + 100,
						throughput: Math.floor(Math.random() * 50) + 10,
						memoryUsed: `${Math.floor(Math.random() * 500) + 100}MB`
					}
				};
				
				processingResults = [mockResult, ...processingResults.slice(0, 9)]; // Keep last 10 results
				isProcessing = false;
			}, Math.random() * 3000 + 1000);
			
		} catch (error) {
			console.error('Task processing failed:', error);
			isProcessing = false;
		}
	};

	// Process multiple tasks in parallel
	const processParallelTasks = async () => {
		if (!selectedProvider || selectedProvider.status !== 'online') {
			alert('Please select an online LLM provider');
			return;
		}

		try {
			isProcessing = true;
			
			const tasks = demoTasks.map(task => ({
				type: task.type,
				priority: 'high' as const,
				provider: selectedProvider!,
				payload: task.payload
			}));

			console.log('🚀 Processing parallel tasks...');
			
			// This would use the actual parallel processing in a real implementation
			const results = await aiServiceWorkerManager.processParallel(tasks);
			
			processingResults = [...results.reverse(), ...processingResults.slice(0, 6)];
			isProcessing = false;
			
		} catch (error) {
			console.error('Parallel processing failed:', error);
			isProcessing = false;
		}
	};

	// Generate mock results for demo
	const generateMockResult = (taskType: string) => {
		switch (taskType) {
			case 'embedding':
				return {
					embedding: Array(384).fill(0).map(() => Math.random() - 0.5),
					dimensions: 384
				};
			case 'analysis':
				return {
					entities: ['GDPR', 'Privacy Policy', 'Data Controller'],
					sentiment: 'neutral',
					compliance_score: 0.85,
					key_points: ['Data retention requirements', 'User consent mechanisms', 'Privacy by design']
				};
			case 'generation':
				return {
					text: 'This document appears to address key privacy regulations including GDPR compliance, data retention policies, and user consent mechanisms. Recommendations include updating privacy notices and implementing data subject request procedures.',
					confidence: 0.92
				};
			case 'vector-search':
				return {
					results: [
						{ id: '1', title: 'Privacy Policy Template', similarity: 0.94 },
						{ id: '2', title: 'GDPR Compliance Guide', similarity: 0.87 },
						{ id: '3', title: 'Data Retention Standards', similarity: 0.81 }
					]
				};
			default:
				return { status: 'completed' };
		}
	};

	// System health monitoring
	let healthCheckInterval: number
	
	onMount(() => {
		// Start health monitoring
		healthCheckInterval = setInterval(() => {
			// Update system metrics (in a real app, this would come from the worker manager)
			systemMetrics.update(metrics => ({
				...metrics,
				totalTasksProcessed: metrics.totalTasksProcessed + Math.floor(Math.random() * 3),
				currentLoad: Math.random() * 100,
				availableWorkers: 4 - Math.floor(Math.random() * 2)
			}));
		}, 2000);
	});

	onDestroy(() => {
		if (healthCheckInterval) {
			clearInterval(healthCheckInterval);
		}
	});

	// Utility functions
	const getTaskTypeColor = (type: string) => {
		switch (type) {
			case 'embedding': return 'bg-blue-500';
			case 'generation': return 'bg-green-500';
			case 'analysis': return 'bg-purple-500';
			case 'vector-search': return 'bg-orange-500';
			default: return 'bg-gray-500';
		}
	};

	const formatDuration = (ms: number) => {
		return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
	};
</script>

<div class="ai-processing-dashboard p-6 space-y-6 bg-yorha-bg-primary min-h-screen">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-bold text-yorha-text-primary">AI Processing Dashboard</h1>
			<p class="text-yorha-text-secondary">Multi-LLM orchestration and task management</p>
		</div>
		
		<!-- System Status -->
		<div class="flex items-center space-x-4">
			<Badge class={selectedProvider?.status === 'online' ? 'bg-yorha-success' : 'bg-yorha-danger'}>
				{selectedProvider?.status?.toUpperCase() || 'NO PROVIDER'}
			</Badge>
			<div class="text-sm text-yorha-text-secondary">
				Queue: {$taskQueue?.length || 0} | Workers: {$systemMetrics?.availableWorkers || 0}
			</div>
		</div>
	</div>

	<!-- Provider Selection -->
	<Card>
		<CardHeader>
			<CardTitle>LLM Provider Configuration</CardTitle>
		</CardHeader>
		<CardContent>
			<LLMProviderSelector 
				bind:selectedProvider
				on:providerSelected={handleProviderSelected}
				on:statusChanged={handleStatusChanged}
			/>
		</CardContent>
	</Card>

	<!-- System Metrics -->
	<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
		<Card>
			<CardContent class="p-4">
				<div class="text-2xl font-bold text-yorha-primary">{$systemMetrics?.totalTasksProcessed || 0}</div>
				<div class="text-sm text-yorha-text-secondary">Tasks Processed</div>
			</CardContent>
		</Card>
		
		<Card>
			<CardContent class="p-4">
				<div class="text-2xl font-bold text-yorha-accent">{$systemMetrics?.averageResponseTime?.toFixed(0) || 0}ms</div>
				<div class="text-sm text-yorha-text-secondary">Avg Response Time</div>
			</CardContent>
		</Card>
		
		<Card>
			<CardContent class="p-4">
				<div class="flex items-center space-x-2">
					<div class="text-2xl font-bold text-yorha-warning">{$systemMetrics?.currentLoad?.toFixed(1) || 0}%</div>
					<Progress value={$systemMetrics?.currentLoad || 0} class="flex-1 h-2" />
				</div>
				<div class="text-sm text-yorha-text-secondary">System Load</div>
			</CardContent>
		</Card>
		
		<Card>
			<CardContent class="p-4">
				<div class="text-2xl font-bold text-yorha-success">{$systemMetrics?.availableWorkers || 0}</div>
				<div class="text-sm text-yorha-text-secondary">Available Workers</div>
			</CardContent>
		</Card>
	</div>

	<!-- Task Controls -->
	<Card>
		<CardHeader>
			<CardTitle>AI Task Processing</CardTitle>
		</CardHeader>
		<CardContent class="space-y-4">
			<!-- Test Input -->
			<div>
				<label class="block text-sm font-medium text-yorha-text-primary mb-2">
					Test Input
				</label>
				<textarea 
					bind:value={testInput}
					class="w-full h-20 px-3 py-2 bg-yorha-bg-secondary border border-yorha-border rounded-md text-yorha-text-primary placeholder-yorha-text-tertiary focus:outline-none focus:ring-2 focus:ring-yorha-primary"
					placeholder="Enter text to process..."
				/>
			</div>

			<!-- Individual Task Buttons -->
			<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
				{#each demoTasks as task}
					<Button
						variant="outline"
						disabled={!selectedProvider || selectedProvider.status !== 'online' || isProcessing}
						onclick={() => processTask(task)}
						class="h-auto p-3 flex flex-col items-start space-y-1"
					>
						<div class="flex items-center space-x-2">
							<div class={`w-3 h-3 rounded-full ${getTaskTypeColor(task.type)}`}></div>
							<span class="font-medium">{task.name}</span>
						</div>
						<span class="text-xs text-yorha-text-secondary text-left">{task.description}</span>
					</Button>
				{/each}
			</div>

			<!-- Parallel Processing -->
			<div class="flex items-center justify-center pt-4 border-t border-yorha-border">
				<Button
					disabled={!selectedProvider || selectedProvider.status !== 'online' || isProcessing}
					onclick={processParallelTasks}
					class="bg-yorha-primary hover:bg-yorha-primary/80"
				>
					{#if isProcessing}
						<div class="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
						Processing...
					{:else}
						🚀 Run All Tasks in Parallel
					{/if}
				</Button>
			</div>
		</CardContent>
	</Card>

	<!-- Results -->
	{#if processingResults.length > 0}
		<Card>
			<CardHeader>
				<CardTitle>Processing Results</CardTitle>
			</CardHeader>
			<CardContent>
				<div class="space-y-3 max-h-96 overflow-y-auto">
					{#each processingResults as result (result.taskId)}
						<div 
							class="p-3 bg-yorha-bg-secondary rounded-md border border-yorha-border"
							transitionfly={{ y: -20, duration: 300 }}
						>
							<div class="flex items-center justify-between mb-2">
								<div class="flex items-center space-x-2">
									<Badge class={result.success ? 'bg-yorha-success' : 'bg-yorha-danger'}>
										{result.success ? 'SUCCESS' : 'ERROR'}
									</Badge>
									<span class="text-sm text-yorha-text-secondary">
										Task ID: {result.taskId.slice(-8)}
									</span>
								</div>
								<div class="text-xs text-yorha-text-tertiary">
									{formatDuration(result.duration)}
								</div>
							</div>
							
							{#if result.success && result.result}
								<div class="text-sm bg-yorha-bg-primary p-2 rounded border">
									<pre class="whitespace-pre-wrap text-yorha-text-primary overflow-x-auto">
{JSON.stringify(result.result, null, 2)}
									</pre>
								</div>
							{/if}
							
							{#if result.metrics}
								<div class="flex items-center space-x-4 mt-2 text-xs text-yorha-text-secondary">
									<span>Tokens: {result.metrics.tokensProcessed}</span>
									<span>Throughput: {result.metrics.throughput} t/s</span>
									<span>Memory: {result.metrics.memoryUsed}</span>
								</div>
							{/if}
						</div>
					{/each}
				</div>
			</CardContent>
		</Card>
	{/if}

	<!-- Worker Status -->
	{#if $workerStatus && $workerStatus.length > 0}
		<Card>
			<CardHeader>
				<CardTitle>Worker Status</CardTitle>
			</CardHeader>
			<CardContent>
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
					{#each $workerStatus as worker (worker.id)}
						<div class="p-3 bg-yorha-bg-secondary rounded border">
							<div class="flex items-center justify-between mb-2">
								<span class="font-medium text-yorha-text-primary">{worker.id}</span>
								<Badge class={
									worker.status === 'idle' ? 'bg-yorha-success' :
									worker.status === 'busy' ? 'bg-yorha-warning' :
									worker.status === 'error' ? 'bg-yorha-danger' : 'bg-yorha-text-secondary'
								}>
									{worker.status.toUpperCase()}
								</Badge>
							</div>
							<div class="text-xs text-yorha-text-secondary space-y-1">
								<div>Type: {worker.type}</div>
								<div>Completed: {worker.tasksCompleted}</div>
								<div>Avg Time: {formatDuration(worker.averageTaskTime)}</div>
								<div>Load: {worker.load.toFixed(1)}%</div>
							</div>
						</div>
					{/each}
				</div>
			</CardContent>
		</Card>
	{/if}
</div>

<style>
	.ai-processing-dashboard {
		background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
	}
</style>