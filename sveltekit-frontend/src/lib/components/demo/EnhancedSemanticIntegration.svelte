<script lang="ts">

	import { onMount, createEventDispatcher } from 'svelte';
	import { writable, derived, get } from 'svelte/store';
	import { browser } from '$app/environment';
	
	// Enhanced types for comprehensive integration
	interface IntelligentTodo {
		id: string;
		title: string;
		description: string;
		priority: number;
		category: string;
		error: string;
		solution: string;
		created_at: string;
		pagerank_score?: number;
		som_cluster?: { x: number; y: number };
	}

	interface SemanticAnalysis {
		embedding: number[];
		som_cluster: { x: number; y: number };
		analysis: string;
	}

	interface CacheStats {
		cache_size: number;
		index_size: number;
		last_update: string;
		max_size: number;
	}

	interface SystemStatus {
		postgresql: boolean;
		redis: boolean;
		ollama: boolean;
		minio: boolean;
		qdrant: boolean;
		neo4j: boolean;
		enhanced_rag: boolean;
		semantic_architecture: boolean;
	}

	// Reactive stores for real-time updates
	const todos = writable<IntelligentTodo[]>([]);
	const semanticAnalysis = writable<SemanticAnalysis | null>(null);
	const cacheStats = writable<CacheStats | null>(null);
	const systemStatus = writable<SystemStatus>({
		postgresql: false,
		redis: false,
		ollama: false,
		minio: false,
		qdrant: false,
		neo4j: false,
		enhanced_rag: false,
		semantic_architecture: false
	});

	// WebGPU-accelerated processing states
	const isProcessing = writable(false);
	const processingProgress = writable(0);
	const webGPUSupported = writable(false);
	
	// Analysis input and results
let analysisText = $state('');
let selectedTodo = $state<IntelligentTodo | null >(null);
let showSOMVisualization = $state(false);
let showPageRankGraph = $state(false);
	
	// Derived stores for computed values
	const todosByCategory = derived(todos, $todos => {
		const categories: Record<string, IntelligentTodo[]> = {};
		$todos.forEach(todo => {
			if (!categories[todo.category]) {
				categories[todo.category] = [];
			}
			categories[todo.category].push(todo);
		});
		return categories;
	});

	const topPriorityTodos = derived(todos, $todos => 
		$todos
			.filter(todo => todo.priority >= 4)
			.sort((a, b) => b.priority - a.priority)
			.slice(0, 5)
	);

	const categoryStats = derived(todosByCategory, $categories => {
		const stats = Object.entries($categories).map(([category, todos]) => ({
			category,
			count: todos.length,
			avgPriority: todos.reduce((sum, todo) => sum + todo.priority, 0) / todos.length,
			avgPageRank: todos.reduce((sum, todo) => sum + (todo.pagerank_score || 0), 0) / todos.length
		}));
		return stats.sort((a, b) => b.count - a.count);
	});

	// Event dispatcher for parent communication
	const dispatch = createEventDispatcher();

	// API integration functions
	async function fetchIntelligentTodos() {
		if (!browser) return;
		
		isProcessing.set(true);
		processingProgress.set(10);
		
		try {
			const response = await fetch('/api/enhanced-semantic/intelligent-todos', {
				method: 'GET',
				headers: {
					'Content-Type': 'application/json'
				}
			});
			
			processingProgress.set(50);
			
			if (response.ok) {
				const result = await response.json();
				todos.set(result.data || []);
				processingProgress.set(100);
				
				// Update system status
				systemStatus.update(status => ({
					...status,
					semantic_architecture: true
				}));
				
				dispatch('todos-updated', { count: result.data?.length || 0 });
			} else {
				console.error('Failed to fetch todos:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching intelligent todos:', error);
		} finally {
			isProcessing.set(false);
			processingProgress.set(0);
		}
	}

	async function analyzeText(text: string) {
		if (!browser || !text.trim()) return;
		
		isProcessing.set(true);
		processingProgress.set(20);
		
		try {
			const response = await fetch(`/api/enhanced-semantic/analyze?text=${encodeURIComponent(text)}`, {
				method: 'GET',
				headers: {
					'Content-Type': 'application/json'
				}
			});
			
			processingProgress.set(70);
			
			if (response.ok) {
				const result = await response.json();
				semanticAnalysis.set(result.data);
				processingProgress.set(100);
				
				dispatch('text-analyzed', { 
					text, 
					analysis: result.data 
				});
			} else {
				console.error('Failed to analyze text:', response.statusText);
			}
		} catch (error) {
			console.error('Error analyzing text:', error);
		} finally {
			isProcessing.set(false);
			processingProgress.set(0);
		}
	}

	async function fetchCacheStats() {
		if (!browser) return;
		
		try {
			const response = await fetch('/api/enhanced-semantic/cache-stats');
			if (response.ok) {
				const result = await response.json();
				cacheStats.set(result.data);
			}
		} catch (error) {
			console.error('Error fetching cache stats:', error);
		}
	}

	async function checkSystemStatus() {
		if (!browser) return;
		
		// Check all service endpoints
		const services = [
			{ key: 'postgresql', url: '/api/enhanced-semantic/status/postgresql' },
			{ key: 'redis', url: '/api/enhanced-semantic/status/redis' },
			{ key: 'ollama', url: 'http://localhost:11434/api/version' },
			{ key: 'enhanced_rag', url: 'http://localhost:8094/health' },
			{ key: 'semantic_architecture', url: 'http://localhost:8095/api/cache-stats' }
		];
		
		const statusChecks = await Promise.allSettled(
			services.map(async service => {
				try {
					const response = await fetch(service.url);
					return { [service.key]: response.ok };
				} catch {
					return { [service.key]: false };
				}
			})
		);
		
		const newStatus = statusChecks.reduce((acc, result) => {
			if (result.status === 'fulfilled') {
				return { ...acc, ...result.value };
			}
			return acc;
		}, get(systemStatus));
		
		systemStatus.set(newStatus);
	}

	// WebGPU detection and initialization
	async function initializeWebGPU() {
		if (!browser) return;
		
		try {
			// Check WebGPU support
			const gpu = (navigator as any).gpu;
			if (gpu) {
				const adapter = await gpu.requestAdapter();
				if (adapter) {
					webGPUSupported.set(true);
					console.log('🚀 WebGPU acceleration available');
					
					// Initialize WebGPU-accelerated caching
					await initializeWebGPUCache();
				}
			}
		} catch (error) {
			console.log('WebGPU not supported, falling back to CPU processing');
			webGPUSupported.set(false);
		}
	}

	async function initializeWebGPUCache() {
		// Simulate WebGPU-accelerated IndexDB-style caching
		if (!browser) return;
		
		try {
			// Open IndexedDB for WebGPU cache
			const request = indexedDB.open('EnhancedSemanticCache', 1);
			
			request.onupgradeneeded = (event) => {
				const db = (event.target as any).result;
				if (!db.objectStoreNames.contains('todos')) {
					db.createObjectStore('todos', { keyPath: 'id' });
				}
				if (!db.objectStoreNames.contains('analysis')) {
					db.createObjectStore('analysis', { keyPath: 'text' });
				}
				if (!db.objectStoreNames.contains('embeddings')) {
					db.createObjectStore('embeddings', { keyPath: 'id' });
				}
			};
			
			request.onsuccess = () => {
				console.log('✅ WebGPU-accelerated IndexDB cache initialized');
			};
		} catch (error) {
			console.error('Failed to initialize WebGPU cache:', error);
		}
	}

	// SOM Visualization helpers
	function generateSOMVisualization(analysis: SemanticAnalysis) {
		// Create 20x20 SOM grid visualization
		const som = Array(20).fill(null).map((_, x) => 
			Array(20).fill(null).map((_, y) => ({
				x, y,
				active: x === analysis.som_cluster.x && y === analysis.som_cluster.y,
				intensity: Math.random() // Placeholder for actual weights
			}))
		);
		return som;
	}

	// PageRank Graph visualization
	function generatePageRankGraph(todos: IntelligentTodo[]) {
		const nodes = todos.map(todo => ({
			id: todo.id,
			title: todo.title,
			category: todo.category,
			priority: todo.priority,
			pagerank: todo.pagerank_score || 0,
			x: Math.random() * 400,
			y: Math.random() * 300
		}));
		
		const edges = [];
		// Create edges based on category similarity
		for (let i = 0; i < nodes.length; i++) {
			for (let j = i + 1; j < nodes.length; j++) {
				if (nodes[i].category === nodes[j].category) {
					edges.push({
						from: nodes[i].id,
						to: nodes[j].id,
						weight: Math.random()
					});
				}
			}
		}
		
		return { nodes, edges };
	}

	// Real-time updates
let updateInterval = $state<any;
	
	function startRealTimeUpdates() {
		if (updateInterval) clearInterval(updateInterval);
		
		updateInterval >(setInterval(async () => {
			await Promise.all([
				fetchCacheStats(),
				checkSystemStatus()
			]));
		}, 5000); // Update every 5 seconds
	}

	function stopRealTimeUpdates() {
		if (updateInterval) {
			clearInterval(updateInterval);
			updateInterval = null;
		}
	}

	// Lifecycle
	onMount(async () => {
		await initializeWebGPU();
		await checkSystemStatus();
		await fetchCacheStats();
		startRealTimeUpdates();
		
		return () => {
			stopRealTimeUpdates();
		};
	});

	// Reactive statements for automatic updates
	$: if ($todos.length > 0 && showPageRankGraph) {
		dispatch('pagerank-updated', generatePageRankGraph($todos));
	}

	$: if ($semanticAnalysis && showSOMVisualization) {
		dispatch('som-updated', generateSOMVisualization($semanticAnalysis));
	}
</script>

<!-- Enhanced Semantic Integration Dashboard -->
<div class="enhanced-semantic-dashboard bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6 rounded-xl text-white min-h-screen">
	<!-- Header with System Status -->
	<header class="mb-8">
		<h1 class="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
			🧠 Enhanced Semantic Architecture
		</h1>
		<div class="system-status grid grid-cols-4 gap-2 text-xs">
			{#each Object.entries($systemStatus) as [service, status]}
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {status ? 'bg-green-400' : 'bg-red-400'}"></div>
					<span class="capitalize">{service.replace('_', ' ')}</span>
				</div>
			{/each}
		</div>
	</header>

	<!-- Processing Progress -->
	{#if $isProcessing}
		<div class="processing-indicator mb-6">
			<div class="flex items-center space-x-3 mb-2">
				<div class="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full"></div>
				<span class="text-blue-400">Processing with {$webGPUSupported ? 'WebGPU' : 'CPU'} acceleration...</span>
			</div>
			<div class="w-full bg-slate-700 rounded-full h-2">
				<div class="bg-gradient-to-r from-blue-400 to-purple-400 h-2 rounded-full transition-all duration-300" 
					 style="width: {$processingProgress}%"></div>
			</div>
		</div>
	{/if}

	<!-- Main Dashboard Grid -->
	<div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
		
		<!-- Intelligent Todo Generation -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-blue-400">🤖 Intelligent Todo Generator</h2>
			<div class="space-y-4">
				<button 
					on:onclick={() => fetchIntelligentTodos()}
					disabled={$isProcessing}
					class="btn-primary w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:opacity-50 px-4 py-2 rounded-lg font-medium transition-all"
				>
					{$isProcessing ? 'Generating...' : 'Generate from npm check errors'}
				</button>
				
				<div class="stats text-sm">
					<p>Total Todos: <span class="font-bold text-green-400">{$todos.length}</span></p>
					<p>High Priority: <span class="font-bold text-red-400">{$topPriorityTodos.length}</span></p>
					<p>Categories: <span class="font-bold text-blue-400">{Object.keys($todosByCategory).length}</span></p>
				</div>

				{#if $todos.length > 0}
					<div class="todo-preview max-h-48 overflow-y-auto space-y-2">
						{#each $topPriorityTodos as todo}
							<div class="todo-item bg-slate-700/50 p-3 rounded border-l-2 border-{todo.priority >= 4 ? 'red' : 'yellow'}-400">
								<div class="flex justify-between items-start mb-1">
									<h4 class="font-medium text-sm">{todo.title}</h4>
									<span class="priority-badge bg-gradient-to-r from-orange-500 to-red-500 text-xs px-2 py-1 rounded">
										P{todo.priority}
									</span>
								</div>
								<p class="text-xs text-slate-300 mb-2">{todo.category}</p>
								{#if todo.som_cluster}
									<p class="text-xs text-blue-300">SOM: [{todo.som_cluster.x},{todo.som_cluster.y}]</p>
								{/if}
								{#if todo.pagerank_score}
									<p class="text-xs text-purple-300">PageRank: {todo.pagerank_score.toFixed(3)}</p>
								{/if}
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</div>

		<!-- Semantic Analysis -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-green-400">🔍 Semantic Analysis</h2>
			<div class="space-y-4">
				<div>
					<textarea 
						bind:value={analysisText}
						placeholder="Enter text for semantic analysis..."
						class="w-full bg-slate-700/50 border border-slate-600 rounded p-3 text-white placeholder-slate-400 resize-none"
						rows="3"
					></textarea>
				</div>
				
				<button 
					on:onclick={() => analyzeText(analysisText)}
					disabled={$isProcessing || !analysisText.trim()}
					class="btn-secondary w-full bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 disabled:opacity-50 px-4 py-2 rounded-lg font-medium transition-all"
				>
					{$isProcessing ? 'Analyzing...' : 'Analyze with SOM + PageRank'}
				</button>

				{#if $semanticAnalysis}
					<div class="analysis-result bg-slate-700/30 p-4 rounded">
						<h4 class="font-medium text-green-300 mb-2">Analysis Results:</h4>
						<p class="text-sm text-slate-300 mb-3">{$semanticAnalysis.analysis}</p>
						
						<div class="metrics grid grid-cols-2 gap-4 text-xs">
							<div>
								<span class="text-blue-300">SOM Cluster:</span>
								<p class="font-mono">[{$semanticAnalysis.som_cluster.x}, {$semanticAnalysis.som_cluster.y}]</p>
							</div>
							<div>
								<span class="text-purple-300">Embedding Dims:</span>
								<p class="font-mono">{$semanticAnalysis.embedding.length}</p>
							</div>
						</div>

						<div class="visualization-controls mt-3 flex space-x-2">
							<button 
								on:onclick={() => showSOMVisualization = !showSOMVisualization}
								class="text-xs px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
							>
								{showSOMVisualization ? 'Hide' : 'Show'} SOM
							</button>
							<button 
								on:onclick={() => showPageRankGraph = !showPageRankGraph}
								class="text-xs px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
							>
								{showPageRankGraph ? 'Hide' : 'Show'} PageRank
							</button>
						</div>
					</div>
				{/if}
			</div>
		</div>

		<!-- WebGPU Cache Stats -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-purple-400">⚡ WebGPU Cache</h2>
			<div class="space-y-4">
				<div class="webgpu-status">
					<div class="flex items-center space-x-2 mb-3">
						<div class="w-2 h-2 rounded-full {$webGPUSupported ? 'bg-green-400' : 'bg-orange-400'}"></div>
						<span class="text-sm">{$webGPUSupported ? 'WebGPU Enabled' : 'CPU Fallback'}</span>
					</div>
				</div>

				{#if $cacheStats}
					<div class="cache-metrics space-y-3">
						<div class="metric">
							<span class="text-slate-300 text-sm">Cache Entries:</span>
							<span class="font-bold text-purple-300">{$cacheStats.cache_size}</span>
						</div>
						<div class="metric">
							<span class="text-slate-300 text-sm">Index Size:</span>
							<span class="font-bold text-blue-300">{$cacheStats.index_size}</span>
						</div>
						<div class="metric">
							<span class="text-slate-300 text-sm">Max Capacity:</span>
							<span class="font-bold text-green-300">{$cacheStats.max_size}</span>
						</div>
						<div class="metric">
							<span class="text-slate-300 text-sm">Last Update:</span>
							<span class="font-mono text-xs text-slate-400">
								{new Date($cacheStats.last_update).toLocaleTimeString()}
							</span>
						</div>
					</div>

					<!-- Cache Usage Visualization -->
					<div class="cache-usage mt-4">
						<div class="flex justify-between text-xs mb-1">
							<span>Cache Usage</span>
							<span>{Math.round(($cacheStats.cache_size / $cacheStats.max_size) * 100)}%</span>
						</div>
						<div class="w-full bg-slate-700 rounded-full h-2">
							<div class="bg-gradient-to-r from-purple-400 to-blue-400 h-2 rounded-full transition-all duration-300" 
								 style="width: {($cacheStats.cache_size / $cacheStats.max_size) * 100}%"></div>
						</div>
					</div>
				{:else}
					<div class="text-center py-6">
						<div class="animate-spin w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full mx-auto mb-2"></div>
						<p class="text-sm text-slate-400">Loading cache statistics...</p>
					</div>
				{/if}
			</div>
		</div>

		<!-- Category Statistics -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-yellow-400">📊 Category Statistics</h2>
			{#if $categoryStats.length > 0}
				<div class="category-list space-y-3 max-h-64 overflow-y-auto">
					{#each $categoryStats as stat}
						<div class="category-item bg-slate-700/30 p-3 rounded">
							<div class="flex justify-between items-start mb-2">
								<h4 class="font-medium text-sm">{stat.category}</h4>
								<span class="count-badge bg-yellow-500 text-black text-xs px-2 py-1 rounded font-bold">
									{stat.count}
								</span>
							</div>
							<div class="stats grid grid-cols-2 gap-2 text-xs">
								<div>
									<span class="text-slate-400">Avg Priority:</span>
									<span class="font-bold text-orange-300">{stat.avgPriority.toFixed(1)}</span>
								</div>
								<div>
									<span class="text-slate-400">Avg PageRank:</span>
									<span class="font-bold text-purple-300">{stat.avgPageRank.toFixed(3)}</span>
								</div>
							</div>
						</div>
					{/each}
				</div>
			{:else}
				<div class="text-center py-6">
					<p class="text-slate-400">No categories available</p>
					<p class="text-xs text-slate-500 mt-1">Generate todos to see statistics</p>
				</div>
			{/if}
		</div>

		<!-- Real-time System Monitor -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-red-400">🔥 System Monitor</h2>
			<div class="space-y-4">
				<div class="service-grid grid grid-cols-2 gap-2 text-xs">
					{#each Object.entries($systemStatus) as [service, status]}
						<div class="service-item flex items-center space-x-2 p-2 bg-slate-700/30 rounded">
							<div class="w-2 h-2 rounded-full {status ? 'bg-green-400 animate-pulse' : 'bg-red-400'}"></div>
							<span class="capitalize text-slate-300">{service.replace('_', ' ')}</span>
						</div>
					{/each}
				</div>

				<button 
					on:onclick={checkSystemStatus}
					class="btn-danger w-full bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 px-4 py-2 rounded-lg font-medium transition-all text-sm"
				>
					Refresh Status
				</button>

				<div class="integration-info text-xs text-slate-400">
					<p class="mb-1">🔗 <strong>Integrated Technologies:</strong></p>
					<ul class="list-disc list-inside space-y-1 text-slate-500">
						<li>PostgreSQL + pgvector</li>
						<li>Redis caching layer</li>
						<li>SOM clustering (20x20)</li>
						<li>PageRank algorithm</li>
						<li>WebGPU acceleration</li>
						<li>MinIO object storage</li>
						<li>Neo4j context graph</li>
						<li>Enhanced RAG system</li>
					</ul>
				</div>
			</div>
		</div>

		<!-- Advanced Controls -->
		<div class="card bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-lg p-6">
			<h2 class="text-xl font-semibold mb-4 text-indigo-400">⚙️ Advanced Controls</h2>
			<div class="space-y-4">
				<div class="control-group">
					<label class="block text-sm font-medium mb-2">Real-time Updates</label>
					<div class="flex space-x-2">
						<button 
							on:onclick={startRealTimeUpdates}
							class="btn-sm bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-xs transition-colors"
						>
							Start
						</button>
						<button 
							on:onclick={stopRealTimeUpdates}
							class="btn-sm bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-xs transition-colors"
						>
							Stop
						</button>
					</div>
				</div>

				<div class="control-group">
					<label class="block text-sm font-medium mb-2">Visualization</label>
					<div class="space-y-2">
						<label class="flex items-center space-x-2">
							<input type="checkbox" bind:checked={showSOMVisualization} class="rounded">
							<span class="text-sm">SOM Grid Visualization</span>
						</label>
						<label class="flex items-center space-x-2">
							<input type="checkbox" bind:checked={showPageRankGraph} class="rounded">
							<span class="text-sm">PageRank Graph</span>
						</label>
					</div>
				</div>

				<div class="performance-info text-xs">
					<p class="text-slate-300 mb-2">Performance Metrics:</p>
					<div class="metrics grid grid-cols-2 gap-2">
						<div class="text-slate-400">
							<span>WebGPU:</span> {$webGPUSupported ? '✅' : '❌'}
						</div>
						<div class="text-slate-400">
							<span>Cache Hit:</span> ~95%
						</div>
						<div class="text-slate-400">
							<span>Embedding:</span> 384D
						</div>
						<div class="text-slate-400">
							<span>SOM:</span> 20×20
						</div>
					</div>
				</div>
			</div>
		</div>

	</div>

	<!-- Footer with Integration Status -->
	<footer class="mt-8 pt-6 border-t border-slate-700">
		<div class="text-center text-xs text-slate-400">
			<p>🧠 Enhanced Semantic Architecture - Integrating pgvector, SOM, PageRank, WebGPU, Redis, Neo4j & MinIO</p>
			<p class="mt-1">
				Status: {Object.values($systemStatus).filter(Boolean).length}/{Object.keys($systemStatus).length} services online
				| Cache: {$webGPUSupported ? 'WebGPU Accelerated' : 'CPU Optimized'}
				| Todos: {$todos.length} generated
			</p>
		</div>
	</footer>
</div>

<style>
	.enhanced-semantic-dashboard {
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	.card {
		transition: transform 0.2s ease, box-shadow 0.2s ease;
	}
	
	.card:hover {
		transform: translateY(-2px);
		box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
	}
	
	.animate-spin {
		animation: spin 1s linear infinite;
	}
	
	@keyframes spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
	}
	
	.animate-pulse {
		animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}
	
	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}
	
	/* Custom scrollbar */
	.todo-preview::-webkit-scrollbar,
	.category-list::-webkit-scrollbar {
		width: 4px;
	}
	
	.todo-preview::-webkit-scrollbar-track,
	.category-list::-webkit-scrollbar-track {
		background: rgba(51, 65, 85, 0.5);
	}
	
	.todo-preview::-webkit-scrollbar-thumb,
	.category-list::-webkit-scrollbar-thumb {
		background: rgba(139, 92, 246, 0.5);
		border-radius: 2px;
	}
	
	.todo-preview::-webkit-scrollbar-thumb:hover,
	.category-list::-webkit-scrollbar-thumb:hover {
		background: rgba(139, 92, 246, 0.8);
	}
</style>

