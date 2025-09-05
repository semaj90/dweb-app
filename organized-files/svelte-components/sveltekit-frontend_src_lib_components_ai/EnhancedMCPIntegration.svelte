<script lang="ts">
  interface Props {
    caseId: string | undefined ;
    enableRealtimeUpdates: boolean
    showMetrics: boolean
    enableClusterMode: boolean
  }
  let {
    caseId = undefined,
    enableRealtimeUpdates = true,
    showMetrics = true,
    enableClusterMode = true
  } = $props();



	/**
	 * Enhanced MCP Integration Component for SvelteKit Frontend
	 * Connects cluster system, MCP tools, and Context7 integration
	 */
	
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import { page } from '$app/stores';
	
	// Props using Svelte 5 syntax compatibility
					
	// Reactive state using writable stores for Svelte 4/5 compatibility
	const mcpStatus = writable<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
	const clusterMetrics = writable<{
		activeWorkers: number
		totalRequests: number
		successRate: number
		averageResponseTime: number
		cacheHitRate: number
	}>({
		activeWorkers: 0,
		totalRequests: 0,
		successRate: 0,
		averageResponseTime: 0,
		cacheHitRate: 0
	});
	
	const mcpTools = writable<Array<{
		id: string
		name: string
		description: string
		status: 'available' | 'busy' | 'error';
		lastUsed?: Date;
		successCount: number
		errorCount: number
	}>>([]);
	
	const queryResults = writable<Array<{
		id: string
		query: string
		result: any
		source: 'enhanced_rag' | 'memory_graph' | 'context7' | 'agent_orchestration';
		timestamp: Date
		responseTime: number
		success: boolean
	}>>([]);
	
	const contextualSuggestions = writable<Array<{
		type: 'mcp_tool' | 'context7_doc' | 'memory_relation' | 'agent_action';
		title: string
		description: string
		action: () => Promise<void>;
		priority: 'high' | 'medium' | 'low';
	}>>([]);
	
	// WebSocket connection for real-time updates
	let wsConnection: WebSocket | null = null;
	let queryInput = '';
	let selectedTool = '';
	let isProcessing = false;
	
	// Available MCP tools
	const availableMCPTools = [
		{ id: 'enhanced_rag_query', name: 'Enhanced RAG Query', description: 'Semantic search with Context7 integration' },
		{ id: 'mcp_memory2_create_relations', name: 'Memory Relations', description: 'Create knowledge graph relations' },
		{ id: 'mcp_memory2_read_graph', name: 'Read Memory Graph', description: 'Query knowledge graph' },
		{ id: 'mcp_memory2_search_nodes', name: 'Search Memory Nodes', description: 'Find specific memory nodes' },
		{ id: 'mcp_context72_get-library-docs', name: 'Context7 Docs', description: 'Get library documentation' },
		{ id: 'mcp_context72_resolve-library-id', name: 'Resolve Library ID', description: 'Resolve library identifiers' },
		{ id: 'agent_orchestrate_claude', name: 'Claude Agent', description: 'Orchestrate Claude AI agent' },
		{ id: 'agent_orchestrate_crewai', name: 'CrewAI Agent', description: 'Orchestrate CrewAI agent' },
		{ id: 'agent_orchestrate_autogen', name: 'AutoGen Agent', description: 'Orchestrate AutoGen agent' }
	];
	
	onMount(async () => {
		await initializeMCPConnection();
		if (enableRealtimeUpdates) {
			setupWebSocketConnection();
		}
		await loadInitialData();
		startMetricsPolling();
	});
	
	async function initializeMCPConnection() {
		mcpStatus.set('connecting');
		
		try {
			// Test connection to Context7 MCP server
			const response = await fetch('http://localhost:40000/health');
			if (response.ok) {
				mcpStatus.set('connected');
				console.log('🚀 Enhanced MCP Integration connected');
				
				// Initialize available tools
				mcpTools.set(availableMCPTools.map(tool => ({
					...tool,
					status: 'available',
					successCount: 0,
					errorCount: 0
				})));
				
			} else {
				throw new Error('MCP server not responding');
			}
		} catch (error) {
			console.error('❌ MCP connection failed:', error);
			mcpStatus.set('error');
		}
	}
	
	function setupWebSocketConnection() {
		try {
			wsConnection = new WebSocket('ws://localhost:40000');
			
			wsConnection.onopen = () => {
				console.log('📡 WebSocket connected for real-time updates');
			};
			
			wsConnection.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleRealtimeUpdate(data);
			};
			
			wsConnection.onclose = () => {
				console.log('📡 WebSocket disconnected');
				// Attempt reconnection after 5 seconds
				setTimeout(setupWebSocketConnection, 5000);
			};
		} catch (error) {
			console.error('WebSocket connection failed:', error);
		}
	}
	
	function handleRealtimeUpdate(data: any) {
		switch (data.type) {
			case 'cluster-metrics-update':
				clusterMetrics.set(data.metrics);
				break;
			case 'mcp-tool-status':
				mcpTools.update(tools => 
					tools.map(tool => 
						tool.id === data.toolId 
							? { ...tool, status: data.status, lastUsed: new Date() }
							: tool
					)
				);
				break;
			case 'query-result':
				queryResults.update(results => [data.result, ...results.slice(0, 9)]);
				break;
		}
	}
	
	async function loadInitialData() {
		// Load contextual suggestions based on current page/case
		const suggestions = await generateContextualSuggestions();
		contextualSuggestions.set(suggestions);
	}
	
	async function generateContextualSuggestions() {
		const suggestions = [];
		
		// Case-specific suggestions
		if (caseId) {
			suggestions.push({
				type: 'mcp_tool',
				title: 'Analyze Case Evidence',
				description: 'Run enhanced RAG analysis on case evidence',
				action: async () => {
					await executeMCPTool('enhanced_rag_query', {
						query: `Analyze all evidence for case ${caseId}`,
						caseId,
						maxResults: 10,
						includeContext7: true
					});
				},
				priority: 'high'
			});
			
			suggestions.push({
				type: 'memory_relation',
				title: 'Update Knowledge Graph',
				description: 'Create memory relations for current case',
				action: async () => {
					await executeMCPTool('mcp_memory2_create_relations', {
						entities: [
							{
								type: 'case',
								id: caseId,
								properties: {
									analyzed_at: new Date().toISOString(),
									source: 'enhanced_mcp_integration'
								}
							}
						]
					});
				},
				priority: 'medium'
			});
		}
		
		// Context7 documentation suggestions
		suggestions.push({
			type: 'context7_doc',
			title: 'Get SvelteKit Best Practices',
			description: 'Fetch Context7 documentation for SvelteKit',
			action: async () => {
				await executeMCPTool('mcp_context72_get-library-docs', {
					libraryId: '/sveltejs/kit',
					topic: 'best-practices'
				});
			},
			priority: 'low'
		});
		
		return suggestions;
	}
	
	function startMetricsPolling() {
		setInterval(async () => {
			if ($mcpStatus === 'connected') {
				await updateClusterMetrics();
			}
		}, 5000); // Update every 5 seconds
	}
	
	async function updateClusterMetrics() {
		try {
			const response = await fetch('http://localhost:40000/mcp/metrics');
			if (response.ok) {
				const data = await response.json();
				if (data.success) {
					clusterMetrics.set({
						activeWorkers: data.metrics.connections || 0,
						totalRequests: data.metrics.totalRequests || 0,
						successRate: 1 - (data.metrics.errorRate || 0),
						averageResponseTime: data.metrics.averageResponseTime || 0,
						cacheHitRate: data.metrics.cache?.hitRate || 0
					});
				}
			}
		} catch (error) {
			console.error('Failed to update cluster metrics:', error);
		}
	}
	
	async function executeMCPTool(toolId: string, args: any = {}) {
		if (isProcessing) return;
		
		isProcessing = true;
		const startTime = Date.now();
		
		try {
			// Update tool status
			mcpTools.update(tools =>
				tools.map(tool =>
					tool.id === toolId ? { ...tool, status: 'busy' } : tool
				)
			);
			
			// Determine the appropriate endpoint based on tool type
			let endpoint = '';
			switch (toolId) {
				case 'enhanced_rag_query':
					endpoint = '/mcp/enhanced-rag/query';
					break;
				case 'mcp_memory2_create_relations':
					endpoint = '/mcp/memory/create-relations';
					break;
				case 'mcp_memory2_read_graph':
					endpoint = '/mcp/memory/read-graph';
					break;
				case 'mcp_memory2_search_nodes':
					endpoint = '/mcp/memory/search-nodes';
					break;
				case 'mcp_context72_get-library-docs':
					endpoint = '/mcp/context7/get-library-docs';
					break;
				case 'mcp_context72_resolve-library-id':
					endpoint = '/mcp/context7/resolve-library-id';
					break;
				case 'agent_orchestrate_claude':
				case 'agent_orchestrate_crewai':
				case 'agent_orchestrate_autogen':
					endpoint = '/mcp/agent/orchestrate';
					args.agent = toolId.replace('agent_orchestrate_', '');
					break;
				default:
					throw new Error(`Unknown tool: ${toolId}`);
			}
			
			const response = await fetch(`http://localhost:40000${endpoint}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(args)
			});
			
			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}
			
			const result = await response.json();
			const responseTime = Date.now() - startTime;
			
			// Update query results
			queryResults.update(results => [{
				id: crypto.randomUUID(),
				query: JSON.stringify(args),
				result,
				source: getSourceType(toolId),
				timestamp: new Date(),
				responseTime,
				success: true
			}, ...results.slice(0, 9)]);
			
			// Update tool success count
			mcpTools.update(tools =>
				tools.map(tool =>
					tool.id === toolId 
						? { ...tool, status: 'available', successCount: tool.successCount + 1, lastUsed: new Date() }
						: tool
				)
			);
			
			console.log(`✅ MCP tool ${toolId} executed successfully in ${responseTime}ms`);
			
		} catch (error) {
			console.error(`❌ MCP tool ${toolId} failed:`, error);
			
			// Update tool error count
			mcpTools.update(tools =>
				tools.map(tool =>
					tool.id === toolId 
						? { ...tool, status: 'error', errorCount: tool.errorCount + 1 }
						: tool
				)
			);
			
			// Add error to results
			queryResults.update(results => [{
				id: crypto.randomUUID(),
				query: JSON.stringify(args),
				result: { error: error.message },
				source: getSourceType(toolId),
				timestamp: new Date(),
				responseTime: Date.now() - startTime,
				success: false
			}, ...results.slice(0, 9)]);
		} finally {
			isProcessing = false;
		}
	}
	
	function getSourceType(toolId: string): 'enhanced_rag' | 'memory_graph' | 'context7' | 'agent_orchestration' {
		if (toolId.includes('rag')) return 'enhanced_rag';
		if (toolId.includes('memory')) return 'memory_graph';
		if (toolId.includes('context7')) return 'context7';
		if (toolId.includes('agent')) return 'agent_orchestration';
		return 'enhanced_rag';
	}
	
	async function executeQuery() {
		if (!queryInput.trim() || !selectedTool) return;
		
		const args = selectedTool === 'enhanced_rag_query' 
			? { query: queryInput, caseId, maxResults: 10, includeContext7: true }
			: selectedTool.includes('memory') && selectedTool.includes('search')
			? { query: queryInput }
			: selectedTool.includes('context7')
			? { libraryName: queryInput }
			: { prompt: queryInput, context: 'sveltekit_frontend' };
			
		await executeMCPTool(selectedTool, args);
		queryInput = '';
	}
</script>

<div class="enhanced-mcp-integration">
	<div class="mcp-header">
		<h2 class="mcp-title">
			🤖 Enhanced MCP Integration
			<span class="status-indicator status-{$mcpStatus}"></span>
		</h2>
		<div class="connection-status">
			Status: <span class="status-text">{$mcpStatus}</span>
		</div>
	</div>

	{#if showMetrics && $mcpStatus === 'connected'}
		<div class="cluster-metrics">
			<h3>📊 Cluster Performance Metrics</h3>
			<div class="metrics-grid">
				<div class="metric">
					<span class="metric-label">Active Workers</span>
					<span class="metric-value">{$clusterMetrics.activeWorkers}</span>
				</div>
				<div class="metric">
					<span class="metric-label">Total Requests</span>
					<span class="metric-value">{$clusterMetrics.totalRequests.toLocaleString()}</span>
				</div>
				<div class="metric">
					<span class="metric-label">Success Rate</span>
					<span class="metric-value">{($clusterMetrics.successRate * 100).toFixed(1)}%</span>
				</div>
				<div class="metric">
					<span class="metric-label">Avg Response</span>
					<span class="metric-value">{$clusterMetrics.averageResponseTime.toFixed(0)}ms</span>
				</div>
				<div class="metric">
					<span class="metric-label">Cache Hit Rate</span>
					<span class="metric-value">{($clusterMetrics.cacheHitRate * 100).toFixed(1)}%</span>
				</div>
			</div>
		</div>
	{/if}

	<div class="mcp-interface">
		<div class="query-section">
			<h3>🔍 MCP Query Interface</h3>
			<div class="query-form">
				<select bind:value={selectedTool} class="tool-selector">
					<option value="">Select MCP Tool</option>
					{#each availableMCPTools as tool}
						<option value={tool.id}>{tool.name} - {tool.description}</option>
					{/each}
				</select>
				<input 
					bind:value={queryInput}
					placeholder="Enter your query..."
					class="query-input"
					onkeydown={(e) => e.key === 'Enter' && executeQuery()}
				/>
				<button 
					onclick={executeQuery}
					disabled={!queryInput.trim() || !selectedTool || isProcessing}
					class="execute-button"
				>
					{isProcessing ? '⏳ Processing...' : '🚀 Execute'}
				</button>
			</div>
		</div>

		<div class="suggestions-section">
			<h3>💡 Contextual Suggestions</h3>
			<div class="suggestions-list">
				{#each $contextualSuggestions as suggestion}
					<button 
						class="suggestion-item suggestion-{suggestion.priority}"
						onclick={suggestion.action}
						disabled={isProcessing}
					>
						<div class="suggestion-title">{suggestion.title}</div>
						<div class="suggestion-description">{suggestion.description}</div>
						<div class="suggestion-type">{suggestion.type}</div>
					</button>
				{/each}
			</div>
		</div>

		<div class="tools-status">
			<h3>🛠️ MCP Tools Status</h3>
			<div class="tools-grid">
				{#each $mcpTools as tool}
					<div class="tool-card tool-{tool.status}">
						<div class="tool-name">{tool.name}</div>
						<div class="tool-stats">
							<span class="success-count">✅ {tool.successCount}</span>
							<span class="error-count">❌ {tool.errorCount}</span>
						</div>
						<div class="tool-status">Status: {tool.status}</div>
						{#if tool.lastUsed}
							<div class="last-used">
								Last used: {tool.lastUsed.toLocaleTimeString()}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		</div>

		<div class="results-section">
			<h3>📋 Recent Query Results</h3>
			<div class="results-list">
				{#each $queryResults as result}
					<div class="result-item result-{result.success ? 'success' : 'error'}">
						<div class="result-header">
							<span class="result-source">{result.source}</span>
							<span class="result-time">{result.responseTime}ms</span>
							<span class="result-timestamp">{result.timestamp.toLocaleTimeString()}</span>
						</div>
						<div class="result-query">{result.query}</div>
						<div class="result-content">
							{#if result.success}
								<pre>{JSON.stringify(result.result, null, 2)}</pre>
							{:else}
								<div class="error-message">Error: {result.result.error}</div>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.enhanced-mcp-integration {
		background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
		border-radius: 12px;
		padding: 24px;
		color: #e5e7eb;
		font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
	}

	.mcp-header {
		display: flex
		justify-content: space-between;
		align-items: center
		margin-bottom: 24px;
		border-bottom: 1px solid rgba(229, 231, 235, 0.1);
		padding-bottom: 16px;
	}

	.mcp-title {
		font-size: 1.5rem;
		font-weight: 600;
		display: flex
		align-items: center
		gap: 8px;
	}

	.status-indicator {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		margin-left: 8px;
	}

	.status-connected { background: #10b981; }
	.status-connecting { background: #f59e0b; animation: pulse 2s infinite; }
	.status-disconnected { background: #6b7280; }
	.status-error { background: #ef4444; }

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}

	.connection-status {
		font-size: 0.875rem;
		color: #9ca3af;
	}

	.cluster-metrics {
		margin-bottom: 24px;
		background: rgba(255, 255, 255, 0.05);
		border-radius: 8px;
		padding: 16px;
	}

	.metrics-grid {
		display: grid
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 16px;
		margin-top: 12px;
	}

	.metric {
		display: flex
		flex-direction: column
		align-items: center
		text-align: center
	}

	.metric-label {
		font-size: 0.75rem;
		color: #9ca3af;
		margin-bottom: 4px;
	}

	.metric-value {
		font-size: 1.25rem;
		font-weight: 600;
		color: #10b981;
	}

	.mcp-interface {
		display: grid
		gap: 24px;
	}

	.query-section h3,
	.suggestions-section h3,
	.tools-status h3,
	.results-section h3 {
		font-size: 1.125rem;
		font-weight: 500;
		margin-bottom: 12px;
		color: #f3f4f6;
	}

	.query-form {
		display: flex
		gap: 12px;
		align-items: center
		flex-wrap: wrap
	}

	.tool-selector,
	.query-input {
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid rgba(255, 255, 255, 0.2);
		border-radius: 6px;
		padding: 8px 12px;
		color: #e5e7eb;
		font-size: 0.875rem;
	}

	.tool-selector {
		min-width: 200px;
	}

	.query-input {
		flex: 1;
		min-width: 200px;
	}

	.execute-button {
		background: linear-gradient(135deg, #3b82f6, #1d4ed8);
		border: none
		border-radius: 6px;
		padding: 8px 16px;
		color: white
		font-weight: 500;
		cursor: pointer
		transition: all 0.2s;
	}

	.execute-button:hover:not(:disabled) {
		transform: translateY(-1px);
		box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
	}

	.execute-button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.suggestions-list {
		display: grid
		gap: 8px;
	}

	.suggestion-item {
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 6px;
		padding: 12px;
		text-align: left
		cursor: pointer
		transition: all 0.2s;
	}

	.suggestion-item:hover:not(:disabled) {
		background: rgba(255, 255, 255, 0.1);
		transform: translateY(-1px);
	}

	.suggestion-high { border-left: 4px solid #ef4444; }
	.suggestion-medium { border-left: 4px solid #f59e0b; }
	.suggestion-low { border-left: 4px solid #10b981; }

	.tools-grid {
		display: grid
		grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
		gap: 12px;
	}

	.tool-card {
		background: rgba(255, 255, 255, 0.05);
		border-radius: 6px;
		padding: 12px;
		border-left: 4px solid #6b7280;
	}

	.tool-available { border-left-color: #10b981; }
	.tool-busy { border-left-color: #f59e0b; }
	.tool-error { border-left-color: #ef4444; }

	.tool-name {
		font-weight: 500;
		margin-bottom: 8px;
	}

	.tool-stats {
		display: flex
		gap: 12px;
		font-size: 0.75rem;
		margin-bottom: 4px;
	}

	.results-list {
		max-height: 400px;
		overflow-y: auto
		display: grid
		gap: 12px;
	}

	.result-item {
		background: rgba(255, 255, 255, 0.05);
		border-radius: 6px;
		padding: 12px;
		border-left: 4px solid #6b7280;
	}

	.result-success { border-left-color: #10b981; }
	.result-error { border-left-color: #ef4444; }

	.result-header {
		display: flex
		gap: 12px;
		font-size: 0.75rem;
		color: #9ca3af;
		margin-bottom: 8px;
	}

	.result-content {
		font-size: 0.875rem;
		max-height: 200px;
		overflow-y: auto
	}

	.result-content pre {
		background: rgba(0, 0, 0, 0.3);
		padding: 8px;
		border-radius: 4px;
		white-space: pre-wrap;
		word-wrap: break-word;
	}

	.error-message {
		color: #ef4444;
		font-weight: 500;
	}
</style>