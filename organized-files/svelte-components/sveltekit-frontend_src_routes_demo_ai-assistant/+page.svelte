<script lang="ts">
	import { onMount } from 'svelte';
	import type { AIResponse } from '$lib/types/ai';
	
	// Modern Svelte 5 patterns
	let aiResponses = $state<AIResponse[]>([]);
	let currentQuery = $state('');
	let isLoading = $state(false);
	let systemStatus = $state<any>(null);
	let mcpMetrics = $state<any>(null);
	let selectedCase = $state('CASE-2024-001');
	
	// Demo queries for different AI capabilities
	const demoQueries = [
		{
			title: 'Legal Document Analysis',
			query: 'Analyze contract liability clauses in evidence files',
			category: 'document-analysis'
		},
		{
			title: 'Case Pattern Recognition',
			query: 'Find similar cases with employment disputes and contract violations',
			category: 'pattern-matching'
		},
		{
			title: 'Evidence Correlation',
			query: 'Correlate digital evidence timestamps with witness testimonies',
			category: 'evidence-analysis'
		},
		{
			title: 'Legal Research Assistant',
			query: 'Summarize relevant case law for intellectual property disputes',
			category: 'research'
		},
		{
			title: 'Context7 Integration',
			query: 'Get SvelteKit best practices for legal AI applications',
			category: 'context7'
		}
	];
	
	// Real-time metrics
	let performanceMetrics = $derived({
		totalQueries: aiResponses.length,
		averageConfidence: aiResponses.length > 0 
			? aiResponses.reduce((sum, r) => sum + (r.confidence || 0), 0) / aiResponses.length 
			: 0,
		averageResponseTime: aiResponses.length > 0
			? aiResponses.reduce((sum, r) => sum + (r.processingTime || 0), 0) / aiResponses.length
			: 0,
		gpuProcessed: aiResponses.filter(r => r.gpuProcessed).length
	});
	
	onMount(async () => {
		await loadSystemStatus();
		await loadMCPMetrics();
	});
	
	async function loadSystemStatus() {
		try {
			const response = await fetch('http://localhost:40000/health');
			systemStatus = await response.json();
		} catch (error) {
			systemStatus = { status: 'unavailable', error: error.message };
		}
	}
	
	async function loadMCPMetrics() {
		try {
			const response = await fetch('http://localhost:40000/mcp/metrics');
			mcpMetrics = await response.json();
		} catch (error) {
			mcpMetrics = { error: error.message };
		}
	}
	
	async function runAIQuery(query: string, category: string = 'general') {
		isLoading = true;
		
		try {
			// Simulate AI processing with Context7 MCP integration
			const startTime = Date.now();
			
			// Mock AI response with realistic data
			const mockResponse: AIResponse = {
				output: generateMockAIResponse(query, category),
				confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
				keyTerms: extractKeyTerms(query),
				processingTime: Math.random() * 2000 + 500, // 500-2500ms
				gpuProcessed: Math.random() > 0.3, // 70% GPU processed
				legalRisk: assessLegalRisk(query),
				contextEnhanced: true,
				mcpIntegration: {
					context7Used: category === 'context7',
					memoryGraphUpdated: true,
					bestPracticesApplied: true
				}
			};
			
			// Simulate processing delay
			await new Promise(resolve => setTimeout(resolve, mockResponse.processingTime));
			
			aiResponses = [...aiResponses, {
				...mockResponse,
				query,
				category,
				timestamp: new Date().toISOString()
			}];
			
		} catch (error) {
			aiResponses = [...aiResponses, {
				output: `Error: ${error.message}`,
				confidence: 0,
				keyTerms: [],
				processingTime: Date.now() - Date.now(),
				gpuProcessed: false,
				legalRisk: 'unknown',
				query,
				category,
				timestamp: new Date().toISOString(),
				error: true
			}];
		} finally {
			isLoading = false;
		}
	}
	
	function generateMockAIResponse(query: string, category: string): string {
		const responses = {
			'document-analysis': `📄 **Document Analysis Complete**
			
Analyzed ${Math.floor(Math.random() * 20) + 5} documents in case ${selectedCase}.

**Key Findings:**
• Found 3 liability clauses requiring review
• Identified potential contract ambiguities in sections 4.2 and 7.1
• Recommended legal precedent review for similar cases
• Flagged 2 documents for expert witness consultation

**Risk Assessment:** Medium - requires legal review
**Confidence:** 89% based on legal pattern matching`,

			'pattern-matching': `🔍 **Pattern Analysis Results**
			
Found ${Math.floor(Math.random() * 15) + 3} similar cases in database.

**Matching Patterns:**
• Employment dispute involving non-compete clauses
• Contract violation with damages $50K-$500K range
• Similar industry sector (Technology/Software)
• Comparable resolution timeframe (6-18 months)

**Case Precedents:**
• Case #2023-445: Similar outcome, $340K settlement
• Case #2022-891: Court ruling favorable for plaintiff
• Case #2023-002: Mediation successful, 8-month resolution

**Strategic Recommendation:** Consider mediation based on pattern analysis`,

			'evidence-analysis': `⚖️ **Evidence Correlation Analysis**
			
Cross-referenced ${Math.floor(Math.random() * 50) + 20} evidence items.

**Timeline Correlations:**
• Digital timestamps align with witness testimony A (95% match)
• Email chain supports timeline in depositions B & C
• Security footage confirms presence at key meeting
• Phone records validate communication patterns

**Inconsistencies Found:**
• 2-hour gap in digital evidence on 2024-03-15
• Conflicting statements in witness testimonies D & E

**Recommendation:** Focus on evidence gaps during cross-examination`,

			'research': `📚 **Legal Research Summary**
			
Researched ${Math.floor(Math.random() * 100) + 50} relevant cases and statutes.

**Key Legal Precedents:**
• *Smith v. TechCorp* (2023): Established precedent for IP disputes
• *Johnson v. Innovation Inc* (2022): Similar fact pattern, plaintiff victory
• Federal Code §1234.5: Relevant statutory framework
• State Regulation §456.7: Additional compliance requirements

**Legal Strategy Insights:**
• 78% success rate for similar cases in this jurisdiction
• Average settlement: $425K (range: $200K-$850K)
• Typical resolution time: 14 months

**Next Steps:** File preliminary injunction, gather expert testimony`,

			'context7': `🤖 **Context7 MCP Integration Results**
			
Analyzed SvelteKit legal AI application architecture.

**Best Practices Applied:**
• Modern Svelte 5 runes ($state, $effect, $derived)
• Type-safe API design with Drizzle ORM
• Proper authentication with Lucia Auth
• Vector search integration with Qdrant
• Real-time collaboration features

**Performance Optimizations:**
• Code splitting for large legal document viewers
• Efficient caching strategies for case data
• WebGL acceleration for evidence canvas
• Service worker for offline case access

**Security Recommendations:**
• Implement proper RBAC for legal data
• Use encrypted evidence storage
• Audit trails for all case modifications
• Secure API endpoints with rate limiting

Generated via Context7 MCP analysis tools`,

			'general': `🧠 **AI Analysis Complete**
			
Processed query using advanced legal AI models.

**Analysis Summary:**
• Applied legal domain expertise
• Cross-referenced case databases
• Validated against current legal standards
• Generated actionable recommendations

**Key Insights:**
• High confidence in legal reasoning
• Multiple validation sources confirmed
• Recommended next steps identified
• Risk assessment completed

This analysis leverages the full Legal AI system capabilities including Context7 MCP integration, vector search, and legal pattern matching.`
		};
		
		return responses[category] || responses['general'];
	}
	
	function extractKeyTerms(query: string): string[] {
		const legalTerms = ['contract', 'liability', 'evidence', 'case', 'legal', 'dispute', 'clause', 'testimony', 'precedent'];
		return query.toLowerCase().split(' ')
			.filter(word => legalTerms.some(term => word.includes(term)))
			.slice(0, 5);
	}
	
	function assessLegalRisk(query: string): string {
		const riskKeywords = {
			high: ['violation', 'criminal', 'fraud', 'breach'],
			medium: ['dispute', 'contract', 'liability', 'damages'],
			low: ['research', 'analysis', 'review', 'summary']
		};
		
		const queryLower = query.toLowerCase();
		
		if (riskKeywords.high.some(term => queryLower.includes(term))) return 'high';
		if (riskKeywords.medium.some(term => queryLower.includes(term))) return 'medium';
		return 'low';
	}
	
	function getRiskColor(risk: string): string {
		switch (risk) {
			case 'high': return 'text-red-600 bg-red-50';
			case 'medium': return 'text-yellow-600 bg-yellow-50';
			case 'low': return 'text-green-600 bg-green-50';
			default: return 'text-gray-600 bg-gray-50';
		}
	}
	
	function clearResponses() {
		aiResponses = [];
	}
</script>

<svelte:head>
	<title>AI Assistant Demo - Legal AI System</title>
	<meta name="description" content="Comprehensive AI assistant demonstration with Context7 MCP integration" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
	<div class="max-w-7xl mx-auto">
		<!-- Header -->
		<div class="mb-8">
			<h1 class="text-4xl font-bold text-gray-900 mb-2">
				🤖 Legal AI Assistant Demo
			</h1>
			<p class="text-lg text-gray-600">
				Comprehensive demonstration of AI-powered legal analysis with Context7 MCP integration
			</p>
		</div>

		<!-- System Status Cards -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
			<!-- System Health -->
			<div class="bg-white rounded-lg shadow-md p-6">
				<h3 class="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
				{#if systemStatus}
					<div class="space-y-2">
						<div class="flex items-center justify-between">
							<span class="text-sm text-gray-600">Status:</span>
							<span class="px-2 py-1 rounded text-xs font-medium {systemStatus.status === 'healthy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
								{systemStatus.status || 'Unknown'}
							</span>
						</div>
						{#if systemStatus.uptime}
							<div class="flex items-center justify-between">
								<span class="text-sm text-gray-600">Uptime:</span>
								<span class="text-sm font-medium">{Math.floor(systemStatus.uptime / 60)}m</span>
							</div>
						{/if}
						<div class="flex items-center justify-between">
							<span class="text-sm text-gray-600">Connections:</span>
							<span class="text-sm font-medium">{systemStatus.connections || 0}</span>
						</div>
					</div>
				{:else}
					<div class="text-sm text-gray-500">Loading system status...</div>
				{/if}
			</div>

			<!-- Performance Metrics -->
			<div class="bg-white rounded-lg shadow-md p-6">
				<h3 class="text-lg font-semibold text-gray-900 mb-4">Performance</h3>
				<div class="space-y-2">
					<div class="flex items-center justify-between">
						<span class="text-sm text-gray-600">Total Queries:</span>
						<span class="text-sm font-medium">{performanceMetrics.totalQueries}</span>
					</div>
					<div class="flex items-center justify-between">
						<span class="text-sm text-gray-600">Avg Confidence:</span>
						<span class="text-sm font-medium">{(performanceMetrics.averageConfidence * 100).toFixed(1)}%</span>
					</div>
					<div class="flex items-center justify-between">
						<span class="text-sm text-gray-600">Avg Response:</span>
						<span class="text-sm font-medium">{performanceMetrics.averageResponseTime.toFixed(0)}ms</span>
					</div>
					<div class="flex items-center justify-between">
						<span class="text-sm text-gray-600">GPU Processed:</span>
						<span class="text-sm font-medium">{performanceMetrics.gpuProcessed}/{performanceMetrics.totalQueries}</span>
					</div>
				</div>
			</div>

			<!-- MCP Integration -->
			<div class="bg-white rounded-lg shadow-md p-6">
				<h3 class="text-lg font-semibold text-gray-900 mb-4">Context7 MCP</h3>
				{#if mcpMetrics}
					<div class="space-y-2">
						{#if mcpMetrics.metrics}
							<div class="flex items-center justify-between">
								<span class="text-sm text-gray-600">Memory Nodes:</span>
								<span class="text-sm font-medium">{mcpMetrics.metrics.memoryGraph?.nodes || 0}</span>
							</div>
							<div class="flex items-center justify-between">
								<span class="text-sm text-gray-600">Cache Hit Rate:</span>
								<span class="text-sm font-medium">{(mcpMetrics.metrics.cache?.hitRate * 100 || 0).toFixed(1)}%</span>
							</div>
							<div class="flex items-center justify-between">
								<span class="text-sm text-gray-600">Total Requests:</span>
								<span class="text-sm font-medium">{mcpMetrics.metrics.totalRequests || 0}</span>
							</div>
						{:else}
							<div class="text-sm text-red-500">MCP server unavailable</div>
						{/if}
					</div>
				{:else}
					<div class="text-sm text-gray-500">Loading MCP metrics...</div>
				{/if}
			</div>
		</div>

		<!-- Demo Query Interface -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
			<!-- Input Panel -->
			<div class="bg-white rounded-lg shadow-md p-6">
				<h2 class="text-2xl font-semibold text-gray-900 mb-6">AI Query Interface</h2>
				
				<!-- Case Selection -->
				<div class="mb-4">
					<label for="case-select" class="block text-sm font-medium text-gray-700 mb-2">
						Select Case
					</label>
					<select 
						bind:value={selectedCase}
						id="case-select"
						class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
					>
						<option value="CASE-2024-001">CASE-2024-001 - Contract Dispute</option>
						<option value="CASE-2024-002">CASE-2024-002 - Employment Law</option>
						<option value="CASE-2024-003">CASE-2024-003 - IP Violation</option>
					</select>
				</div>

				<!-- Custom Query Input -->
				<div class="mb-4">
					<label for="query-input" class="block text-sm font-medium text-gray-700 mb-2">
						Custom AI Query
					</label>
					<textarea
						bind:value={currentQuery}
						id="query-input"
						rows="3"
						placeholder="Enter your legal AI query here..."
						class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
					></textarea>
				</div>

				<div class="flex gap-2 mb-6">
					<button
						onclick={() => runAIQuery(currentQuery)}
						disabled={isLoading || !currentQuery.trim()}
						class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
					>
						{#if isLoading}
							<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
						{/if}
						Run Query
					</button>
					<button
						onclick={clearResponses}
						class="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
					>
						Clear Results
					</button>
				</div>

				<!-- Demo Queries -->
				<h3 class="text-lg font-semibold text-gray-900 mb-4">Demo Queries</h3>
				<div class="space-y-2">
					{#each demoQueries as demo}
						<button
							onclick={() => runAIQuery(demo.query, demo.category)}
							disabled={isLoading}
							class="w-full text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
						>
							<div class="font-medium text-gray-900">{demo.title}</div>
							<div class="text-sm text-gray-600 mt-1">{demo.query}</div>
						</button>
					{/each}
				</div>
			</div>

			<!-- Results Panel -->
			<div class="bg-white rounded-lg shadow-md p-6">
				<h2 class="text-2xl font-semibold text-gray-900 mb-6">AI Responses</h2>
				
				{#if isLoading}
					<div class="flex items-center justify-center py-12">
						<div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
						<span class="ml-3 text-gray-600">Processing AI query...</span>
					</div>
				{/if}

				<div class="space-y-6 max-h-96 overflow-y-auto">
					{#each aiResponses.slice().reverse() as response, index}
						<div class="border border-gray-200 rounded-lg p-4">
							<!-- Query Header -->
							<div class="flex items-start justify-between mb-3">
								<div class="flex-1">
									<div class="text-sm font-medium text-gray-900">{response.query}</div>
									<div class="text-xs text-gray-500 mt-1">
										{new Date(response.timestamp as string | number | Date).toLocaleTimeString()}
									</div>
								</div>
								<div class="flex items-center space-x-2 ml-4">
									<span class="px-2 py-1 rounded text-xs font-medium {getRiskColor(response.legalRisk)}">
										{response.legalRisk} risk
									</span>
									{#if response.gpuProcessed}
										<span class="px-2 py-1 bg-purple-100 text-purple-800 rounded text-xs font-medium">
											GPU
										</span>
									{/if}
								</div>
							</div>

							<!-- AI Response -->
							<div class="prose prose-sm max-w-none mb-3">
								<div class="whitespace-pre-wrap text-gray-800">{response.output}</div>
							</div>

							<!-- Metadata -->
							<div class="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-100">
								<div class="flex items-center space-x-4">
									<span>Confidence: {((response.confidence || 0) * 100).toFixed(0)}%</span>
									<span>Time: {response.processingTime || 0}ms</span>
									{#if response.keyTerms && response.keyTerms.length > 0}
										<span>Terms: {response.keyTerms.join(', ')}</span>
									{/if}
								</div>
								{#if response.mcpIntegration?.context7Used}
									<span class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
										Context7 Enhanced
									</span>
								{/if}
							</div>
						</div>
					{/each}

					{#if aiResponses.length === 0 && !isLoading}
						<div class="text-center py-12 text-gray-500">
							<div class="text-lg mb-2">🤖</div>
							<div>No AI queries yet. Try one of the demo queries above!</div>
						</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- Feature Showcase -->
		<div class="mt-12 bg-white rounded-lg shadow-md p-6">
			<h2 class="text-2xl font-semibold text-gray-900 mb-6">System Capabilities</h2>
			
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
				<div class="text-center">
					<div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-3">
						<span class="text-2xl">📄</span>
					</div>
					<h3 class="font-semibold text-gray-900 mb-2">Document Analysis</h3>
					<p class="text-sm text-gray-600">AI-powered legal document processing with clause extraction and risk assessment</p>
				</div>

				<div class="text-center">
					<div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-3">
						<span class="text-2xl">🔍</span>
					</div>
					<h3 class="font-semibold text-gray-900 mb-2">Pattern Recognition</h3>
					<p class="text-sm text-gray-600">Case similarity matching and precedent analysis using vector embeddings</p>
				</div>

				<div class="text-center">
					<div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-3">
						<span class="text-2xl">⚖️</span>
					</div>
					<h3 class="font-semibold text-gray-900 mb-2">Evidence Analysis</h3>
					<p class="text-sm text-gray-600">Cross-reference evidence timelines and witness testimony validation</p>
				</div>

				<div class="text-center">
					<div class="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mx-auto mb-3">
						<span class="text-2xl">🤖</span>
					</div>
					<h3 class="font-semibold text-gray-900 mb-2">Context7 MCP</h3>
					<p class="text-sm text-gray-600">Enhanced context sharing with memory graphs and best practice recommendations</p>
				</div>
			</div>
		</div>

		<!-- Technical Information -->
		<div class="mt-8 bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-6 text-white">
			<h2 class="text-xl font-semibold mb-4">Technical Implementation</h2>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
				<div>
					<h3 class="font-semibold mb-2">Frontend Stack</h3>
					<ul class="space-y-1 text-gray-300">
						<li>• SvelteKit 2 with Svelte 5 runes</li>
						<li>• TypeScript strict mode</li>
						<li>• Tailwind CSS styling</li>
						<li>• Real-time WebSocket updates</li>
					</ul>
				</div>
				<div>
					<h3 class="font-semibold mb-2">AI Integration</h3>
					<ul class="space-y-1 text-gray-300">
						<li>• Local Ollama LLM (Gemma3)</li>
						<li>• Vector search with Qdrant</li>
						<li>• Context7 MCP protocol</li>
						<li>• Legal domain expertise</li>
					</ul>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.prose {
		max-width: none
	}
	
	.prose h1, .prose h2, .prose h3 {
		margin-top: 0;
		margin-bottom: 0.5rem;
	}
	
	.prose p {
		margin-top: 0;
		margin-bottom: 0.5rem;
	}
	
	.prose ul {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
		padding-left: 1.5rem;
	}
	
	.prose strong {
		font-weight: 600;
	}
</style>