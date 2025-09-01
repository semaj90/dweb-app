<script lang="ts">

	import { onMount } from 'svelte';
  import { fade, fly, scale } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';
	
	// Import our advanced services
	import { advancedCache } from '$lib/services/advanced-cache-manager';
	import { aiRecommendationEngine } from '$lib/services/ai-recommendation-engine';
	import { context7MCPIntegration } from '$lib/services/context7-mcp-integration';
	import TypewriterResponse from '$lib/components/ai/TypewriterResponse.svelte';
	
	// Demo state
	let cacheStats = $state({ hits: 0, misses: 0, evictions: 0, total_size: 0, items_count: 0 });
	let recommendations: any[] = $state([]);
	let bestPractices: any[] = $state([]);
	let demoQuery = $state('Review contract liability clauses for potential risks');
	let isLoading = $state(false);
	let showTypewriter = $state(false);
	let aiResponse = $state('');
	let userActivity: any[] = $state([]);
	
	// Demo data
	const legalQueries = [
		'Analyze employment contract termination clause',
		'Review intellectual property licensing agreement',
		'Assess compliance with GDPR regulations',
		'Evaluate litigation risk for breach of contract',
		'Examine patent infringement claims'
	];
	
	onMount(async () => {
		// Load cache stats
		loadCacheStats();
		
		// Generate sample user activity
		generateSampleActivity();
		
		// Load initial best practices
		await loadBestPractices();
		
		// Set up periodic updates
		const interval = setInterval(loadCacheStats, 2000);
		return () => clearInterval(interval);
	});
	
	async function loadCacheStats() {
		const stats = advancedCache.getStats();
		cacheStats = { ...(await new Promise(resolve => stats.subscribe(resolve))) };
	}
	
	async function loadBestPractices() {
		try {
			bestPractices = await context7MCPIntegration.generateBestPractices('performance');
		} catch (error) {
			console.error('Failed to load best practices:', error);
		}
	}
	
	async function generateRecommendations() {
		isLoading = true;
		
		try {
			// Simulate thinking time
			await new Promise(resolve => setTimeout(resolve, 1500));
			
			recommendations = await aiRecommendationEngine.generateRecommendations({
				userQuery: demoQuery,
				legalDomain: 'contract',
				userRole: 'legal_analyst',
				priority: 'high'
			});
			
			// Simulate AI response
			aiResponse = `Based on my analysis of "${demoQuery}", I've identified several key considerations:\n\n1. **Liability Limitations**: Review indemnification clauses for scope and mutual obligations.\n2. **Risk Assessment**: Evaluate consequential damages exclusions and caps.\n3. **Jurisdiction**: Ensure governing law aligns with business operations.\n4. **Termination**: Assess notice periods and post-termination obligations.`;
			
			showTypewriter = true;
		} catch (error) {
			console.error('Failed to generate recommendations:', error);
		} finally {
			isLoading = false;
		}
	}
	
	function generateSampleActivity() {
		userActivity = [
			{ timestamp: Date.now() - 5000, action: 'typing', content: 'Review contract', duration: 800 },
			{ timestamp: Date.now() - 4000, action: 'pause', duration: 300 },
			{ timestamp: Date.now() - 3500, action: 'typing', content: ' liability clauses', duration: 600 },
			{ timestamp: Date.now() - 2800, action: 'delete', duration: 200 },
			{ timestamp: Date.now() - 2500, action: 'typing', content: ' for potential risks', duration: 700 },
			{ timestamp: Date.now() - 1800, action: 'pause', duration: 500 }
		];
	}
	
	async function testCaching() {
		const testKey = `demo_${Date.now()}`;
		const testData = { message: 'Cached legal document', timestamp: Date.now() };
		
		// Set cache item
		await advancedCache.set(testKey, testData, {
			priority: 'high',
			ttl: 30000,
			tags: ['demo', 'legal-doc']
		});
		
		// Get cache item (should be a hit)
		const retrieved = await advancedCache.get(testKey);
		console.log('Cache test result:', retrieved);
		
		// Update stats
		await loadCacheStats();
	}
	
	async function testLazyLoading() {
		const loader = async () => {
			// Simulate API call
			await new Promise(resolve => setTimeout(resolve, 1000));
			return {
				title: 'Legal Document Analysis',
				content: 'This document has been analyzed for compliance and risk factors.',
				analysis: {
					risk_level: 'medium',
					compliance_score: 85,
					recommendations: ['Review clause 4.2', 'Update termination notice']
				}
			};
		};
		
		const result = await advancedCache.lazyLoad(
			'lazy_demo_document',
			loader,
			{ priority: 'medium', prefetch: true }
		);
		
		console.log('Lazy loading result:', result);
		await loadCacheStats();
	}
	
	function selectQuery(query: string) {
		demoQuery = query;
		showTypewriter = false;
		recommendations = [];
	}
	
	function getRiskLevelClass(level: string) {
		return `risk-indicator ${level}`;
	}
	
	function getConfidenceWidth(confidence: number) {
		return `${Math.round(confidence * 100)}%`;
	}
</script>

<!-- Load advanced interaction styles -->
<svelte:head>
	<link rel="stylesheet" href="/src/lib/styles/advanced-interactions.css">
</svelte:head>

<div class="advanced-cache-demo p-6 bg-yorha-bg-secondary min-h-screen">
	<div class="max-w-6xl mx-auto">
		<!-- Header -->
		<header class="mb-8" in:fade={{ duration: 600 }}>
			<h1 class="text-3xl font-bold font-mono text-yorha-primary mb-2">
				Advanced Caching & AI Interaction Demo
			</h1>
			<p class="text-yorha-text-secondary">
				Showcase of intelligent caching, lazy loading, typewriter effects, and AI recommendations
			</p>
		</header>

		<!-- Cache Statistics -->
		<section class="cache-stats-section mb-8" in:fly={{ y: 50, duration: 600, delay: 200 }}>
			<h2 class="text-xl font-semibold text-yorha-primary mb-4">Cache Performance Metrics</h2>
			
			<div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
				<div class="cache-stat-card bg-yorha-bg-tertiary border border-yorha-border p-4 rounded">
					<div class="text-2xl font-bold text-yorha-accent">{cacheStats.hits}</div>
					<div class="text-sm text-yorha-text-muted">Cache Hits</div>
					<div class="cache-hit-indicator mt-2"></div>
				</div>
				
				<div class="cache-stat-card bg-yorha-bg-tertiary border border-yorha-border p-4 rounded">
					<div class="text-2xl font-bold text-yorha-warning">{cacheStats.misses}</div>
					<div class="text-sm text-yorha-text-muted">Cache Misses</div>
					<div class="cache-miss-indicator mt-2"></div>
				</div>
				
				<div class="cache-stat-card bg-yorha-bg-tertiary border border-yorha-border p-4 rounded">
					<div class="text-2xl font-bold text-yorha-text-primary">{cacheStats.items_count}</div>
					<div class="text-sm text-yorha-text-muted">Items Cached</div>
				</div>
				
				<div class="cache-stat-card bg-yorha-bg-tertiary border border-yorha-border p-4 rounded">
					<div class="text-2xl font-bold text-yorha-text-primary">
						{Math.round(cacheStats.total_size / 1024)}KB
					</div>
					<div class="text-sm text-yorha-text-muted">Cache Size</div>
				</div>
				
				<div class="cache-stat-card bg-yorha-bg-tertiary border border-yorha-border p-4 rounded">
					<div class="text-2xl font-bold text-yorha-success">{cacheStats.evictions}</div>
					<div class="text-sm text-yorha-text-muted">Evictions</div>
				</div>
			</div>
			
			<div class="flex gap-4">
				<button 
					class="bg-yorha-primary text-yorha-bg-primary px-4 py-2 rounded border border-yorha-primary hover:bg-yorha-secondary transition-colors focus-ring-enhanced"
					on:onclick={testCaching}
				>
					Test Caching
				</button>
				
				<button 
					class="bg-yorha-accent text-yorha-bg-primary px-4 py-2 rounded border border-yorha-accent hover:opacity-80 transition-opacity focus-ring-enhanced"
					on:onclick={testLazyLoading}
				>
					Test Lazy Loading
				</button>
			</div>
		</section>

		<!-- AI Query Interface -->
		<section class="ai-query-section mb-8" in:fly={{ y: 50, duration: 600, delay: 400 }}>
			<h2 class="text-xl font-semibold text-yorha-primary mb-4">AI-Powered Legal Analysis</h2>
			
			<!-- Sample Queries -->
			<div class="sample-queries mb-4">
				<p class="text-sm text-yorha-text-muted mb-2">Quick start with sample queries:</p>
				<div class="flex flex-wrap gap-2">
					{#each legalQueries as query, i}
						<button
							class="sample-query-btn text-sm px-3 py-1 bg-yorha-bg-tertiary border border-yorha-border rounded hover:border-yorha-primary transition-colors focus-ring-enhanced"
							on:onclick={() => selectQuery(query)}
							in:fly={{ x: -20, duration: 400, delay: i * 100 }}
						>
							{query}
						</button>
					{/each}
				</div>
			</div>
			
			<!-- Query Input -->
			<div class="query-input-section mb-4">
				<div class="flex gap-4">
					<input
						type="text"
						bind:value={demoQuery}
						placeholder="Enter your legal query..."
						class="flex-1 px-4 py-2 bg-yorha-bg-tertiary border border-yorha-border rounded text-yorha-text-primary placeholder-yorha-text-muted focus:border-yorha-primary focus:outline-none focus-ring-enhanced"
					/>
					
					<button
						class="bg-gradient-to-r from-yorha-primary to-yorha-secondary text-yorha-bg-primary px-6 py-2 rounded border border-yorha-primary hover:scale-105 transition-transform disabled:opacity-50 disabled:cursor-not-allowed focus-ring-enhanced"
						on:onclick={generateRecommendations}
						disabled={isLoading || !demoQuery.trim()}
					>
						{isLoading ? 'Analyzing...' : 'Analyze Query'}
					</button>
				</div>
			</div>
			
			<!-- Loading State -->
			{#if isLoading}
				<div class="ai-loading-container" in:fade={{ duration: 300 }}>
					<div class="ai-loading-neural-network">
						<div class="neural-node"></div>
						<div class="neural-node"></div>
						<div class="neural-node"></div>
						<div class="neural-node"></div>
						<div class="neural-node"></div>
					</div>
					<div class="ai-processing-text">
						Processing legal query with advanced AI models...
					</div>
				</div>
			{/if}
			
			<!-- Typewriter Response -->
			{#if showTypewriter}
				<div class="typewriter-section bg-yorha-bg-tertiary border border-yorha-border p-4 rounded mb-4" 
					 in:scale={{ duration: 400, start: 0.95 }}>
					<h3 class="text-lg font-semibold text-yorha-primary mb-2">AI Analysis Results</h3>
					<TypewriterResponse 
						text={aiResponse}
						speed={30}
						showCursor={true}
						cacheKey="demo_analysis"
						{userActivity}
						enableThinking={true}
						autoStart={true}
					/>
				</div>
			{/if}
		</section>

		<!-- AI Recommendations -->
		{#if recommendations.length > 0}
			<section class="recommendations-section mb-8" in:fly={{ y: 50, duration: 600, delay: 600 }}>
				<h2 class="text-xl font-semibold text-yorha-primary mb-4">
					AI Recommendations 
					<span class="text-sm text-yorha-text-muted">({recommendations.length} suggestions)</span>
				</h2>
				
				<div class="grid gap-4">
					{#each recommendations as rec, i}
						<div 
							class="ai-recommendation enhanced-feature"
							in:fly={{ x: -50, duration: 400, delay: i * 100 }}
						>
							<div class="flex items-start justify-between mb-2">
								<div class="flex items-center gap-2">
									<span class="recommendation-type text-xs px-2 py-1 bg-yorha-primary text-yorha-bg-primary rounded uppercase">
										{rec.type}
									</span>
									<div class={getRiskLevelClass(rec.riskLevel)}>
										{rec.riskLevel}
									</div>
								</div>
								
								<div class="confidence-section text-right">
									<div class="text-sm text-yorha-text-muted">
										{Math.round(rec.confidence * 100)}% confidence
									</div>
									<div class="confidence-indicator">
										<div 
											class="confidence-bar"
											style="width: {getConfidenceWidth(rec.confidence)}"
										></div>
									</div>
								</div>
							</div>
							
							<div class="recommendation-content mb-2">
								<p class="text-yorha-text-primary font-medium">{rec.content}</p>
								<p class="text-sm text-yorha-text-muted mt-1">{rec.reasoning}</p>
							</div>
							
							{#if rec.estimatedTime}
								<div class="recommendation-meta text-xs text-yorha-text-muted">
									⏱️ Estimated time: {rec.estimatedTime}
								</div>
							{/if}
						</div>
					{/each}
				</div>
			</section>
		{/if}

		<!-- Context7 Best Practices -->
		{#if bestPractices.length > 0}
			<section class="best-practices-section mb-8" in:fly={{ y: 50, duration: 600, delay: 800 }}>
				<h2 class="text-xl font-semibold text-yorha-primary mb-4">
					Context7 Best Practices 
					<span class="mcp-connection-indicator">
						MCP Connected
					</span>
				</h2>
				
				<div class="grid gap-4">
					{#each bestPractices as practice, i}
						<div 
							class="context7-enhancement performance-optimized"
							in:fly={{ y: 30, duration: 400, delay: i * 150 }}
						>
							<div class="flex items-start justify-between mb-2">
								<h3 class="font-semibold text-yorha-text-primary">{practice.title}</h3>
								<div class={getRiskLevelClass(practice.priority)}>
									{practice.priority}
								</div>
							</div>
							
							<p class="text-yorha-text-secondary mb-2">{practice.description}</p>
							<p class="text-sm text-yorha-text-muted mb-3">{practice.implementation}</p>
							
							{#if practice.codeExample}
								<details class="code-example">
									<summary class="text-sm text-yorha-accent cursor-pointer hover:text-yorha-primary">
										View Code Example
									</summary>
									<pre class="text-xs bg-yorha-bg-primary p-3 rounded mt-2 overflow-x-auto border border-yorha-border">
										<code class="text-yorha-text-primary">{practice.codeExample}</code>
									</pre>
								</details>
							{/if}
							
							<div class="practice-meta flex items-center gap-4 mt-3 text-xs text-yorha-text-muted">
								<span>⏱️ {practice.estimatedEffort}</span>
								{#if practice.dependencies.length > 0}
									<span>📦 {practice.dependencies.join(', ')}</span>
								{/if}
								{#if practice.legalSpecific}
									<span class="legal-accent">⚖️ Legal-specific</span>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			</section>
		{/if}

		<!-- Demo Controls -->
		<section class="demo-controls" in:fade={{ duration: 600, delay: 1000 }}>
			<h2 class="text-xl font-semibold text-yorha-primary mb-4">Demo Controls</h2>
			
			<div class="flex flex-wrap gap-4">
				<button 
					class="bg-yorha-warning text-yorha-bg-primary px-4 py-2 rounded border border-yorha-warning hover:opacity-80 transition-opacity focus-ring-enhanced"
					on:onclick={() => { recommendations = []; showTypewriter = false; }}
				>
					Clear Results
				</button>
				
				<button 
					class="bg-yorha-secondary text-yorha-bg-primary px-4 py-2 rounded border border-yorha-secondary hover:opacity-80 transition-opacity focus-ring-enhanced"
					on:onclick={loadBestPractices}
				>
					Refresh Best Practices
				</button>
				
				<button 
					class="bg-yorha-error text-white px-4 py-2 rounded border border-yorha-error hover:opacity-80 transition-opacity focus-ring-enhanced"
					on:onclick={() => advancedCache.clearRecommendations()}
				>
					Clear Cache
				</button>
			</div>
		</section>
	</div>
</div>

<style>
	/* Import advanced interactions first */
	@import '../../../lib/styles/advanced-interactions.css';
	
	.advanced-cache-demo {
		font-family: 'Inter', system-ui, sans-serif;
	}
	
	.cache-stat-card {
		transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
	}
	
	.cache-stat-card:hover {
		transform: translateY(-2px);
		border-color: rgba(0, 255, 0, 0.4);
	}
	
	.sample-query-btn:hover {
		transform: translateX(2px);
	}
	
	.code-example summary:hover {
		text-decoration: underline;
	}
	
	.recommendation-type {
		font-family: 'Monaco', 'Menlo', monospace;
	}
</style>

