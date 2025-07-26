<script lang="ts">
	import { onMount } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';
	import AIButton from '$components/ai/AIButton.svelte';
	import AIChatInterface from '$components/ai/AIChatInterface.svelte';
	
	// State
	let chatVisible = false;
	let chatMinimized = false;
	let systemStats = {
		services: {
			gemma3: 'checking',
			postgres: 'checking', 
			qdrant: 'checking',
			redis: 'checking'
		},
		model: 'gemma3-unsloth-legal:latest',
		parameters: '7.3GB Unsloth-trained'
	};
	
	// Sample queries for demo
	const sampleQueries = [
		{
			title: 'Contract Analysis',
			description: 'Review software license agreements',
			query: 'What are the key liability clauses I should review in a software license agreement? Please provide a detailed analysis.',
			icon: '📄'
		},
		{
			title: 'Risk Assessment', 
			description: 'Identify potential legal risks',
			query: 'Help me understand indemnification clauses and their potential risks. What should I watch out for?',
			icon: '⚖️'
		},
		{
			title: 'Compliance Check',
			description: 'Ensure regulatory compliance',
			query: 'What compliance requirements should I consider for a SaaS agreement? Include GDPR and data protection.',
			icon: '✅'
		},
		{
			title: 'Legal Consultation',
			description: 'Get expert legal guidance',
			query: 'Explain termination clauses and notice requirements in contracts. What are best practices?',
			icon: '💬'
		}
	];
	
	// Feature cards
	const features = [
		{
			title: 'Advanced RAG',
			description: 'Semantic search across legal documents with vector embeddings',
			details: 'PostgreSQL + pgvector + Qdrant for high-performance document retrieval',
			icon: '🔍'
		},
		{
			title: 'Custom AI Model',
			description: 'Unsloth-trained Gemma3 specialized for legal analysis',
			details: '7.3GB model optimized for contract review and legal reasoning',
			icon: '🤖'
		},
		{
			title: 'Real-time Processing',
			description: 'Live document analysis and instant AI responses',
			details: 'Event streaming with Redis and real-time UI updates',
			icon: '⚡'
		},
		{
			title: 'Professional UI',
			description: 'YoRHa-inspired design with advanced interactions',
			details: 'Draggable chat, streaming responses, and immersive effects',
			icon: '🎨'
		}
	];
	
	// Check system health on mount
	onMount(async () => {
		await checkSystemHealth();
		
		// Periodic health checks
		setInterval(checkSystemHealth, 30000);
	});
	
	async function checkSystemHealth() {
		const endpoints = [
			{ key: 'gemma3', url: 'http://localhost:11434/api/version' },
			{ key: 'qdrant', url: 'http://localhost:6333' },
			{ key: 'postgres', url: 'http://localhost:9000/health' },
			{ key: 'redis', url: 'http://localhost:9000/health' }
		];
		
		for (const endpoint of endpoints) {
			try {
				const response = await fetch(endpoint.url, { 
					method: 'GET',
					signal: AbortSignal.timeout(5000)
				});
				
				systemStats.services[endpoint.key] = response.ok ? 'online' : 'offline';
			} catch (error) {
				systemStats.services[endpoint.key] = 'offline';
			}
		}
		
		// Trigger reactivity
		systemStats = { ...systemStats };
	}
	
	function openChat() {
		chatVisible = true;
		chatMinimized = false;
	}
	
	function closeChat() {
		chatVisible = false;
	}
	
	function minimizeChat() {
		chatMinimized = true;
	}
	
	function maximizeChat() {
		chatMinimized = false;
	}
	
	function handleSampleQuery(query: string) {
		if (!chatVisible) {
			openChat();
		}
		// Send query to chat interface
		setTimeout(() => {
			// This would be handled by the chat component
			console.log('Sample query:', query);
		}, 300);
	}
</script>

<svelte:head>
	<title>YoRHa Legal AI - Advanced Contract Analysis</title>
	<meta name="description" content="Professional legal AI assistant powered by Gemma3 with advanced RAG capabilities">
</svelte:head>

<!-- Background Effects -->
<div class="fixed inset-0 bg-yorha-bg-primary overflow-hidden pointer-events-none">
	<!-- Floating particles -->
	{#each Array(8) as _, i}
		<div 
			class="absolute w-1 h-1 bg-yorha-accent rounded-full opacity-30 animate-float"
			style="
				left: {Math.random() * 100}%;
				top: {Math.random() * 100}%;
				animation-delay: {i * 0.5}s;
				animation-duration: {8 + (i * 2)}s;
			"
		></div>
	{/each}
	
	<!-- Scan lines -->
	<div class="absolute inset-0 bg-gradient-to-b from-transparent via-yorha-primary/5 to-transparent animate-scan-line"></div>
</div>

<!-- Main Content -->
<main class="relative z-10 min-h-screen">
	<!-- Header -->
	<header class="border-b-2 border-yorha-primary bg-gradient-to-r from-yorha-bg-secondary to-yorha-bg-tertiary">
		<div class="max-w-7xl mx-auto px-6 py-8">
			<div class="flex items-center justify-between">
				<!-- Logo -->
				<div class="flex items-center space-x-4">
					<div class="w-12 h-12 bg-gradient-to-br from-yorha-primary to-yorha-secondary flex items-center justify-center">
						<svg class="w-6 h-6 text-yorha-bg-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
						</svg>
					</div>
					<div>
						<h1 class="text-2xl font-bold font-mono bg-gradient-to-r from-yorha-primary to-yorha-accent bg-clip-text text-transparent">
							YoRHa Legal AI
						</h1>
						<p class="text-yorha-text-secondary text-sm">Advanced Contract Analysis System</p>
					</div>
				</div>
				
				<!-- System Status -->
				<div class="text-right">
					<div class="flex items-center space-x-2 mb-2">
						<div class="w-2 h-2 bg-yorha-success rounded-full animate-pulse"></div>
						<span class="text-yorha-success text-sm font-mono">SYSTEM ONLINE</span>
					</div>
					<div class="text-xs text-yorha-text-muted space-y-1">
						<div>Model: {systemStats.model}</div>
						<div>Size: {systemStats.parameters}</div>
					</div>
				</div>
			</div>
		</div>
	</header>
	
	<!-- Hero Section -->
	<section class="py-16 px-6">
		<div class="max-w-7xl mx-auto text-center">
			<h2 class="text-4xl font-bold font-mono text-yorha-text-primary mb-6">
				Professional Legal AI Assistant
			</h2>
			<p class="text-xl text-yorha-text-secondary mb-8 max-w-3xl mx-auto">
				Experience advanced legal document analysis powered by your custom Unsloth-trained Gemma3 model. 
				Phase 3 RAG capabilities with real-time AI assistance for contract review and legal research.
			</p>
			
			<button 
				type="button"
				on:click={openChat}
				class="bg-gradient-to-r from-yorha-primary to-yorha-secondary text-yorha-bg-primary font-mono font-semibold px-8 py-4 border-2 border-yorha-primary hover:scale-105 transition-transform duration-300 shadow-lg hover:shadow-yorha-primary/20"
			>
				Launch Legal AI Assistant
			</button>
		</div>
	</section>
	
	<!-- Features Grid -->
	<section class="py-16 px-6 bg-yorha-bg-secondary/30">
		<div class="max-w-7xl mx-auto">
			<h3 class="text-2xl font-bold font-mono text-yorha-primary text-center mb-12">
				System Capabilities
			</h3>
			
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
				{#each features as feature, i}
					<div 
						class="bg-yorha-bg-secondary border border-yorha-border p-6 hover:border-yorha-primary transition-all duration-300 hover:shadow-lg hover:shadow-yorha-primary/10 hover:-translate-y-1 group cursor-pointer"
						in:fly={{ y: 50, duration: 600, delay: i * 100, easing: quintOut }}
					>
						<div class="text-3xl mb-4">{feature.icon}</div>
						<h4 class="text-lg font-semibold text-yorha-primary mb-2 group-hover:text-yorha-accent transition-colors">
							{feature.title}
						</h4>
						<p class="text-yorha-text-secondary text-sm mb-3">
							{feature.description}
						</p>
						<p class="text-yorha-text-muted text-xs">
							{feature.details}
						</p>
					</div>
				{/each}
			</div>
		</div>
	</section>
	
	<!-- Interactive Demo Section -->
	<section class="py-16 px-6">
		<div class="max-w-7xl mx-auto">
			<div class="text-center mb-12">
				<h3 class="text-2xl font-bold font-mono text-yorha-primary mb-4">
					Try Sample Legal Queries
				</h3>
				<p class="text-yorha-text-secondary">
					Click any query below to test the AI assistant, or use the chat button to ask your own questions
				</p>
			</div>
			
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each sampleQueries as query, i}
					<button
						type="button"
						on:click={() => handleSampleQuery(query.query)}
						class="text-left bg-yorha-bg-secondary border border-yorha-border p-6 hover:border-yorha-primary transition-all duration-300 hover:shadow-lg hover:shadow-yorha-primary/10 group"
						in:fly={{ x: i % 2 === 0 ? -50 : 50, duration: 600, delay: i * 150, easing: quintOut }}
					>
						<div class="flex items-start space-x-4">
							<div class="text-2xl mt-1">{query.icon}</div>
							<div class="flex-1">
								<h4 class="text-lg font-semibold text-yorha-primary mb-2 group-hover:text-yorha-accent transition-colors">
									{query.title}
								</h4>
								<p class="text-yorha-text-secondary text-sm mb-3">
									{query.description}
								</p>
								<p class="text-yorha-text-muted text-xs italic">
									"{query.query.substring(0, 80)}..."
								</p>
							</div>
							<div class="text-yorha-border group-hover:text-yorha-primary transition-colors">
								<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
								</svg>
							</div>
						</div>
					</button>
				{/each}
			</div>
		</div>
	</section>
	
	<!-- System Status Section -->
	<section class="py-16 px-6 bg-yorha-bg-secondary/30">
		<div class="max-w-4xl mx-auto">
			<h3 class="text-2xl font-bold font-mono text-yorha-primary text-center mb-12">
				System Status
			</h3>
			
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
				{#each Object.entries(systemStats.services) as [service, status]}
					<div class="bg-yorha-bg-secondary border border-yorha-border p-4 text-center">
						<div class="flex items-center justify-center mb-2">
							<div class="w-3 h-3 rounded-full mr-2 {
								status === 'online' ? 'bg-yorha-success animate-pulse' :
								status === 'offline' ? 'bg-yorha-error' :
								'bg-yorha-warning animate-pulse'
							}"></div>
							<span class="text-sm font-mono text-yorha-text-primary capitalize">
								{service}
							</span>
						</div>
						<div class="text-xs text-yorha-text-muted capitalize">
							{status}
						</div>
					</div>
				{/each}
			</div>
			
			<div class="mt-8 text-center">
				<div class="inline-flex items-center space-x-4 bg-yorha-bg-secondary border border-yorha-border px-6 py-3">
					<div class="text-sm font-mono text-yorha-text-secondary">
						Powered by: <span class="text-yorha-primary">{systemStats.model}</span>
					</div>
					<div class="w-px h-6 bg-yorha-border"></div>
					<div class="text-sm font-mono text-yorha-text-secondary">
						Size: <span class="text-yorha-accent">{systemStats.parameters}</span>
					</div>
				</div>
			</div>
		</div>
	</section>
	
	<!-- Footer -->
	<footer class="border-t-2 border-yorha-primary bg-yorha-bg-secondary py-8 px-6">
		<div class="max-w-7xl mx-auto text-center">
			<div class="flex flex-wrap justify-center items-center space-x-6 text-sm text-yorha-text-muted">
				<span>Phase 3+4 Legal AI System</span>
				<span>•</span>
				<span>Advanced RAG + Event Streaming</span>
				<span>•</span>
				<span>Unsloth-trained Gemma3</span>
				<span>•</span>
				<span>YoRHa UI Design</span>
			</div>
		</div>
	</footer>
</main>

<!-- AI Chat Components -->
<AIButton 
	on:click={openChat}
	tooltip="Legal AI Assistant (Ctrl+K)"
	notification={false}
	position="bottom-right"
	size="lg"
/>

{#if chatVisible}
	<AIChatInterface 
		bind:visible={chatVisible}
		bind:minimized={chatMinimized}
		modelName="gemma3-unsloth-legal:latest"
		title="YoRHa Legal AI"
		subtitle="Unsloth Gemma3"
		on:close={closeChat}
		on:minimize={minimizeChat}
		on:maximize={maximizeChat}
	/>
{/if}
