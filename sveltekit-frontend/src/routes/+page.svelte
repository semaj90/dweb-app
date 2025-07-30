<script lang="ts">
	import { onMount } from 'svelte';
	import { fade, fly } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';
	
	// State - Svelte 5 runes for proper reactivity
	let chatVisible = $state(false);
	let chatMinimized = $state(false);
	let systemStats = $state({
		services: {
			gemma3: 'checking',
			postgres: 'checking', 
			qdrant: 'checking',
			redis: 'checking'
		},
		model: 'gemma3-unsloth-legal:latest',
		parameters: '7.3GB Unsloth-trained'
	});
	
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
		console.log('🚀 Component mounted, starting health check...');
		await checkSystemHealth();
		
		// Periodic health checks
		console.log('⏰ Setting up periodic health checks every 30 seconds');
		setInterval(checkSystemHealth, 30000);
	});
	
	async function checkSystemHealth() {
		// Set checking state immediately for instant UI feedback
		systemStats.services.gemma3 = 'checking';
		systemStats.services.qdrant = 'checking';
		systemStats.services.postgres = 'checking';
		systemStats.services.redis = 'checking';
		
		try {
			const startTime = performance.now();
			const response = await fetch('/api/phase13/integration?action=services', { 
				method: 'GET',
				signal: AbortSignal.timeout(3000) // Reduced timeout for faster response
			});
			
			const responseTime = performance.now() - startTime;
			console.log(`📡 API response: ${response.status} (${responseTime.toFixed(1)}ms)`);
			
			if (response.ok) {
				const data = await response.json();
				const services = data.data.services;
				
				// Instant UI update with optimized state changes
				systemStats.services.gemma3 = services.ollama ? 'online' : 'offline';
				systemStats.services.qdrant = services.qdrant ? 'online' : 'offline';
				systemStats.services.postgres = services.database ? 'online' : 'offline';
				systemStats.services.redis = services.redis ? 'online' : 'offline';
				
				console.log(`✅ Health check complete (${responseTime.toFixed(1)}ms):`, systemStats.services);
			} else {
				// Immediate fallback with proper error handling
				systemStats.services.gemma3 = 'offline';
				systemStats.services.qdrant = 'offline';
				systemStats.services.postgres = 'offline';
				systemStats.services.redis = 'offline';
				console.warn('⚠️ Phase 13 API failed, services marked offline');
			}
		} catch (error) {
			// Instant error state update
			systemStats.services.gemma3 = 'offline';
			systemStats.services.qdrant = 'offline';
			systemStats.services.postgres = 'offline';
			systemStats.services.redis = 'offline';
			console.error('❌ Health check failed:', error.message);
		}
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
		// Trigger FindModal with pre-filled query
		window.dispatchEvent(new KeyboardEvent('keydown', {
			key: 'k',
			ctrlKey: true,
			bubbles: true
		}));
		
		// Set the query after a brief delay to ensure modal is open
		setTimeout(() => {
			const searchInput = document.querySelector('[data-testid="search-input"]') as HTMLInputElement;
			if (searchInput) {
				searchInput.value = query;
				searchInput.dispatchEvent(new Event('input', { bubbles: true }));
			}
		}, 300);
	}
</script>

<svelte:head>
	<title>YoRHa Legal AI - Advanced Contract Analysis</title>
	<meta name="description" content="Professional legal AI assistant powered by Gemma3 with advanced RAG capabilities">
</svelte:head>

<!-- Background Effects -->
<div class="fixed inset-0 bg-gray-50 overflow-hidden pointer-events-none">
	<!-- Floating particles -->
	{#each Array(8) as _, i}
		<div 
			class="absolute w-1 h-1 bg-blue-500 rounded-full opacity-30 animate-pulse"
			style="
				left: {Math.random() * 100}%;
				top: {Math.random() * 100}%;
				animation-delay: {i * 0.5}s;
				animation-duration: {8 + (i * 2)}s;
			"
		></div>
	{/each}
</div>

<!-- Main Content -->
<main class="relative z-10 min-h-screen">
	<!-- Header -->
	<header class="border-b-2 border-gray-300 bg-gradient-to-r from-gray-100 to-gray-200">
		<div class="max-w-7xl mx-auto px-6 py-8">
			<div class="flex items-center justify-between">
				<!-- Logo -->
				<div class="flex items-center space-x-4">
					<div class="w-12 h-12 bg-gradient-to-br from-gray-800 to-gray-600 flex items-center justify-center">
						<svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
						</svg>
					</div>
					<div>
						<h1 class="text-2xl font-bold font-mono bg-gradient-to-r from-gray-800 to-blue-600 bg-clip-text text-transparent">
							YoRHa Legal AI
						</h1>
						<p class="text-gray-600 text-sm">Advanced Contract Analysis System</p>
					</div>
				</div>
				
				<!-- System Status -->
				<div class="text-right">
					<div class="flex items-center space-x-2 mb-2">
						<div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
						<span class="text-green-500 text-sm font-mono">SYSTEM ONLINE</span>
					</div>
					<div class="text-xs text-gray-500 space-y-1">
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
			<h2 class="text-4xl font-bold font-mono text-gray-800 mb-6">
				Professional Legal AI Assistant
			</h2>
			<p class="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
				Experience advanced legal document analysis powered by your custom Unsloth-trained Gemma3 model. 
				Phase 3 RAG capabilities with real-time AI assistance for contract review and legal research.
			</p>
			
			<button 
				type="button"
				onclick={() => {
					// Trigger AI Search Modal instead of chat
					window.dispatchEvent(new KeyboardEvent('keydown', {
						key: 'k',
						ctrlKey: true,
						bubbles: true
					}));
				}}
				class="bg-gradient-to-r from-blue-600 to-blue-800 text-white font-mono font-semibold px-8 py-4 border-2 border-blue-600 hover:scale-105 transition-transform duration-300 shadow-lg hover:shadow-blue-500/20 rounded-lg"
			>
				🔍 Launch AI Search Assistant
			</button>
		</div>
	</section>
	
	<!-- Features Grid -->
	<section class="py-16 px-6 bg-gray-100">
		<div class="max-w-7xl mx-auto">
			<h3 class="text-2xl font-bold font-mono text-gray-800 text-center mb-12">
				System Capabilities
			</h3>
			
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
				{#each features as feature, i}
					<div 
						class="bg-white border border-gray-200 p-6 hover:border-blue-500 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-1 group cursor-pointer rounded-lg"
						in:fly={{ y: 50, duration: 600, delay: i * 100, easing: quintOut }}
					>
						<div class="text-3xl mb-4">{feature.icon}</div>
						<h4 class="text-lg font-semibold text-gray-800 mb-2 group-hover:text-blue-600 transition-colors">
							{feature.title}
						</h4>
						<p class="text-gray-600 text-sm mb-3">
							{feature.description}
						</p>
						<p class="text-gray-500 text-xs">
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
				<h3 class="text-2xl font-bold font-mono text-gray-800 mb-4">
					Try Sample Legal Queries
				</h3>
				<p class="text-gray-600">
					Click any query below to test the AI assistant, or use the chat button to ask your own questions
				</p>
			</div>
			
			<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
				{#each sampleQueries as query, i}
					<button
						type="button"
						on:click={() => handleSampleQuery(query.query)}
						class="text-left bg-white border border-gray-200 p-6 hover:border-blue-500 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10 group rounded-lg"
						in:fly={{ x: i % 2 === 0 ? -50 : 50, duration: 600, delay: i * 150, easing: quintOut }}
					>
						<div class="flex items-start space-x-4">
							<div class="text-2xl mt-1">{query.icon}</div>
							<div class="flex-1">
								<h4 class="text-lg font-semibold text-gray-800 mb-2 group-hover:text-blue-600 transition-colors">
									{query.title}
								</h4>
								<p class="text-gray-600 text-sm mb-3">
									{query.description}
								</p>
								<p class="text-gray-500 text-xs italic">
									"{query.query.substring(0, 80)}..."
								</p>
							</div>
							<div class="text-gray-400 group-hover:text-blue-500 transition-colors">
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
	<section class="py-16 px-6 bg-gray-100">
		<div class="max-w-4xl mx-auto">
			<h3 class="text-2xl font-bold font-mono text-gray-800 text-center mb-12">
				System Status
			</h3>
			
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
				{#each Object.entries(systemStats.services) as [service, status]}
					<div class="bg-white border border-gray-200 p-4 text-center rounded-lg">
						<div class="flex items-center justify-center mb-2">
							<div class="w-3 h-3 rounded-full mr-2 {
								status === 'online' ? 'bg-green-500 animate-pulse' :
								status === 'offline' ? 'bg-red-500' :
								'bg-yellow-500 animate-pulse'
							}"></div>
							<span class="text-sm font-mono text-gray-800 capitalize">
								{service}
							</span>
						</div>
						<div class="text-xs text-gray-500 capitalize">
							{status}
						</div>
					</div>
				{/each}
			</div>
			
			<div class="mt-8 text-center">
				<div class="inline-flex items-center space-x-4 bg-white border border-gray-200 px-6 py-3 rounded-lg">
					<div class="text-sm font-mono text-gray-600">
						Powered by: <span class="text-gray-800">{systemStats.model}</span>
					</div>
					<div class="w-px h-6 bg-gray-300"></div>
					<div class="text-sm font-mono text-gray-600">
						Size: <span class="text-blue-600">{systemStats.parameters}</span>
					</div>
				</div>
			</div>
		</div>
	</section>
	
	<!-- Footer -->
	<footer class="border-t-2 border-gray-300 bg-gray-200 py-8 px-6">
		<div class="max-w-7xl mx-auto text-center">
			<div class="flex flex-wrap justify-center items-center space-x-6 text-sm text-gray-600">
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

<!-- Chat placeholder -->
{#if chatVisible}
	<div class="fixed bottom-4 right-4 w-96 h-96 bg-white border border-gray-300 rounded-lg shadow-lg p-4">
		<div class="flex justify-between items-center mb-4">
			<h3 class="font-semibold">Legal AI Chat</h3>
			<button 
				on:click={closeChat}
				class="text-gray-500 hover:text-gray-800"
			>
				✕
			</button>
		</div>
		<div class="text-gray-600 text-sm">
			AI Chat interface will be implemented here once component issues are resolved.
		</div>
	</div>
{/if}