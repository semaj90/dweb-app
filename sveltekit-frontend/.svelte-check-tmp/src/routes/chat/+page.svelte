<!-- Enhanced Chat Test Page -->
<script lang="ts">
  import EnhancedChat from '$lib/components/EnhancedChat.svelte';
	import { onMount } from 'svelte';

	let connectionStatus = $state('testing');
	let services = $state({
		ollama: false,
		postgresql: false,
		redis: false,
		minio: false
	});

	async function checkServices() {
		connectionStatus = 'testing';
		
		// Test Ollama
		try {
			const response = await fetch('http://localhost:11434/api/version');
			services.ollama = response.ok;
		} catch {
			services.ollama = false;
		}

		// Test PostgreSQL via our API
		try {
			const response = await fetch('/api/health/database');
			services.postgresql = response.ok;
		} catch {
			services.postgresql = false;
		}

		// Test Redis (if accessible)
		try {
			const response = await fetch('/api/health/redis');
			services.redis = response.ok;
		} catch {
			services.redis = false;
		}

		// Test MinIO (if accessible)
		try {
			const response = await fetch('http://localhost:9000/minio/health/live');
			services.minio = response.ok;
		} catch {
			services.minio = false;
		}

		connectionStatus = 'complete';
	}

	onMount(() => {
		checkServices();
	});
</script>

<svelte:head>
	<title>Legal AI Chat - Enhanced Interface</title>
	<meta name="description" content="Enhanced Legal AI Chat with vector storage and modern UI components" />
</svelte:head>

<div class="chat-page min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
	<div class="container mx-auto px-4 py-8">
		<!-- Page Header -->
		<div class="text-center mb-8">
			<h1 class="text-4xl font-bold text-gray-900 mb-4">Legal AI Assistant</h1>
			<p class="text-lg text-gray-600 max-w-2xl mx-auto">
				Enhanced chat interface with vector storage, document integration, and multi-model support.
				Ask questions about legal documents, contracts, or general legal matters.
			</p>
		</div>

		<!-- Service Status -->
		<div class="service-status mb-6 p-4 bg-white/70 rounded-lg backdrop-blur-sm">
			<h3 class="text-sm font-semibold text-gray-700 mb-3">Service Status</h3>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
				<div class="service-item flex items-center space-x-2">
					<div class={`w-3 h-3 rounded-full ${services.ollama ? 'bg-green-500' : 'bg-red-500'}`}></div>
					<span class="text-sm text-gray-700">Ollama AI</span>
				</div>
				<div class="service-item flex items-center space-x-2">
					<div class={`w-3 h-3 rounded-full ${services.postgresql ? 'bg-green-500' : 'bg-red-500'}`}></div>
					<span class="text-sm text-gray-700">PostgreSQL</span>
				</div>
				<div class="service-item flex items-center space-x-2">
					<div class={`w-3 h-3 rounded-full ${services.redis ? 'bg-green-500' : 'bg-red-500'}`}></div>
					<span class="text-sm text-gray-700">Redis</span>
				</div>
				<div class="service-item flex items-center space-x-2">
					<div class={`w-3 h-3 rounded-full ${services.minio ? 'bg-green-500' : 'bg-red-500'}`}></div>
					<span class="text-sm text-gray-700">MinIO</span>
				</div>
			</div>
			{#if connectionStatus === 'testing'}
				<div class="mt-2 text-xs text-gray-500">Testing connections...</div>
			{/if}
		</div>

		<!-- Chat Interface -->
		<div class="chat-interface bg-white/80 rounded-2xl shadow-xl backdrop-blur-sm p-2">
			<EnhancedChat />
		</div>

		<!-- Features Info -->
		<div class="features-info mt-8 grid md:grid-cols-3 gap-6">
			<div class="feature-card p-6 bg-white/70 rounded-lg backdrop-blur-sm">
				<div class="feature-icon w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
					<svg class="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
					</svg>
				</div>
				<h3 class="text-lg font-semibold text-gray-900 mb-2">Document Analysis</h3>
				<p class="text-gray-600 text-sm">Upload and analyze legal documents with AI-powered insights and vector similarity search.</p>
			</div>

			<div class="feature-card p-6 bg-white/70 rounded-lg backdrop-blur-sm">
				<div class="feature-icon w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
					<svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
					</svg>
				</div>
				<h3 class="text-lg font-semibold text-gray-900 mb-2">GPU Acceleration</h3>
				<p class="text-gray-600 text-sm">Powered by RTX 3060 Ti for fast AI inference and real-time response generation.</p>
			</div>

			<div class="feature-card p-6 bg-white/70 rounded-lg backdrop-blur-sm">
				<div class="feature-icon w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
					<svg class="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"></path>
					</svg>
				</div>
				<h3 class="text-lg font-semibold text-gray-900 mb-2">Vector Storage</h3>
				<p class="text-gray-600 text-sm">Chat history and documents stored with pgvector for semantic search and context retrieval.</p>
			</div>
		</div>
	</div>
</div>

<style>
	.chat-page {
		min-height: 100vh;
	}
	
	.backdrop-blur-sm {
		backdrop-filter: blur(4px);
	}
	
	.feature-card {
		transition: transform 0.2s ease-in-out;
	}
	
	.feature-card:hover {
		transform: translateY(-2px);
	}
</style>
