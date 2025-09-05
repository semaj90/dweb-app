

	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import AIChat from '$lib/components/AIChat.svelte';
	import { Card } from '$lib/components/ui';
	import { Button } from '$lib/components/ui';
	import { Badge } from '$lib/components/ui';

	// System status stores
	const systemHealth = writable({ status: 'checking', services: {} });
	const availableModels = writable([]);
	const testResults = writable({});

	// Test configuration
	let selectedCaseId = 'CASE-2024-001';
	let testSessionId = null;

	// Health check configuration
	const services = {
		ollama: { url: 'http://localhost:11434/api/tags', name: 'Ollama' },
		database: { url: '/api/chat?action=health', name: 'Database' },
		embedding: { url: '/api/chat?action=health', name: 'Embeddings' },
		rag: { url: '/api/chat?action=health', name: 'RAG System' }
	};

	onMount(async () => {
		await runHealthChecks();
		await loadAvailableModels();

		// Set up periodic health checks
		const healthInterval = setInterval(runHealthChecks, 30000);

		return () => clearInterval(healthInterval);
	});

	/**
	 * Run comprehensive health checks
	 */
	async function runHealthChecks() {
		const results = {};

		for (const [key, service] of Object.entries(services)) {
			try {
				const response = await fetch(service.url);
				const data = await response.json();

				results[key] = {
					status: response.ok ? 'healthy' : 'error',
					name: service.name,
					details: data,
					timestamp: new Date().toISOString()
				};
			} catch (error) {
				results[key] = {
					status: 'error',
					name: service.name,
					error: error.message,
					timestamp: new Date().toISOString()
				};
			}
		}

		const overallStatus = Object.values(results).every(r => r.status === 'healthy')
			? 'healthy'
			: 'partial';

		systemHealth.set({
			status: overallStatus,
			services: results,
			lastCheck: new Date().toISOString()
		});
	}

	/**
	 * Load available models from Ollama
	 */
	async function loadAvailableModels() {
		try {
			const response = await fetch('/api/chat?action=models');
			const data = await response.json();
			availableModels.set(data.models || []);
		} catch (error) {
			console.error('Error loading models:', error);
			availableModels.set([]);
		}
	}

	/**
	 * Run RAG system tests
	 */
	async function runRAGTests() {
		const tests = [
			{
				name: 'Basic Legal Query',
				query: 'What are the elements of embezzlement?',
				expected: 'legal knowledge'
			},
			{
				name: 'Case-Specific Query',
				query: 'What evidence do we have in the Anderson case?',
				expected: 'case documents'
			},
			{
				name: 'Procedural Question',
				query: 'How should I prepare witnesses for testimony?',
				expected: 'procedure guidance'
			},
			{
				name: 'Document Authentication',
				query: 'What are the requirements for authenticating digital evidence?',
				expected: 'authentication standards'
			}
		];

		const results = {};

		for (const test of tests) {
			try {
				const startTime = Date.now();

				const response = await fetch('/api/chat', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						message: test.query,
						caseId: selectedCaseId,
						options: { stream: false }
					})
				});

				const data = await response.json();
				const responseTime = Date.now() - startTime;

				results[test.name] = {
					status: response.ok ? 'pass' : 'fail',
					query: test.query,
					response: data.response?.substring(0, 200) + '...',
					sources: data.sources?.length || 0,
					responseTime,
					tokensUsed: data.tokensUsed || 0
				};
			} catch (error) {
				results[test.name] = {
					status: 'error',
					query: test.query,
					error: error.message
				};
			}
		}

		testResults.set(results);
	}

	/**
	 * Test streaming functionality
	 */
	async function testStreaming() {
		// This will be handled by the AIChat component
		console.log('Streaming test initiated through chat interface');
	}

	/**
	 * Get status badge variant
	 */
	function getStatusVariant(status) {
		switch (status) {
			case 'healthy':
			case 'pass':
				return 'success';
			case 'partial':
				return 'warning';
			case 'error':
			case 'fail':
				return 'destructive';
			default:
				return 'secondary';
		}
	}

	/**
	 * Format timestamp for display
	 */
	function formatTime(timestamp) {
		return new Date(timestamp).toLocaleTimeString();
	}



	Legal AI Assistant - RAG System Test



	
		
		
			
				Legal AI Assistant - RAG System Test
			
			
				Comprehensive testing and validation of the RAG-powered legal AI system
			
		

		
		
			
				System Health
				
					
						{$systemHealth.status}
					
					
						Refresh
					
				
			

			
				{#each Object.entries($systemHealth.services) as [key, service]}
					
						
							{service.name}
							
								{service.status}
							
						

						{#if service.error}
							{service.error}
						{:else if service.details}
							
								{#if service.details.models}
									Models: {service.details.models.length}
								{/if}
								{#if service.details.rag}
									Document chunks: {service.details.rag.documentChunks}
									Knowledge entries: {service.details.rag.knowledgeBaseEntries}
								{/if}
								{#if service.details.embedding}
									Embedding model: {service.details.embedding.embeddingModel}
								{/if}
							
						{/if}

						
							Last check: {formatTime(service.timestamp)}
						
					
				{/each}
			
		

		
		{#if $availableModels.length > 0}
			
				Available Models
				
					{#each $availableModels as model}
						
							{model.name}
							
								Size: {(model.size / 1024 / 1024 / 1024).toFixed(2)} GB
								Modified: {new Date(model.modified_at).toLocaleDateString()}
								{#if model.details?.family}
									Family: {model.details.family}
								{/if}
							
						
					{/each}
				
			
		{/if}

		
		
			
				RAG System Tests
				
					Run Tests
				
			

			{#if Object.keys($testResults).length > 0}
				
					{#each Object.entries($testResults) as [testName, result]}
						
							
								{testName}
								
									{result.status}
								
							

							
								
									Query: {result.query}
								

								{#if result.response}
									
										Response: {result.response}
									
								{/if}

								{#if result.sources !== undefined}
									
										Sources: {result.sources} context sources used
									
								{/if}

								{#if result.responseTime}
									
										Performance:
										{result.responseTime}ms response time,
										{result.tokensUsed} tokens used
									
								{/if}

								{#if result.error}
									
										Error: {result.error}
									
								{/if}
							
						
					{/each}
				
			{:else}
				
					Click "Run Tests" to validate RAG system functionality
				
			{/if}
		

		
		
			Interactive Chat Test

			
				
					Test Case ID:
					<select
						bind:value={selectedCaseId}
						class="ml-2 bg-gray-800 border border-gray-600 rounded px-2 py-1"
					>
						CASE-2024-001 (Anderson Embezzlement)
						No specific case
					
				

				
					RAG Enabled: Context retrieval + Legal knowledge
				
			

			
				<AIChat
					caseId={selectedCaseId}
					sessionId={testSessionId}
					enableRAG={true}
					maxContextChunks={5}
					placeholder="Test RAG functionality: Ask about embezzlement elements, evidence authentication, witness preparation, or case-specific questions..."
				/>
			
		

		
		
			Quick Test Queries
			
				
					Legal Knowledge Tests:
					
						• "What are the elements of embezzlement?"
						• "How do I authenticate digital evidence?"
						• "What are Brady disclosure requirements?"
						• "Explain witness preparation guidelines"
					
				

				
					RAG Context Tests:
					
						• "Analyze the Anderson embezzlement case"
						• "What evidence do we have for financial fraud?"
						• "Timeline of the defendant's access to funds"
						• "Recommend prosecution strategy"
					
				
			
		

		
		
			Legal AI Assistant v2.0.0 - RAG-Powered System
			
				System Status:
				
					{$systemHealth.status}
				
				{#if $systemHealth.lastCheck}
					| Last Check: {formatTime($systemHealth.lastCheck)}
				{/if}
			
		
	



	:global(body) {
		background-color: #111827;
		font-family: 'Inter', sans-serif;
	}

