<!-- LLM Provider Selector with Bits UI v2 and Real-time Status -->
<script lang="ts">
  interface Props {
    selectedProvider: LLMProvider | null ;
    disabled?: any;
  }
  let {
    selectedProvider = null,
    disabled = false
  } = $props();



	
	import { Badge } from 'bits-ui';
	import { Card, CardContent } from 'bits-ui';
	import { writable, derived, type Writable } from 'svelte/store';
	import { createEventDispatcher, onMount } from 'svelte';
	import { fade, fly } from 'svelte/transition';

	 // LLMProvider | null = null;
	let { disabled = $bindable() } = $props(); // false;

	const dispatch = createEventDispatcher<{
		providerSelected: { provider: LLMProvider };
		statusChanged: { provider: LLMProvider status: LLMStatus };
	}>();

	// LLM Provider Types based on your architecture
	interface LLMProvider {
		id: string
		name: string
		type: 'ollama' | 'vllm' | 'autogen' | 'crewai';
		endpoint: string
		models: LLMModel[];
		capabilities: string[];
		status: LLMStatus
		performance?: PerformanceMetrics;
	}

	interface LLMModel {
		id: string
		name: string
		size: string
		specialization: 'general' | 'legal' | 'code' | 'reasoning';
		performance: PerformanceMetrics
	}

	interface PerformanceMetrics {
		avgResponseTime: number
		tokensPerSecond: number
		memoryUsage: string
		uptime: number
	}

	type LLMStatus = 'online' | 'offline' | 'busy' | 'loading';

	// Mock providers - replace with real API calls
	const providers: Writable<LLMProvider[]> = writable([
		{
			id: 'ollama-local',
			name: 'Ollama (Local)',
			type: 'ollama',
			endpoint: 'http://localhost:11434',
			status: 'online',
			capabilities: ['text-generation', 'embeddings', 'chat'],
			models: [
				{
					id: 'gemma3-legal',
					name: 'Gemma3 Legal',
					size: '7.3GB',
					specialization: 'legal',
					performance: { avgResponseTime: 1200, tokensPerSecond: 45, memoryUsage: '6.2GB', uptime: 99.2 }
				},
				{
					id: 'nomic-embed-text',
					name: 'Nomic Embed',
					size: '274MB',
					specialization: 'general',
					performance: { avgResponseTime: 150, tokensPerSecond: 200, memoryUsage: '512MB', uptime: 99.8 }
				}
			]
		},
		{
			id: 'vllm-server',
			name: 'vLLM Server',
			type: 'vllm',
			endpoint: 'http://localhost:8000',
			status: 'offline',
			capabilities: ['high-throughput', 'batch-processing', 'streaming'],
			models: []
		},
		{
			id: 'autogen-framework',
			name: 'AutoGen Agents',
			type: 'autogen',
			endpoint: 'http://localhost:8001',
			status: 'loading',
			capabilities: ['multi-agent', 'conversation', 'code-execution'],
			models: []
		},
		{
			id: 'crewai-team',
			name: 'CrewAI Teams',
			type: 'crewai',
			endpoint: 'http://localhost:8002',
			status: 'offline',
			capabilities: ['role-based', 'collaborative', 'workflow'],
			models: []
		}
	]);

	// Real-time status checking
	let statusCheckInterval: number

	const checkProviderStatus = async (provider: LLMProvider): Promise<LLMStatus> => {
		try {
			const response = await fetch(`${provider.endpoint}/health`, {
				method: 'GET',
				timeout: 5000
			});
			return response.ok ? 'online' : 'offline';
		} catch {
			return 'offline';
		}
	};

	const updateProviderStatuses = async () => {
		const currentProviders = $providers;
		for (let i = 0; i < currentProviders.length; i++) {
			const newStatus = await checkProviderStatus(currentProviders[i]);
			if (currentProviders[i].status !== newStatus) {
				currentProviders[i].status = newStatus;
				dispatch('statusChanged', { provider: currentProviders[i], status: newStatus });
			}
		}
		providers.set(currentProviders);
	};

	onMount(() => {
		// Initial status check
		updateProviderStatuses();
		
		// Periodic status updates every 10 seconds
		statusCheckInterval = setInterval(updateProviderStatuses, 10000);
		
		return () => {
			clearInterval(statusCheckInterval);
		};
	});

	// Melt UI Select setup
	const {
		elements: { trigger, menu, option, group, groupLabel, label },
		states: { selectedLabel, open, selected },
		helpers: { isSelected }
	} = createSelect<LLMProvider>({
		forceVisible: true,
		positioning: {
			placement: 'bottom',
			fitViewport: true,
		}
	});

	// Reactive selection handling
	$effect(() => { if ($selected && $selected.value !== selectedProvider) {
		selectedProvider = $selected.value;
		dispatch('providerSelected', { provider: selectedProvider });
	}

	// Status badge styling
	const getStatusColor = (status: LLMStatus) => {
		switch (status) {
			case 'online': return 'bg-yorha-success text-yorha-bg-primary';
			case 'offline': return 'bg-yorha-danger text-yorha-bg-primary';
			case 'busy': return 'bg-yorha-warning text-yorha-bg-primary';
			case 'loading': return 'bg-yorha-accent text-yorha-bg-primary animate-pulse';
			default: return 'bg-yorha-text-secondary text-yorha-bg-primary';
		}
	};

	const getTypeIcon = (type: string) => {
		switch (type) {
			case 'ollama': return '🦙';
			case 'vllm': return '⚡';
			case 'autogen': return '🤖';
			case 'crewai': return '👥';
			default: return '🔧';
		}
	};
</script>

<div class="llm-provider-selector">
	<!-- Label -->
	<label 
		use:melt={$label}
		class="block text-sm font-medium text-yorha-text-primary mb-2"
	>
		LLM Provider
	</label>

	<!-- Select Trigger -->
	<button
		use:melt={$trigger}
		class="flex h-10 w-full items-center justify-between rounded-md border border-yorha-border bg-yorha-bg-secondary px-3 py-2 text-sm placeholder:text-yorha-text-tertiary focus:outline-none focus:ring-2 focus:ring-yorha-primary focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200"
		class:opacity-50={disabled}
		{disabled}
		aria-label="Select LLM Provider"
	>
		<span class="truncate">
			{#if selectedProvider}
				<span class="flex items-center space-x-2">
					<span class="text-lg" role="img">{getTypeIcon(selectedProvider.type)}</span>
					<span>{selectedProvider.name}</span>
					<Badge class={`text-xs ${getStatusColor(selectedProvider.status)}`}>
						{selectedProvider.status.toUpperCase()}
					</Badge>
				</span>
			{:else}
				<span class="text-yorha-text-tertiary">Select a provider...</span>
			{/if}
		</span>
		
		<!-- Dropdown Arrow -->
		<svg 
			class="h-4 w-4 text-yorha-text-secondary transition-transform duration-200"
			class:rotate-180={$open}
			fill="none" 
			stroke="currentColor" 
			viewBox="0 0 24 24"
			aria-hidden="true"
		>
			<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
		</svg>
	</button>

	<!-- Dropdown Menu -->
	{#if $open}
		<div
			use:melt={$menu}
			class="z-50 min-w-[320px] rounded-md border border-yorha-border bg-yorha-bg-primary p-1 shadow-lg focus:outline-none"
			transitionfly={{ y: -5, duration: 150 }}
		>
			{#each $providers as provider (provider.id)}
				<div
					use:melt={$option({ value: provider, label: provider.name })}
					class="relative cursor-default select-none rounded-sm px-2 py-2 text-sm outline-none transition-colors duration-150"
					class:bg-yorha-bg-secondary={$isSelected(provider)}
					class:text-yorha-text-primary={$isSelected(provider)}
					class:hover:bg-yorha-bg-tertiary={!$isSelected(provider)}
				>
					<Card class="border-none bg-transparent">
						<CardContent class="p-3">
							<!-- Provider Header -->
							<div class="flex items-center justify-between mb-2">
								<div class="flex items-center space-x-2">
									<span class="text-lg" role="img">{getTypeIcon(provider.type)}</span>
									<div>
										<div class="font-medium text-yorha-text-primary">{provider.name}</div>
										<div class="text-xs text-yorha-text-secondary">{provider.endpoint}</div>
									</div>
								</div>
								<Badge class={`text-xs ${getStatusColor(provider.status)}`}>
									{provider.status.toUpperCase()}
								</Badge>
							</div>

							<!-- Capabilities -->
							<div class="flex flex-wrap gap-1 mb-2">
								{#each provider.capabilities as capability}
									<Badge variant="outline" class="text-xs text-yorha-text-tertiary border-yorha-border">
										{capability}
									</Badge>
								{/each}
							</div>

							<!-- Models (if available) -->
							{#if provider.models.length > 0}
								<div class="text-xs text-yorha-text-secondary">
									<div class="font-medium mb-1">Available Models:</div>
									{#each provider.models.slice(0, 2) as model}
										<div class="flex justify-between">
											<span>{model.name}</span>
											<span class="text-yorha-text-tertiary">{model.size}</span>
										</div>
									{/each}
									{#if provider.models.length > 2}
										<div class="text-yorha-text-tertiary">+{provider.models.length - 2} more...</div>
									{/if}
								</div>
							{/if}

							<!-- Performance Metrics (if available and online) -->
							{#if provider.status === 'online' && provider.models[0]?.performance}
								<div class="mt-2 pt-2 border-t border-yorha-border text-xs">
									<div class="grid grid-cols-2 gap-2 text-yorha-text-secondary">
										<div>Response: {provider.models[0].performance.avgResponseTime}ms</div>
										<div>Speed: {provider.models[0].performance.tokensPerSecond} t/s</div>
									</div>
								</div>
							{/if}
						</CardContent>
					</Card>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.llm-provider-selector {
		@apply relative;
	}
	
	/* Scan line animation for cyberpunk theme */
	@keyframes scan {
		0% { transform: translateX(-100%); }
		100% { transform: translateX(100%); }
	}
	
	.animate-scan {
		animation: scan 2s linear infinite;
	}
</style>