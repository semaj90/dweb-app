<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';

	interface ServiceHealth {
		name: string;
		url: string;
		status: 'online' | 'offline' | 'degraded';
		responseTime?: number;
		lastCheck: number;
		details?: any;
	}

	interface HealthData {
		timestamp: number;
		overall_status: 'healthy' | 'degraded' | 'critical';
		health_percentage: number;
		services_online: number;
		services_total: number;
		cuda: {
			service_available: boolean;
			worker_available: boolean;
			gpu_ready: boolean;
			response_time: number | null;
		};
		services: ServiceHealth[];
		summary: {
			critical_services: string[];
			degraded_services: string[];
			offline_services: string[];
		};
		recommendations: string[];
	}

	const healthData = writable<HealthData | null>(null);
	const loading = writable(true);
	const error = writable<string | null>(null);
	let refreshInterval: number;

	const fetchHealth = async () => {
		try {
			loading.set(true);
			const response = await fetch('/api/v1/health/cuda');
			
			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}
			
			const data = await response.json();
			healthData.set(data);
			error.set(null);
		} catch (err) {
			console.error('Health check failed:', err);
			error.set(err instanceof Error ? err.message : 'Unknown error');
		} finally {
			loading.set(false);
		}
	};

	const getStatusColor = (status: string) => {
		switch (status) {
			case 'online':
			case 'healthy':
				return 'text-green-600 bg-green-50 border-green-200';
			case 'degraded':
				return 'text-yellow-600 bg-yellow-50 border-yellow-200';
			case 'offline':
			case 'critical':
				return 'text-red-600 bg-red-50 border-red-200';
			default:
				return 'text-gray-600 bg-gray-50 border-gray-200';
		}
	};

	const getStatusIcon = (status: string) => {
		switch (status) {
			case 'online':
			case 'healthy':
				return '✅';
			case 'degraded':
				return '⚠️';
			case 'offline':
			case 'critical':
				return '❌';
			default:
				return '🔍';
		}
	};

	const formatTimestamp = (timestamp: number) => {
		return new Date(timestamp).toLocaleString();
	};

	const formatResponseTime = (time?: number) => {
		if (!time) return 'N/A';
		return `${time}ms`;
	};

	onMount(() => {
		fetchHealth();
		// Refresh every 10 seconds
		refreshInterval = setInterval(fetchHealth, 10000);
		
		return () => {
			clearInterval(refreshInterval);
		};
	});
</script>

<svelte:head>
	<title>System Health Dashboard - Legal AI Platform</title>
</svelte:head>

<div class="container mx-auto p-6 max-w-7xl">
	<div class="flex justify-between items-center mb-8">
		<div>
			<h1 class="text-3xl font-bold text-gray-900">System Health Dashboard</h1>
			<p class="text-gray-600 mt-1">Legal AI Platform - CUDA GPU Integration Status</p>
		</div>
		<button
			on:click={fetchHealth}
			class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
			disabled={$loading}
		>
			{#if $loading}
				<div class="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
			{:else}
				🔄
			{/if}
			Refresh
		</button>
	</div>

	{#if $error}
		<div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
			<div class="flex items-center gap-2">
				<span class="text-xl">❌</span>
				<div>
					<h3 class="font-semibold">Health Check Failed</h3>
					<p class="text-sm mt-1">{$error}</p>
				</div>
			</div>
		</div>
	{/if}

	{#if $healthData}
		<!-- Overall Status -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
			<div class={`p-6 rounded-lg border-2 ${getStatusColor($healthData.overall_status)}`}>
				<div class="flex items-center gap-3">
					<span class="text-3xl">{getStatusIcon($healthData.overall_status)}</span>
					<div>
						<h3 class="text-lg font-semibold capitalize">{$healthData.overall_status}</h3>
						<p class="text-sm opacity-75">Overall System Status</p>
					</div>
				</div>
			</div>
			
			<div class="p-6 rounded-lg border-2 border-blue-200 bg-blue-50">
				<div class="flex items-center gap-3">
					<span class="text-3xl">📊</span>
					<div>
						<h3 class="text-lg font-semibold text-blue-700">{$healthData.health_percentage}%</h3>
						<p class="text-sm text-blue-600">Services Online ({$healthData.services_online}/{$healthData.services_total})</p>
					</div>
				</div>
			</div>

			<div class={`p-6 rounded-lg border-2 ${$healthData.cuda.gpu_ready ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
				<div class="flex items-center gap-3">
					<span class="text-3xl">🎯</span>
					<div>
						<h3 class={`text-lg font-semibold ${$healthData.cuda.gpu_ready ? 'text-green-700' : 'text-red-700'}`}>
							{$healthData.cuda.gpu_ready ? 'GPU Ready' : 'GPU Not Available'}
						</h3>
						<p class={`text-sm ${$healthData.cuda.gpu_ready ? 'text-green-600' : 'text-red-600'}`}>
							CUDA Worker Status
						</p>
					</div>
				</div>
			</div>
		</div>

		<!-- CUDA Service Details -->
		<div class="mb-8">
			<h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
				<span>⚡</span> CUDA GPU Service
			</h2>
			<div class="bg-white rounded-lg border shadow-sm p-6">
				<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
					<div class="text-center">
						<div class={`text-2xl mb-2 ${$healthData.cuda.service_available ? 'text-green-600' : 'text-red-600'}`}>
							{$healthData.cuda.service_available ? '✅' : '❌'}
						</div>
						<h4 class="font-semibold">Service</h4>
						<p class="text-sm text-gray-600">{$healthData.cuda.service_available ? 'Running' : 'Offline'}</p>
					</div>
					<div class="text-center">
						<div class={`text-2xl mb-2 ${$healthData.cuda.worker_available ? 'text-green-600' : 'text-red-600'}`}>
							{$healthData.cuda.worker_available ? '🔧' : '❌'}
						</div>
						<h4 class="font-semibold">Worker</h4>
						<p class="text-sm text-gray-600">{$healthData.cuda.worker_available ? 'Available' : 'Not Built'}</p>
					</div>
					<div class="text-center">
						<div class={`text-2xl mb-2 ${$healthData.cuda.gpu_ready ? 'text-green-600' : 'text-red-600'}`}>
							{$healthData.cuda.gpu_ready ? '🚀' : '❌'}
						</div>
						<h4 class="font-semibold">GPU Ready</h4>
						<p class="text-sm text-gray-600">{$healthData.cuda.gpu_ready ? 'Yes' : 'No'}</p>
					</div>
					<div class="text-center">
						<div class="text-2xl mb-2 text-blue-600">⏱️</div>
						<h4 class="font-semibold">Response Time</h4>
						<p class="text-sm text-gray-600">{formatResponseTime($healthData.cuda.response_time)}</p>
					</div>
				</div>
			</div>
		</div>

		<!-- Services Grid -->
		<div class="mb-8">
			<h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
				<span>🏗️</span> All Services
			</h2>
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
				{#each $healthData.services as service}
					<div class={`p-4 rounded-lg border-2 ${getStatusColor(service.status)}`}>
						<div class="flex items-center justify-between mb-2">
							<h3 class="font-semibold capitalize">{service.name.replace('-', ' ')}</h3>
							<span class="text-xl">{getStatusIcon(service.status)}</span>
						</div>
						<div class="space-y-1 text-sm">
							<p><span class="font-medium">Status:</span> {service.status}</p>
							{#if service.responseTime}
								<p><span class="font-medium">Response:</span> {formatResponseTime(service.responseTime)}</p>
							{/if}
							<p class="text-xs opacity-75">Last check: {formatTimestamp(service.lastCheck)}</p>
							{#if service.details && typeof service.details === 'object'}
								<details class="mt-2">
									<summary class="cursor-pointer text-xs opacity-75">Details</summary>
									<pre class="text-xs mt-1 opacity-60 overflow-x-auto">{JSON.stringify(service.details, null, 2)}</pre>
								</details>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		</div>

		<!-- Recommendations -->
		{#if $healthData.recommendations.length > 0}
			<div class="mb-8">
				<h2 class="text-2xl font-semibold mb-4 flex items-center gap-2">
					<span>💡</span> Recommendations
				</h2>
				<div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
					<ul class="space-y-2">
						{#each $healthData.recommendations as recommendation}
							<li class="flex items-start gap-2">
								<span class="text-blue-600 mt-0.5">•</span>
								<code class="text-sm bg-blue-100 px-2 py-1 rounded">{recommendation}</code>
							</li>
						{/each}
					</ul>
				</div>
			</div>
		{/if}

		<!-- Summary -->
		<div class="bg-gray-50 rounded-lg p-4">
			<h3 class="font-semibold mb-2">Summary</h3>
			<div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
				{#if $healthData.summary.critical_services.length > 0}
					<div>
						<h4 class="font-medium text-red-600 mb-1">Critical Services Down:</h4>
						<ul class="space-y-1">
							{#each $healthData.summary.critical_services as service}
								<li class="text-red-700">• {service}</li>
							{/each}
						</ul>
					</div>
				{/if}
				{#if $healthData.summary.degraded_services.length > 0}
					<div>
						<h4 class="font-medium text-yellow-600 mb-1">Degraded Services:</h4>
						<ul class="space-y-1">
							{#each $healthData.summary.degraded_services as service}
								<li class="text-yellow-700">• {service}</li>
							{/each}
						</ul>
					</div>
				{/if}
				{#if $healthData.summary.offline_services.length > 0}
					<div>
						<h4 class="font-medium text-gray-600 mb-1">Offline Services:</h4>
						<ul class="space-y-1">
							{#each $healthData.summary.offline_services as service}
								<li class="text-gray-700">• {service}</li>
							{/each}
						</ul>
					</div>
				{/if}
			</div>
			<p class="text-xs text-gray-500 mt-4">
				Last updated: {formatTimestamp($healthData.timestamp)} | Auto-refresh: 10s
			</p>
		</div>
	{:else if $loading}
		<div class="flex items-center justify-center py-12">
			<div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
			<span class="ml-2 text-gray-600">Loading health data...</span>
		</div>
	{/if}
</div>

<style>
	.container {
		font-family: 'Inter', system-ui, -apple-system, sans-serif;
	}
	
	pre {
		white-space: pre-wrap;
		word-break: break-all;
	}
</style>