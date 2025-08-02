<script lang="ts">
	import { onMount } from 'svelte';
	import { writable, derived } from 'svelte/store';
	import { page } from '$app/stores';
	import { createDialog, createDropzone } from '@melt-ui/svelte';
	import { Button } from '$lib/components/ui/button/index.js';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { Input } from '$lib/components/ui/input/index.js';
	import { Textarea } from '$lib/components/ui/textarea/index.js';
	import GoldenRatioLoader from '$lib/components/ui/enhanced-bits/GoldenRatioLoader.svelte';
	import { 
		Upload, 
		FileText, 
		Image, 
		Search, 
		Filter, 
		MoreVertical,
		Eye,
		Download,
		Trash2,
		Brain,
		Zap,
		Target
	} from 'lucide-svelte';

	// Evidence management stores
	let evidenceItems = $state<any[]>([]);
	let filteredEvidence = $state<any[]>([]);
	let searchQuery = $state('');
	let selectedFilter = $state('all');
	let isUploading = $state(false);
	let uploadProgress = $state(0);
	let processingStatus = $state<'loading' | 'processing' | 'success' | 'error'>('loading');

	// Context7 integration state
	let context7Enabled = $state(true);
	let semanticSearchResults = $state<any[]>([]);
	let ragEnhanced = $state(true);

	interface EvidenceItem {
		id: string;
		filename: string;
		type: 'pdf' | 'image' | 'document';
		uploadDate: Date;
		size: number;
		status: 'processing' | 'ready' | 'error';
		summary?: string;
		entities?: any[];
		prosecutionScore?: number;
		context7Analysis?: any;
		tags: string[];
		embedding?: number[];
	}

	// Melt UI Components
	const {
		elements: { trigger, overlay, content, title, description, close },
		states: { open }
	} = createDialog();

	const {
		elements: { root: dropzoneRoot, input: dropzoneInput },
		states: { isDragover }
	} = createDropzone({
		accept: 'application/pdf,image/*,.doc,.docx',
		multiple: true,
		onFilesChange: handleFilesUpload
	});

	// Computed properties
	let totalEvidence = $derived(evidenceItems.length);
	let processingCount = $derived(evidenceItems.filter(item => item.status === 'processing').length);
	let readyCount = $derived(evidenceItems.filter(item => item.status === 'ready').length);

	onMount(async () => {
		await loadExistingEvidence();
		startRealTimeUpdates();
	});

	async function loadExistingEvidence() {
		try {
			const response = await fetch('/api/evidence/list');
			if (response.ok) {
				const data = await response.json();
				evidenceItems = data.evidence || [];
				filterEvidence();
			}
		} catch (error) {
			console.error('Failed to load evidence:', error);
		}
	}

	async function handleFilesUpload(files: FileList | null) {
		if (!files || files.length === 0) return;

		isUploading = true;
		uploadProgress = 0;
		processingStatus = 'loading';

		try {
			const formData = new FormData();
			
			Array.from(files).forEach(file => {
				formData.append('files', file);
			});

			// Add processing options
			formData.append('context7Enabled', context7Enabled.toString());
			formData.append('ragEnhanced', ragEnhanced.toString());
			formData.append('extractEntities', 'true');
			formData.append('generateSummary', 'true');

			const response = await fetch('/api/evidence/upload', {
				method: 'POST',
				body: formData
			});

			if (response.ok) {
				const result = await response.json();
				
				// Add new evidence items
				evidenceItems = [...evidenceItems, ...result.evidence];
				filterEvidence();

				processingStatus = 'success';
				
				// Trigger Context7 analysis if enabled
				if (context7Enabled) {
					await triggerContext7Analysis(result.evidence);
				}

			} else {
				throw new Error('Upload failed');
			}

		} catch (error) {
			console.error('Upload error:', error);
			processingStatus = 'error';
		} finally {
			isUploading = false;
			uploadProgress = 0;
		}
	}

	async function triggerContext7Analysis(newEvidence: EvidenceItem[]) {
		try {
			for (const item of newEvidence) {
				const response = await fetch('/api/context7/analyze', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						evidenceId: item.id,
						content: item.summary,
						type: 'legal_evidence'
					})
				});

				if (response.ok) {
					const analysis = await response.json();
					
					// Update evidence item with Context7 analysis
					const index = evidenceItems.findIndex(e => e.id === item.id);
					if (index !== -1) {
						evidenceItems[index].context7Analysis = analysis;
						evidenceItems = [...evidenceItems];
					}
				}
			}
		} catch (error) {
			console.error('Context7 analysis failed:', error);
		}
	}

	async function performSemanticSearch(query: string) {
		if (!query.trim()) {
			semanticSearchResults = [];
			return;
		}

		try {
			const response = await fetch('/api/evidence/search', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					query,
					useSemanticSearch: true,
					includeContext7: context7Enabled,
					maxResults: 10
				})
			});

			if (response.ok) {
				const results = await response.json();
				semanticSearchResults = results.matches || [];
			}
		} catch (error) {
			console.error('Semantic search failed:', error);
		}
	}

	function filterEvidence() {
		let filtered = evidenceItems;

		// Apply search filter
		if (searchQuery.trim()) {
			filtered = filtered.filter(item =>
				item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
				item.summary?.toLowerCase().includes(searchQuery.toLowerCase()) ||
				item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
			);
		}

		// Apply type filter
		if (selectedFilter !== 'all') {
			filtered = filtered.filter(item => item.type === selectedFilter);
		}

		filteredEvidence = filtered;
	}

	// Watch for search and filter changes
	$effect(() => {
		filterEvidence();
	});

	// Debounced semantic search
	let searchTimeout: number;
	$effect(() => {
		if (searchQuery) {
			clearTimeout(searchTimeout);
			searchTimeout = setTimeout(() => {
				performSemanticSearch(searchQuery);
			}, 500);
		}
	});

	function startRealTimeUpdates() {
		// Simulate real-time processing updates
		setInterval(() => {
			evidenceItems = evidenceItems.map(item => {
				if (item.status === 'processing' && Math.random() > 0.7) {
					return {
						...item,
						status: 'ready',
						prosecutionScore: Math.random() * 0.4 + 0.6,
						summary: `AI-generated summary for ${item.filename}`,
						entities: ['entity1', 'entity2', 'entity3']
					};
				}
				return item;
			});
		}, 3000);
	}

	async function deleteEvidence(evidenceId: string) {
		try {
			const response = await fetch(`/api/evidence/${evidenceId}`, {
				method: 'DELETE'
			});

			if (response.ok) {
				evidenceItems = evidenceItems.filter(item => item.id !== evidenceId);
				filterEvidence();
			}
		} catch (error) {
			console.error('Delete failed:', error);
		}
	}

	function getFileIcon(type: string) {
		switch (type) {
			case 'pdf': return FileText;
			case 'image': return Image;
			default: return FileText;
		}
	}

	function getStatusColor(status: string) {
		switch (status) {
			case 'ready': return 'bg-green-100 text-green-800';
			case 'processing': return 'bg-yellow-100 text-yellow-800';
			case 'error': return 'bg-red-100 text-red-800';
			default: return 'bg-gray-100 text-gray-800';
		}
	}

	function formatFileSize(bytes: number) {
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		if (bytes === 0) return '0 Bytes';
		const i = Math.floor(Math.log(bytes) / Math.log(1024));
		return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
	}
</script>

<svelte:head>
	<title>Evidence Board - Legal AI Suite</title>
	<meta name="description" content="Upload, analyze, and manage legal evidence with AI-powered insights" />
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
	<div class="max-w-7xl mx-auto">
		<!-- Header -->
		<div class="mb-8">
			<h1 class="text-4xl font-bold text-slate-900 mb-2">
				📋 Evidence Board
			</h1>
			<p class="text-lg text-slate-600">
				Upload, analyze, and manage legal evidence with Context7 AI integration
			</p>
		</div>

		<!-- Stats Cards -->
		<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Total Evidence</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="text-2xl font-bold text-blue-600">{totalEvidence}</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Processing</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="text-2xl font-bold text-yellow-600">{processingCount}</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Ready</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="text-2xl font-bold text-green-600">{readyCount}</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader class="pb-2">
					<CardTitle class="text-sm font-medium">Context7 AI</CardTitle>
				</CardHeader>
				<CardContent>
					<div class="flex items-center space-x-2">
						<div class="w-3 h-3 {context7Enabled ? 'bg-green-500' : 'bg-gray-400'} rounded-full"></div>
						<span class="text-sm font-medium">{context7Enabled ? 'Enabled' : 'Disabled'}</span>
					</div>
				</CardContent>
			</Card>
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Upload Section -->
			<div class="lg:col-span-2">
				<Card>
					<CardHeader>
						<CardTitle class="flex items-center space-x-2">
							<Upload class="h-5 w-5" />
							<span>Upload Evidence</span>
						</CardTitle>
					</CardHeader>
					<CardContent>
						<!-- File Drop Zone -->
						<div
							use:dropzoneRoot
							class="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center transition-colors duration-200 {$isDragover ? 'border-blue-500 bg-blue-50' : 'hover:border-slate-400'}"
						>
							<input use:dropzoneInput />
							<Upload class="h-12 w-12 text-slate-400 mx-auto mb-4" />
							<p class="text-lg font-medium text-slate-700 mb-2">
								Drop files here or click to upload
							</p>
							<p class="text-sm text-slate-500">
								Supports PDF, images, and documents. Max 50MB per file.
							</p>
						</div>

						<!-- Upload Progress -->
						{#if isUploading}
							<div class="mt-6">
								<GoldenRatioLoader
									bind:status={processingStatus}
									bind:progress={uploadProgress}
									loadingText="Uploading and processing evidence..."
									aiOutput="Evidence successfully processed and added to your board."
								/>
							</div>
						{/if}

						<!-- Upload Options -->
						<div class="mt-6 flex flex-wrap gap-4">
							<label class="flex items-center space-x-2">
								<input type="checkbox" bind:checked={context7Enabled} class="rounded" />
								<span class="text-sm">Enable Context7 AI Analysis</span>
							</label>
							<label class="flex items-center space-x-2">
								<input type="checkbox" bind:checked={ragEnhanced} class="rounded" />
								<span class="text-sm">Enhanced RAG Processing</span>
							</label>
						</div>
					</CardContent>
				</Card>

				<!-- Evidence Grid -->
				<Card class="mt-6">
					<CardHeader>
						<CardTitle>Evidence Collection ({filteredEvidence.length})</CardTitle>
						
						<!-- Search and Filter -->
						<div class="flex flex-col sm:flex-row gap-4 mt-4">
							<div class="flex-1">
								<Input
									bind:value={searchQuery}
									placeholder="Search evidence..."
									class="w-full"
								/>
							</div>
							<select bind:value={selectedFilter} class="px-3 py-2 border border-slate-300 rounded-md">
								<option value="all">All Types</option>
								<option value="pdf">PDFs</option>
								<option value="image">Images</option>
								<option value="document">Documents</option>
							</select>
						</div>
					</CardHeader>
					
					<CardContent>
						{#if filteredEvidence.length === 0}
							<div class="text-center py-12">
								<FileText class="h-16 w-16 text-slate-300 mx-auto mb-4" />
								<p class="text-slate-500">No evidence files found. Upload some files to get started.</p>
							</div>
						{:else}
							<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
								{#each filteredEvidence as item (item.id)}
									<div class="border border-slate-200 rounded-lg p-4 hover:shadow-md transition-shadow">
										<div class="flex items-start justify-between mb-3">
											<div class="flex items-center space-x-2">
												<svelte:component this={getFileIcon(item.type)} class="h-5 w-5 text-slate-500" />
												<h4 class="font-medium text-slate-900 truncate flex-1">
													{item.filename}
												</h4>
											</div>
											<button class="text-slate-400 hover:text-slate-600">
												<MoreVertical class="h-4 w-4" />
											</button>
										</div>

										<div class="space-y-2 mb-3">
											<Badge class="{getStatusColor(item.status)} text-xs">
												{item.status}
											</Badge>
											<p class="text-xs text-slate-500">
												{formatFileSize(item.size)} • Uploaded {new Date(item.uploadDate).toLocaleDateString()}
											</p>
										</div>

										{#if item.summary}
											<p class="text-sm text-slate-700 mb-3 line-clamp-2">
												{item.summary}
											</p>
										{/if}

										{#if item.prosecutionScore}
											<div class="mb-3">
												<div class="flex justify-between text-xs text-slate-600 mb-1">
													<span>Prosecution Relevance</span>
													<span>{(item.prosecutionScore * 100).toFixed(0)}%</span>
												</div>
												<div class="w-full bg-slate-200 rounded-full h-2">
													<div 
														class="h-2 rounded-full bg-gradient-to-r from-green-400 to-blue-500"
														style:width="{item.prosecutionScore * 100}%"
													></div>
												</div>
											</div>
										{/if}

										{#if item.tags.length > 0}
											<div class="flex flex-wrap gap-1 mb-3">
												{#each item.tags.slice(0, 3) as tag}
													<Badge variant="outline" class="text-xs">{tag}</Badge>
												{/each}
											</div>
										{/if}

										<!-- Actions -->
										<div class="flex space-x-2">
											<Button size="sm" variant="outline" class="flex-1">
												<Eye class="h-3 w-3 mr-1" />
												View
											</Button>
											<Button size="sm" variant="outline">
												<Download class="h-3 w-3" />
											</Button>
											<Button size="sm" variant="outline" onclick={() => deleteEvidence(item.id)}>
												<Trash2 class="h-3 w-3" />
											</Button>
										</div>
									</div>
								{/each}
							</div>
						{/if}
					</CardContent>
				</Card>
			</div>

			<!-- AI Insights Panel -->
			<div class="space-y-6">
				<!-- Context7 Analysis -->
				<Card>
					<CardHeader>
						<CardTitle class="flex items-center space-x-2">
							<Brain class="h-5 w-5" />
							<span>AI Insights</span>
						</CardTitle>
					</CardHeader>
					<CardContent>
						{#if context7Enabled}
							<div class="space-y-4">
								<div class="flex items-center space-x-2">
									<Zap class="h-4 w-4 text-blue-500" />
									<span class="text-sm font-medium">Context7 Analysis Active</span>
								</div>
								
								<div class="text-sm text-slate-600">
									AI is continuously analyzing your evidence for:
								</div>
								
								<ul class="text-sm text-slate-600 space-y-1">
									<li>• Legal entity extraction</li>
									<li>• Case law connections</li>
									<li>• Prosecution relevance scoring</li>
									<li>• Semantic relationship mapping</li>
								</ul>
							</div>
						{:else}
							<div class="text-center py-6">
								<Target class="h-12 w-12 text-slate-300 mx-auto mb-2" />
								<p class="text-sm text-slate-500">Enable Context7 for AI insights</p>
							</div>
						{/if}
					</CardContent>
				</Card>

				<!-- Semantic Search Results -->
				{#if semanticSearchResults.length > 0}
					<Card>
						<CardHeader>
							<CardTitle class="flex items-center space-x-2">
								<Search class="h-5 w-5" />
								<span>Semantic Search</span>
							</CardTitle>
						</CardHeader>
						<CardContent>
							<div class="space-y-3">
								{#each semanticSearchResults.slice(0, 5) as result}
									<div class="p-3 bg-slate-50 rounded-md">
										<h5 class="font-medium text-sm text-slate-900 mb-1">
											{result.filename}
										</h5>
										<p class="text-xs text-slate-600 mb-2">
											Similarity: {(result.similarity * 100).toFixed(0)}%
										</p>
										<p class="text-xs text-slate-700 line-clamp-2">
											{result.content}
										</p>
									</div>
								{/each}
							</div>
						</CardContent>
					</Card>
				{/if}

				<!-- Quick Actions -->
				<Card>
					<CardHeader>
						<CardTitle>Quick Actions</CardTitle>
					</CardHeader>
					<CardContent>
						<div class="space-y-3">
							<Button class="w-full justify-start">
								<Brain class="h-4 w-4 mr-2" />
								Generate Case Summary
							</Button>
							<Button variant="outline" class="w-full justify-start">
								<Search class="h-4 w-4 mr-2" />
								Find Similar Cases
							</Button>
							<Button variant="outline" class="w-full justify-start">
								<Download class="h-4 w-4 mr-2" />
								Export Evidence Report
							</Button>
						</div>
					</CardContent>
				</Card>
			</div>
		</div>
	</div>
</div>

<style>
	.line-clamp-2 {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>