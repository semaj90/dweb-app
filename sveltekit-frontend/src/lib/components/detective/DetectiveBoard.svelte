<!-- Detective Board - Enhanced 3-Column Grid with NES.css, RabbitMQ & GPU Integration -->
<script lang="ts">
	import Badge from '$lib/components/ui/Badge.svelte';
	import Button from '$lib/components/ui/button/Button.svelte';
	import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import { page } from '$app/stores';
	import Fuse from 'fuse.js';
	import { dndzone } from 'svelte-dnd-action';
	import { onDestroy, onMount } from 'svelte';
	import { GPU, Activity, Database, MessageSquare, Cpu, Zap } from 'lucide-svelte';

	// SVELTE 5: External, app-wide stores are still valid.
	// Access page data directly
	import { page as $page } from '$app/stores';
	import { evidenceStore } from '$lib/stores/evidence-unified';
	import { callContext7Tool, getContextAwareSuggestions } from '$lib/ai/mcp-helpers';
	import EvidenceNode from '../canvas/EvidenceNode.svelte';
	import EvidenceCard from './EvidenceCard.svelte';
	import UploadZone from './UploadZone.svelte';
	import OptimizedMinIOUpload from '../upload/OptimizedMinIOUpload.svelte';

	// Enhanced integrations
	import { rabbitMQService } from '$lib/services/rabbitmq-service';
	import { vectorService } from '$lib/services/postgresql-vector-service';
	import { gpuService } from '$lib/services/gpu-acceleration-service';

	// --- Svelte 5 State Management ---
	// SVELTE 5: Subscribe to external evidenceStore manually.
	// The evidenceStore returns a state object with evidence array
	let evidenceStoreState = $state<any>({ evidence: [], isLoading: false, error: null, isConnected: false });

	// Create a derived state for just the evidence array
	let allEvidence = $derived(evidenceStoreState.evidence || []);

	// Case ID for associating uploaded evidence
	let caseId = $state('');

	// SVELTE 5: Use runes (`$state`) for all component-local state.
	let viewMode = $state<'columns' | 'canvas'>('columns');
	let canvasContainer: HTMLDivElement;
	let columns = $state([
		{ id: 'new', title: 'New Evidence', items: [] },
		{ id: 'processing', title: 'Processing', items: [] },
		{ id: 'verified', title: 'Verified', items: [] }
	]);
	let canvasEvidence = $state([]);

	// SVELTE 5: Converted from writable store to a rune.
	let activeUsers = $state<{ name?: string; email?: string }[]>([]);

	// Enhanced system status tracking
	let systemStatus = $state({
		rabbitMQ: { connected: false, health: 'unknown' },
		postgreSQL: { connected: false, vectorCount: 0 },
		gpu: { available: false, utilization: 0, model: 'RTX 3060 Ti' },
		processingStats: { totalFiles: 0, processed: 0, queued: 0 }
	});

	let contextMenu = $state({
		show: false,
		x: 0,
		y: 0,
		item: null as any
	});

	let miniModal = $state({
		show: false,
		x: 0,
		y: 0,
		type: ''
	});

	let findModal = $state({
		show: false,
		query: '',
		results: [] as any[],
		loading: false,
		error: '',
		suggestions: [] as any[]
	});

	// --- Component Logic & Functions ---

	// Enhanced system initialization
	onMount(async () => {
		await initializeEnhancedSystems();
		setupRealTimeUpdates();
		
		// Subscribe to evidence store
		const unsubscribeEvidence = evidenceStore.subscribe((value) => {
			evidenceStoreState = value;
		});
		
		onDestroy(() => {
			unsubscribeEvidence();
		});
	});

	async function initializeEnhancedSystems() {
		// RabbitMQ connection
		try {
			const rabbitMQStatus = await rabbitMQService.connect();
			systemStatus.rabbitMQ.connected = rabbitMQStatus.connected;
			systemStatus.rabbitMQ.health = rabbitMQStatus.health;
		} catch (error) {
			console.warn('RabbitMQ connection failed:', error);
		}

		// PostgreSQL vector status
		try {
			const vectorStatus = await vectorService.getStatus();
			systemStatus.postgreSQL.connected = vectorStatus.connected;
			systemStatus.postgreSQL.vectorCount = vectorStatus.vectorCount;
		} catch (error) {
			console.warn('PostgreSQL vector service failed:', error);
		}

		// GPU service status
		try {
			const gpuStatus = await gpuService.getStatus();
			systemStatus.gpu.available = gpuStatus.available;
			systemStatus.gpu.utilization = gpuStatus.utilization;
		} catch (error) {
			console.warn('GPU service failed:', error);
		}
	}

	function setupRealTimeUpdates() {
		// RabbitMQ real-time evidence updates
		rabbitMQService.subscribe('evidence.processing', (message) => {
			updateProcessingStats(message);
		});

		rabbitMQService.subscribe('evidence.completed', (message) => {
			updateEvidenceStatus(message);
		});
	}

	function updateProcessingStats(message: any) {
		systemStatus.processingStats.queued = message.queuedCount || 0;
		systemStatus.processingStats.processed = message.processedCount || 0;
	}

	function updateEvidenceStatus(message: any) {
		// Update evidence item status based on RabbitMQ message
		const evidenceId = message.evidenceId;
		const newStatus = message.status;
		// Update column positions based on processing status
		moveEvidenceBetweenColumns(evidenceId, newStatus);
	}

	function moveEvidenceBetweenColumns(evidenceId: string, newStatus: string) {
		// Logic to move evidence between columns based on processing status
		const targetColumnId = newStatus === 'completed' ? 'verified' : 'processing';

		// Find and move evidence item
		columns.forEach(column => {
			const itemIndex = column.items.findIndex((item: any) => item.id === evidenceId);
			if (itemIndex !== -1) {
				const item = column.items.splice(itemIndex, 1)[0];
				const targetColumn = columns.find(col => col.id === targetColumnId);
				if (targetColumn) {
					targetColumn.items.push(item);
				}
			}
		});
	}

	function switchViewMode(mode: 'columns' | 'canvas') {
		viewMode = mode;
	}

	function handleFileUpload(result: any, columnId: string) {
		console.log('File uploaded to MinIO:', result, 'for column:', columnId);

		// Create evidence item from MinIO upload result
		const newEvidence = {
			id: result.id || `evidence-${Date.now()}-${Math.random()}`,
			title: result.originalName || result.fileName,
			fileName: result.fileName,
			fileSize: result.fileSize,
			type: result.metadata?.evidenceType || 'document',
			evidenceType: result.metadata?.evidenceType || 'document',
			createdAt: new Date(result.metadata?.uploadedAt || Date.now()),
			tags: [],
			x: 100 + Math.random() * 200,
			y: 100 + Math.random() * 200,
			// MinIO specific fields
			url: result.url,
			bucket: result.bucket,
			hash: result.hash,
			minioId: result.id,
			caseId: result.metadata?.caseId
		};

		// Add to the appropriate column
		columns = columns.map(col =>
			col.id === columnId
				? { ...col, items: [...col.items, newEvidence] }
				: col
		);

		// Also update the evidence store if needed
		// evidenceStore.addEvidence(newEvidence);
	}

	function handleUploadError(error: string, columnId: string) {
		console.error('Upload to MinIO failed:', error);
		// You could show a notification or alert here
	}

	function handleDndConsider(event: any, columnId: string) {
		console.log('DnD consider:', event, columnId);
	}

	function handleDndFinalize(event: any, columnId: string) {
		console.log('DnD finalize:', event, columnId);
	}

	function handleRightClick(event: MouseEvent, item: any) {
		event.preventDefault();
		contextMenu.show = true;
		contextMenu.x = event.clientX;
		contextMenu.y = event.clientY;
		contextMenu.item = item;
	}

	function closeContextMenu() {
		// SVELTE 5: Direct mutation is the idiomatic way to update state objects.
		contextMenu.show = false;
	}

	function showMiniModal(type: string, event: MouseEvent) {
		miniModal.show = true;
		miniModal.type = type;
		// Position modal near the cursor
		miniModal.x = event.clientX + 15;
		miniModal.y = event.clientY + 15;
	}

	function hideMiniModal() {
		miniModal.show = false;
	}

	function broadcastPositionUpdate(id: string, x: number, y: number) {
		console.log('Position update:', id, x, y);
	}

	function handleViewEvidence(item: any) {
		console.log('Viewing evidence:', item.title);
		// Add your logic to open a modal or navigate to a details page
		window.open(`/evidence/${item.id}`, '_blank');
	}

	function handleShowMoreOptions(item: any) {
		console.log('Showing more options for:', item.title);
		// Add your logic to show a context menu
		contextMenu.show = true;
		contextMenu.item = item;
	}

	function handleGlobalKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			closeContextMenu();
			closeFindModal();
		}
	}

	async function saveTo(target: string) {
		if (!contextMenu.item) return closeContextMenu();
		const itemToSave = contextMenu.item;
		closeContextMenu(); // Close menu immediately for better UX

		// Note: All API calls are stubbed and will work as before.
		try {
			await fetch('/api/user-activity', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					userId: $page.data?.user?.id,
					evidenceId: itemToSave.id,
					action: 'save',
					target
				})
			});
		} catch (e) {
			console.warn('User activity store update failed', e);
		}

		// ... other API calls ...
	}

	function openFindModal() {
		findModal.show = true;
		findModal.query = contextMenu.item?.title || '';
		findModal.results = [];
		findModal.loading = false;
		findModal.error = '';
		findModal.suggestions = [];
	}
	function closeFindModal() {
		findModal.show = false;
		closeContextMenu();
	}

	async function runFindSearch() {
		if (!contextMenu.item) return closeFindModal();
		findModal.loading = true;
		findModal.error = '';
		findModal.results = [];
		findModal.suggestions = [];

		// 1. Local fuzzy search (Fuse.js)
		try {
			// SVELTE 5: Use the reactive `allEvidence` rune directly. No `get()` needed.
			const items = allEvidence || [];
			const fuse = new Fuse(items, { keys: ['title', 'description', 'tags'] });
			const fuseResults = fuse.search(findModal.query || contextMenu.item.title || '');
			findModal.results = fuseResults.map((r) => r.item); // Extract the items
		} catch (e) {
			findModal.error = 'Local search failed';
		}

		// 2. Qdrant/Vector search (stubbed)
		try {
			const resp = await fetch('/api/vector-search', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					query: findModal.query || contextMenu.item.title
				})
			});
			const vectorResults = await resp.json();
			findModal.results = [...findModal.results, ...vectorResults];
		} catch (e) {
			findModal.error += ' Qdrant search failed.';
		}

		findModal.loading = false;
	}

	// Canvas-specific handlers
	function handleCanvasDrop(event: DragEvent) {
		event.preventDefault();
		// Handle dropping evidence onto canvas
		const data = event.dataTransfer?.getData('text/plain');
		if (data) {
			try {
				const item = JSON.parse(data);
				const rect = canvasContainer?.getBoundingClientRect();
				if (rect) {
					item.x = event.clientX - rect.left;
					item.y = event.clientY - rect.top;
					canvasEvidence = [...canvasEvidence, item];
				}
			} catch (e) {
				console.error('Failed to parse dropped data:', e);
			}
		}
	}

	function handleCanvasDragStart(event: DragEvent, item: any) {
		if (event.dataTransfer) {
			event.dataTransfer.effectAllowed = 'move';
			event.dataTransfer.setData('text/plain', JSON.stringify(item));
		}
	}

	function handleCanvasDragEnd(event: DragEvent, item: any) {
		// Update item position after drag
		const rect = canvasContainer?.getBoundingClientRect();
		if (rect) {
			const newX = event.clientX - rect.left;
			const newY = event.clientY - rect.top;
			canvasEvidence = canvasEvidence.map(e =>
				e.id === item.id ? { ...e, x: newX, y: newY } : e
			);
			broadcastPositionUpdate(item.id, newX, newY);
		}
	}

	function getConnections() {
		// Return an array of connection lines between related evidence
		// This is a placeholder - you can implement actual relationship logic
		const connections = [];
		// Example: connect items that share tags or are related
		for (let i = 0; i < canvasEvidence.length - 1; i++) {
			for (let j = i + 1; j < canvasEvidence.length; j++) {
				const item1 = canvasEvidence[i];
				const item2 = canvasEvidence[j];
				// Check if items are related (example logic)
				if (item1.tags?.some(tag => item2.tags?.includes(tag))) {
					connections.push({
						x1: (item1.x || 100) + 100, // Center of card
						y1: (item1.y || 100) + 50,
						x2: (item2.x || 100) + 100,
						y2: (item2.y || 100) + 50
					});
				}
			}
		}
		return connections;
	}
</script>

<svelte:window onclick={closeContextMenu} onkeydown={handleGlobalKeydown} />

<div class="w-full h-full min-h-screen bg-background detective-board-nes">
	<!-- Header -->
	<Card class="mb-6">
		<CardHeader>
			<div class="flex justify-between items-center">
				<div class="flex items-center gap-4">
					<div
						class="w-12 h-12 bg-primary bg-opacity-10 rounded-full flex items-center justify-center"
					>
						<span class="text-2xl">🕵️</span>
					</div>
					<div>
						<CardTitle class="text-2xl">Detective Board</CardTitle>
						<p class="text-muted-foreground">Case Evidence Management System</p>
					</div>
				</div>

				<div class="flex items-center gap-4">
					<!-- View Mode Switcher -->
					<div class="flex gap-2">
						<Button
							variant={viewMode === 'columns' ? 'default' : 'outline'}
							size="sm"
							onclick={() => switchViewMode('columns')}
						>
							<span class="mr-2">📋</span>
							Columns
						</Button>
						<Button
							variant={viewMode === 'canvas' ? 'default' : 'outline'}
							size="sm"
							onclick={() => switchViewMode('canvas')}
						>
							<span class="mr-2">🎨</span>
							Canvas
						</Button>
					</div>

					<!-- SVELTE 5: No more `$` prefix for store subscriptions in the template -->
					{#if activeUsers.length > 0}
						<div class="flex items-center gap-2">
							<div class="flex -space-x-2">
								{#each activeUsers.slice(0, 3) as user}
									<div
										class="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-medium border-2 border-background"
									>
										{user.name?.charAt(0) || user.email?.charAt(0) || '?'}
									</div>
								{/each}
								{#if activeUsers.length > 3}
									<div
										class="w-8 h-8 bg-muted text-muted-foreground rounded-full flex items-center justify-center text-sm border-2 border-background"
									>
										+{activeUsers.length - 3}
									</div>
								{/if}
							</div>
							<Badge variant="outline">{activeUsers.length} online</Badge>
						</div>
					{/if}

					<Button size="sm">
						<span class="mr-2">➕</span>
						New Case
					</Button>
				</div>
			</div>
		</CardHeader>
	</Card>

	<!-- Main Board Area -->
	<main class="flex-1">
		{#if viewMode === 'columns'}
			<!-- Columns Container -->
			<div class="grid grid-cols-1 md:grid-cols-3 gap-6 h-full">
				{#each columns as column (column.id)}
					<Card class="h-fit">
						<CardHeader class="pb-3">
							<div class="flex justify-between items-center">
								<CardTitle class="text-lg flex items-center gap-2">
									<div class="w-3 h-3 bg-primary rounded-full" />
									{column.title}
								</CardTitle>
								<Badge variant="secondary">
									{column.items.length}
								</Badge>
							</div>
						</CardHeader>

						<CardContent class="space-y-4">
							{#if column.id === 'new'}
								<UploadZone
									onUpload={(result) => handleFileUpload(result, column.id)}
									onUploadError={(error) => handleUploadError(error, column.id)}
									caseId={caseId}
								/>
							{/if}

							<div
								class="space-y-3 min-h-[200px]"
								use:dndzone={{
									items: column.items,
									flipDurationMs: 200,
									dropTargetStyle: {
										background: 'hsl(var(--muted))',
										border: '2px dashed hsl(var(--primary))',
										borderRadius: '8px'
									}
								}}
								on:consider={(e) => handleDndConsider(e, column.id)}
								on:finalize={(e) => handleDndFinalize(e, column.id)}
							>
								{#each column.items as item (item.id)}
									<div
										class="cursor-grab active:cursor-grabbing transition-transform hover:scale-105"
										oncontextmenu={(e) => handleRightClick(e, item)}
										role="button"
										tabindex="0"
									>
										<EvidenceCard
											item={item}
											onView={handleViewEvidence}
											onMoreOptions={handleShowMoreOptions}
										/>
									</div>
								{/each}
							</div>
						</CardContent>
					</Card>
				{/each}
			</div>
		{:else}
			<!-- Canvas Container -->
			<Card class="h-[calc(100vh-200px)]">
				<CardContent class="p-0 h-full">
					<div
						bind:this={canvasContainer}
						class="relative w-full h-full bg-slate-50 dark:bg-slate-900 overflow-auto"
						ondrop={(e) => handleCanvasDrop(e)}
						ondragover={(e) => e.preventDefault()}
					>
						<!-- Grid background -->
						<div class="absolute inset-0 bg-grid-pattern opacity-5"></div>

						<!-- Evidence nodes on canvas -->
						{#each canvasEvidence as item (item.id)}
							<div
								class="absolute p-4 bg-background border-2 border-border rounded-lg shadow-lg cursor-move hover:shadow-xl transition-shadow"
								style="left: {item.x || 100}px; top: {item.y || 100}px; min-width: 200px;"
								draggable="true"
								ondragstart={(e) => handleCanvasDragStart(e, item)}
								ondragend={(e) => handleCanvasDragEnd(e, item)}
								oncontextmenu={(e) => handleRightClick(e, item)}
								role="button"
								tabindex="0"
							>
								<EvidenceCard
									item={item}
									onView={handleViewEvidence}
									onMoreOptions={handleShowMoreOptions}
								/>
							</div>
						{/each}

						<!-- Connection lines (optional - for relationships) -->
						<svg class="absolute inset-0 pointer-events-none" style="width: 100%; height: 100%;">
							{#each getConnections() as connection}
								<line
									x1={connection.x1}
									y1={connection.y1}
									x2={connection.x2}
									y2={connection.y2}
									stroke="currentColor"
									stroke-width="2"
									stroke-dasharray="5,5"
									opacity="0.3"
								/>
							{/each}
						</svg>

						<!-- Empty state -->
						{#if canvasEvidence.length === 0}
							<div class="absolute inset-0 flex items-center justify-center">
								<div class="text-center text-muted-foreground">
									<p class="text-lg mb-2">No evidence on canvas</p>
									<p class="text-sm">Drag evidence here or switch to column view to add items</p>
								</div>
							</div>
						{/if}
					</div>
				</CardContent>
			</Card>
		{/if}
	</main>
</div>

<!-- Context Menu -->
{#if contextMenu.show}
	<ContextMenu.Root open>
		<ContextMenu.Content
			class="fixed"
			style="left: {contextMenu.x}px; top: {contextMenu.y}px;"
		>
			<ContextMenu.Item onselect={() => window.open(`/evidence/${contextMenu.item?.id}`, '_blank')}>
				View Details
			</ContextMenu.Item>
			<ContextMenu.Item
				onselect={() => (window.location.href = `/evidence/${contextMenu.item?.id}/edit`)}
			>
				Edit
			</ContextMenu.Item>
			<ContextMenu.Separator />
			<ContextMenu.Item
				onmouseenter={(e) => showMiniModal('citation', e)}
				onmouseleave={hideMiniModal}
				onselect={() => saveTo('savedcitations')}
			>
				Add to /savedcitations
			</ContextMenu.Item>
			<ContextMenu.Item
				onmouseenter={(e) => showMiniModal('mcpcontext', e)}
				onmouseleave={hideMiniModal}
				onselect={() => saveTo('mcpcontext')}
			>
				Add to MCP Context (LLM)
			</ContextMenu.Item>
			<ContextMenu.Separator />
			<ContextMenu.Item
				onmouseenter={(e) => showMiniModal('find', e)}
				onmouseleave={hideMiniModal}
				onselect={openFindModal}
			>
				Find Related...
			</ContextMenu.Item>
		</ContextMenu.Content>
	</ContextMenu.Root>
{/if}

<!-- Find Modal -->
{#if findModal.show}
	<div
		class="fixed z-50 inset-0 bg-black/60 flex items-center justify-center"
		onclick|self={closeFindModal}
	>
		<div
			class="bg-background border border-primary rounded-lg shadow-lg p-6 w-full max-w-lg"
			role="dialog"
			aria-modal="true"
		>
			<div class="flex flex-col gap-4">
				<input
					class="w-full border rounded px-3 py-2 text-base bg-muted text-foreground focus:outline-none focus:ring focus:border-primary"
					type="text"
					bind:value={findModal.query}
					placeholder="Enter keywords or question..."
					onkeydown={(e) => {
						if (e.key === 'Enter') runFindSearch();
					}}
				/>
				<div class="flex gap-2">
					<Button onclick={runFindSearch} disabled={findModal.loading}>
						{#if findModal.loading}Searching...{:else}Search{/if}
					</Button>
					<Button variant="outline" onclick={closeFindModal}>Close</Button>
				</div>

				{#if findModal.error}
					<div class="text-red-500">{findModal.error}</div>
				{/if}
				{#if findModal.results.length > 0}
					<div class="border-t pt-4">
						<h3 class="font-semibold mb-2">Results:</h3>
						<ul class="space-y-2 max-h-60 overflow-y-auto">
							{#each findModal.results as result}
								<li class="p-2 rounded hover:bg-muted cursor-pointer border-b border-muted-foreground/10">
									{result.title || result.text || JSON.stringify(result)}
								</li>
							{/each}
						</ul>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<!-- Mini-modal for context menu hover -->
{#if miniModal.show}
	<!-- ... miniModal markup remains the same ... -->
{/if}

<!-- Enhanced NES.css Styles -->
<style>
	@import url('https://unpkg.com/nes.css@2.3.0/css/nes.min.css');
	@import url('https://fonts.googleapis.com/css?family=Press+Start+2P&display=swap');

	/* Detective Board Container */
	.detective-board-container {
		font-family: "Press Start 2P", cursive;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		min-height: 100vh;
		padding: 20px;
	}

	/* Enhanced Header Styles */
	.detective-header {
		background: #212529 !important;
		border: 4px solid #fff !important;
		margin-bottom: 20px;
		padding: 20px;
	}

	.header-content {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 20px;
	}

	.detective-title h1 {
		font-size: 24px;
		margin-bottom: 10px;
		color: #92cc41;
	}

	.detective-title .subtitle {
		font-size: 10px;
		color: #fff;
		margin: 0;
	}

	/* System Status Grid */
	.system-status-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
		gap: 10px;
		margin-top: 15px;
	}

	.status-card {
		padding: 10px;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 5px;
		min-height: 70px;
		background: #ffffff;
		border: 2px solid #212529;
	}

	.status-card.is-success {
		background: #92cc41;
		color: white;
	}

	.status-card.is-error {
		background: #e76e55;
		color: white;
	}

	.status-card.is-warning {
		background: #f7d51d;
		color: #212529;
	}

	.status-label {
		font-size: 8px;
		text-align: center;
	}

	.status-value {
		font-size: 10px;
		font-weight: bold;
		text-align: center;
	}

	/* Control Panel */
	.control-panel {
		display: flex;
		gap: 15px;
		align-items: center;
		flex-wrap: wrap;
	}

	.view-switcher {
		display: flex;
		gap: 10px;
	}

	/* Performance Bar */
	.performance-bar {
		background: #ffffff;
		border: 4px solid #212529;
		margin-bottom: 20px;
		padding: 15px;
	}

	.perf-stats {
		display: flex;
		justify-content: space-between;
		align-items: center;
		flex-wrap: wrap;
		gap: 10px;
	}

	.stats-row {
		display: flex;
		gap: 20px;
		align-items: center;
		font-size: 10px;
	}

	.stats-row span {
		display: flex;
		align-items: center;
		gap: 5px;
	}

	/* Enhanced Evidence Grid */
	.evidence-grid-container {
		width: 100%;
	}

	.evidence-columns {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
		gap: 20px;
		width: 100%;
	}

	/* Enhanced Column Styles */
	.evidence-column {
		background: #ffffff;
		border: 4px solid #212529;
		min-height: 500px;
		display: flex;
		flex-direction: column;
	}

	.evidence-column[data-column="new"] {
		border-color: #92cc41;
		background: #f8fff8;
	}

	.evidence-column[data-column="processing"] {
		border-color: #f7d51d;
		background: #fffef8;
	}

	.evidence-column[data-column="verified"] {
		border-color: #e76e55;
		background: #fff8f8;
	}

	.column-title {
		font-size: 14px;
		text-align: center;
		margin-bottom: 0;
		padding: 10px;
	}

	.column-header {
		padding: 15px;
		border-bottom: 2px solid #212529;
		background: rgba(255, 255, 255, 0.5);
	}

	.column-status {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}

	/* Column Content */
	.column-content {
		flex: 1;
		padding: 15px;
		display: flex;
		flex-direction: column;
		gap: 15px;
	}

	/* Upload Section */
	.upload-section {
		background: rgba(146, 204, 65, 0.1);
		border: 2px dashed #92cc41;
		border-radius: 8px;
		padding: 15px;
		margin-bottom: 15px;
	}

	/* Evidence Items */
	.evidence-items {
		display: flex;
		flex-direction: column;
		gap: 12px;
		min-height: 200px;
	}

	.evidence-item {
		background: #ffffff;
		border: 3px solid #212529;
		padding: 12px;
		transition: transform 0.2s ease, box-shadow 0.2s ease;
		cursor: pointer;
	}

	.evidence-item:hover {
		transform: translateY(-2px);
		box-shadow: 4px 4px 0px #212529;
	}

	.evidence-item.is-dark {
		background: #212529;
		color: white;
		border-color: #f7d51d;
	}

	.evidence-item.is-success {
		background: #92cc41;
		color: white;
		border-color: #ffffff;
	}

	/* GPU Processing Indicator */
	.gpu-indicator {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 10px;
		padding: 5px 10px;
		background: rgba(247, 213, 29, 0.2);
		border: 1px solid #f7d51d;
		border-radius: 4px;
		font-size: 8px;
	}

	/* Vector Score Display */
	.vector-score {
		margin-top: 8px;
		padding: 5px 10px;
		background: rgba(146, 204, 65, 0.2);
		border: 1px solid #92cc41;
		border-radius: 4px;
		text-align: center;
		font-size: 8px;
	}

	/* Active Users */
	.active-users {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: 10px;
	}

	.user-avatar {
		width: 32px;
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 8px;
		background: #92cc41;
		color: white;
		border: 2px solid #212529;
	}

	/* Board Main */
	.board-main {
		flex: 1;
		display: flex;
		flex-direction: column;
	}

	/* Responsive Design */
	@media (max-width: 768px) {
		.evidence-columns {
			grid-template-columns: 1fr;
		}

		.header-content {
			flex-direction: column;
			text-align: center;
		}

		.system-status-grid {
			grid-template-columns: repeat(2, 1fr);
		}

		.perf-stats {
			flex-direction: column;
			text-align: center;
		}
	}

	/* Original Grid Pattern */
	.bg-grid-pattern {
		background-image:
			linear-gradient(rgba(0, 0, 0, 0.1) 1px, transparent 1px),
			linear-gradient(90deg, rgba(0, 0, 0, 0.1) 1px, transparent 1px);
		background-size: 50px 50px;
	}

	:global(.dark) .bg-grid-pattern {
		background-image:
			linear-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255, 255, 255, 0.1) 1px, transparent 1px);
	}

	.detective-board-nes {
		/* Add any additional detective board styles here */
	}
</style>