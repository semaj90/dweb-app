<script lang="ts">
  import type { Evidence } from '$lib/types';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import ReportEditor from "$lib/components/ReportEditor.svelte";
	import CanvasEditor from "$lib/components/CanvasEditor.svelte";
	import type { Report, CanvasState, CitationPoint } from "$lib/data/types";

	let currentReport: Report | null = null;
	let currentCanvasState: CanvasState | null = null;
	let evidence: Evidence[] = [];
	let citationPoints: CitationPoint[] = [];
	let activeTab: 'editor' | 'canvas' = 'editor';
	let isLoading = false;
	let error = '';

	// Demo case ID - in real app this would come from the route
	const caseId = $page.params.caseId || 'demo-case-123';

	onMount(async () => {
		await loadDemoData();
	});

	async function loadDemoData() {
		try {
			isLoading = true;
			
			// Load sample citation points
			const citationsResponse = await fetch(`/api/citations?caseId=${caseId}`);
			if (citationsResponse.ok) {
				citationPoints = await citationsResponse.json();
}
			// Load sample evidence (mock for now)
			evidence = [
				{
					id: '1',
					caseId,
					criminalId: null,
					title: 'Security Camera Footage',
					description: 'CCTV footage from main entrance',
					evidenceType: 'video',
					fileType: 'video/mp4',
					subType: null,
					fileUrl: null,
					fileName: 'security_footage.mp4',
					fileSize: null,
					mimeType: 'video/mp4',
					hash: 'abc123def456',
					tags: [],
					chainOfCustody: [],
					collectedAt: null,
					collectedBy: null,
					location: null,
					labAnalysis: {},
					aiAnalysis: {},
					aiTags: [],
					aiSummary: null,
					summary: null,
					isAdmissible: true,
					confidentialityLevel: 'standard',
					canvasPosition: {},
					uploadedBy: '1',
					uploadedAt: new Date(),
					updatedAt: new Date()
				},
				{
					id: '2',
					caseId,
					criminalId: null,
					title: 'Witness Statement - John Doe',
					description: 'Eyewitness account of the incident',
					evidenceType: 'document',
					fileType: 'application/pdf',
					subType: null,
					fileUrl: null,
					fileName: 'witness_statement.pdf',
					fileSize: null,
					mimeType: 'application/pdf',
					hash: 'def456ghi789',
					tags: [],
					chainOfCustody: [],
					collectedAt: null,
					collectedBy: null,
					location: null,
					labAnalysis: {},
					aiAnalysis: {},
					aiTags: [],
					aiSummary: null,
					summary: null,
					isAdmissible: true,
					confidentialityLevel: 'standard',
					canvasPosition: {},
					uploadedBy: '1',
					uploadedAt: new Date(),
					updatedAt: new Date()
				},
				{
					id: '3',
					caseId,
					criminalId: null,
					title: 'Physical Evidence - Weapon',
					description: 'Photograph of recovered weapon',
					evidenceType: 'photo',
					fileType: 'image/jpeg',
					subType: null,
					fileUrl: null,
					fileName: 'weapon_photo.jpg',
					fileSize: null,
					mimeType: 'image/jpeg',
					hash: 'ghi789jkl012',
					tags: [],
					chainOfCustody: [],
					collectedAt: null,
					collectedBy: null,
					location: null,
					labAnalysis: {},
					aiAnalysis: {},
					aiTags: [],
					aiSummary: null,
					summary: null,
					isAdmissible: true,
					confidentialityLevel: 'standard',
					canvasPosition: {},
					uploadedBy: '1',
					uploadedAt: new Date(),
					updatedAt: new Date()
}
			];

		} catch (err) {
			console.error('Failed to load demo data:', err);
			error = 'Failed to load demo data';
		} finally {
			isLoading = false;
}}
	async function handleReportSave(report: Report) {
		try {
			currentReport = report;
			console.log('Report saved:', report);
		} catch (err) {
			console.error('Failed to save report:', err);
			error = 'Failed to save report';
}}
	async function handleCanvasSave(canvasState: CanvasState) {
		try {
			currentCanvasState = canvasState;
			console.log('Canvas saved:', canvasState);
		} catch (err) {
			console.error('Failed to save canvas:', err);
			error = 'Failed to save canvas';
}}
	function createNewReport() {
		currentReport = null;
		activeTab = 'editor';
}
	function createNewCanvas() {
		currentCanvasState = null;
		activeTab = 'canvas';
}
</script>

<svelte:head>
	<title>Report Builder - Prosecutor's Case Management</title>
	<meta name="description" content="AI-powered report builder for legal case analysis" />
</svelte:head>

<div class="container mx-auto px-4">
	<!-- Header -->
	<header class="container mx-auto px-4">
		<div class="container mx-auto px-4">
			<h1>📝 Report Builder</h1>
			<p class="container mx-auto px-4">AI-powered case analysis and report generation</p>
			
			<div class="container mx-auto px-4">
				<button class="container mx-auto px-4" on:click={() => createNewReport()}>
					📄 New Report
				</button>
				<button class="container mx-auto px-4" on:click={() => createNewCanvas()}>
					🎨 New Canvas
				</button>
			</div>
		</div>
	</header>

	<!-- Error Message -->
	{#if error}
		<div class="container mx-auto px-4">
			❌ {error}
			<button on:click={() => error = ''} class="container mx-auto px-4">×</button>
		</div>
	{/if}

	<!-- Loading State -->
	{#if isLoading}
		<div class="container mx-auto px-4">
			<div class="container mx-auto px-4">⏳</div>
			<p>Loading demo data...</p>
		</div>
	{:else}
		<!-- Tab Navigation -->
		<div class="container mx-auto px-4">
			<button 
				class="container mx-auto px-4"
				class:active={activeTab === 'editor'}
				on:click={() => activeTab = 'editor'}
			>
				📝 Report Editor
			</button>
			<button 
				class="container mx-auto px-4"
				class:active={activeTab === 'canvas'}
				on:click={() => activeTab = 'canvas'}
			>
				🎨 Interactive Canvas
			</button>
		</div>

		<!-- Main Content -->
		<main class="container mx-auto px-4">
			{#if activeTab === 'editor'}
				<!-- Report Editor Tab -->
				<div class="container mx-auto px-4">
					<div class="container mx-auto px-4">
						<h2>Prosecutor's Report</h2>
						<p>Write, edit, and analyze case reports with AI assistance</p>
					</div>
					
					<ReportEditor
						report={currentReport}
						{caseId}
						onSave={handleReportSave}
						autoSaveEnabled={true}
					/>
				</div>
			{:else if activeTab === 'canvas'}
				<!-- Canvas Editor Tab -->
				<div class="container mx-auto px-4">
					<div class="container mx-auto px-4">
						<h2>Interactive Evidence Canvas</h2>
						<p>Visualize evidence, create diagrams, and annotate with AI insights</p>
					</div>
					
					<CanvasEditor
						canvasState={currentCanvasState}
						reportId={currentReport?.id || 'temp-report-id'}
						{evidence}
						{citationPoints}
						onSave={handleCanvasSave}
					/>
				</div>
			{/if}
		</main>

		<!-- Sidebar with Features Overview -->
		<aside class="container mx-auto px-4">
			<div class="container mx-auto px-4">
				<h3>🤖 AI Features</h3>
				<ul class="container mx-auto px-4">
					<li>✨ Auto-complete suggestions</li>
					<li>📊 Case analysis insights</li>
					<li>🔍 Citation recommendations</li>
					<li>📝 Content summarization</li>
				</ul>
			</div>

			<div class="container mx-auto px-4">
				<h3>📚 Citation Library</h3>
				<p class="container mx-auto px-4">{citationPoints.length} citations available</p>
				<div class="container mx-auto px-4">
					{#each citationPoints.slice(0, 3) as citation}
						<div class="container mx-auto px-4">
							<div class="container mx-auto px-4">{citation.source}</div>
							<div class="container mx-auto px-4">{citation.text.substring(0, 60)}...</div>
						</div>
					{/each}
				</div>
			</div>

			<div class="container mx-auto px-4">
				<h3>📋 Evidence Repository</h3>
				<p class="container mx-auto px-4">{evidence.length} pieces of evidence</p>
				<div class="container mx-auto px-4">
					{#each evidence as item}
						<div class="container mx-auto px-4">
							<div class="container mx-auto px-4">{item.title}</div>
							<div class="container mx-auto px-4">{item.fileType}</div>
						</div>
					{/each}
				</div>
			</div>

			<div class="container mx-auto px-4">
				<h3>⚡ Quick Actions</h3>
				<div class="container mx-auto px-4">
					<button class="container mx-auto px-4">📤 Export PDF</button>
					<button class="container mx-auto px-4">💾 Save Template</button>
					<button class="container mx-auto px-4">🔄 Sync Offline</button>
				</div>
			</div>
		</aside>
	{/if}
</div>

<style>
  /* @unocss-include */
	.report-builder-page {
		min-height: 100vh;
		background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
		display: flex;
		flex-direction: column;
}
	.page-header {
		background: white;
		border-bottom: 1px solid #e2e8f0;
		box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
}
	.header-content {
		max-width: 1200px;
		margin: 0 auto;
		padding: 24px;
		display: flex;
		align-items: center;
		justify-content: space-between;
}
	.header-content h1 {
		font-size: 28px;
		font-weight: 700;
		color: #1e293b;
		margin: 0;
}
	.subtitle {
		font-size: 14px;
		color: #64748b;
		margin: 4px 0 0 0;
}
	.header-actions {
		display: flex;
		gap: 12px;
}
	.btn-secondary {
		padding: 10px 20px;
		background: white;
		border: 1px solid #d1d5db;
		border-radius: 8px;
		color: #374151;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s;
}
	.btn-secondary:hover {
		background: #f9fafb;
		border-color: #9ca3af;
}
	.error-message {
		background: #fee2e2;
		border: 1px solid #fecaca;
		color: #dc2626;
		padding: 12px 16px;
		margin: 16px 24px;
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: space-between;
}
	.close-error {
		background: none;
		border: none;
		color: #dc2626;
		font-size: 18px;
		cursor: pointer;
		padding: 0;
		margin-left: 12px;
}
	.loading-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 64px;
		color: #64748b;
}
	.loading-spinner {
		font-size: 32px;
		animation: spin 1s linear infinite;
		margin-bottom: 16px;
}
	@keyframes spin {
		from { transform: rotate(0deg); }
		to { transform: rotate(360deg); }
}
	.tab-navigation {
		background: white;
		border-bottom: 1px solid #e2e8f0;
		padding: 0 24px;
		display: flex;
		gap: 2px;
}
	.tab-btn {
		padding: 12px 24px;
		background: none;
		border: none;
		border-bottom: 3px solid transparent;
		color: #64748b;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s;
}
	.tab-btn.active {
		color: #3b82f6;
		border-bottom-color: #3b82f6;
}
	.tab-btn:hover:not(.active) {
		color: #374151;
		background: #f8fafc;
}
	.main-content {
		flex: 1;
		display: grid;
		grid-template-columns: 1fr 300px;
		gap: 24px;
		padding: 24px;
		max-width: 1400px;
		margin: 0 auto;
		width: 100%;
}
	.editor-section,
	.canvas-section {
		background: white;
		border-radius: 12px;
		box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
		overflow: hidden;
}
	.section-header {
		padding: 24px 24px 16px 24px;
		border-bottom: 1px solid #e2e8f0;
		background: #f8fafc;
}
	.section-header h2 {
		font-size: 20px;
		font-weight: 600;
		color: #1e293b;
		margin: 0 0 4px 0;
}
	.section-header p {
		color: #64748b;
		font-size: 14px;
		margin: 0;
}
	.features-sidebar {
		display: flex;
		flex-direction: column;
		gap: 20px;
}
	.sidebar-section {
		background: white;
		border-radius: 12px;
		padding: 20px;
		box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
	.sidebar-section h3 {
		font-size: 16px;
		font-weight: 600;
		color: #1e293b;
		margin: 0 0 12px 0;
}
	.feature-list {
		list-style: none;
		padding: 0;
		margin: 0;
}
	.feature-list li {
		padding: 6px 0;
		color: #64748b;
		font-size: 14px;
}
	.citation-count,
	.evidence-count {
		font-size: 12px;
		color: #3b82f6;
		font-weight: 500;
		margin-bottom: 12px;
}
	.citation-preview,
	.evidence-preview {
		display: flex;
		flex-direction: column;
		gap: 8px;
}
	.citation-item,
	.evidence-item {
		padding: 8px;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		background: #f8fafc;
}
	.citation-source,
	.evidence-title {
		font-size: 12px;
		font-weight: 500;
		color: #374151;
		margin-bottom: 2px;
}
	.citation-text,
	.evidence-type {
		font-size: 11px;
		color: #64748b;
		line-height: 1.3;
}
	.quick-actions {
		display: flex;
		flex-direction: column;
		gap: 8px;
}
	.action-btn {
		padding: 8px 12px;
		background: #f1f5f9;
		border: 1px solid #e2e8f0;
		border-radius: 6px;
		color: #374151;
		font-size: 12px;
		cursor: pointer;
		transition: all 0.2s;
		text-align: left;
}
	.action-btn:hover {
		background: #e2e8f0;
}
	@media (max-width: 1024px) {
		.main-content {
			grid-template-columns: 1fr;
			gap: 16px;
}
		.features-sidebar {
			order: -1;
			flex-direction: row;
			overflow-x: auto;
			padding-bottom: 8px;
}
		.sidebar-section {
			min-width: 250px;
}}
	@media (max-width: 768px) {
		.header-content {
			flex-direction: column;
			align-items: flex-start;
			gap: 16px;
}
		.header-actions {
			width: 100%;
			justify-content: stretch;
}
		.header-actions button {
			flex: 1;
}
		.tab-navigation {
			overflow-x: auto;
}
		.main-content {
			padding: 16px;
}
		.features-sidebar {
			flex-direction: column;
}}
</style>
