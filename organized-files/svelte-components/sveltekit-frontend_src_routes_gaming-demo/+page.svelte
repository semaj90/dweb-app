<script lang="ts">
	import GamingLayout from '$lib/components/gaming/GamingLayout.svelte';
	import GamingPanel from '$lib/components/gaming/GamingPanel.svelte';
	import GamingButton from '$lib/components/gaming/GamingButton.svelte';
	
	let currentCase = $state("CASE-2024-001");
	let analysisProgress = $state(75);
	let documents = $state([
		{ id: 1, name: "Contract_Agreement.pdf", status: "analyzed", confidence: 94.2 },
		{ id: 2, name: "Evidence_Report.docx", status: "processing", confidence: 0 },
		{ id: 3, name: "Witness_Statement.pdf", status: "queued", confidence: 0 }
	]);
	
	let isAnalyzing = $state(false);
	
	function startAnalysis() {
		isAnalyzing = true;
		setTimeout(() => {
			isAnalyzing = false;
			analysisProgress = Math.min(100, analysisProgress + 15);
		}, 3000);
	}
	
	function generateReport() {
		alert("Report generation started! Check the Reports section in 5 minutes.");
	}
</script>

<svelte:head>
	<title>Legal AI Gaming Demo</title>
	<meta name="description" content="Gaming-style interface for legal AI document analysis" />
</svelte:head>

<GamingLayout
	title="YoRHa Legal AI"
	subtitle="Evidence Analysis System"
	user={{
		level: 8,
		experience: 2450,
		maxExperience: 3000
	}}
	stats={{
		documentsAnalyzed: 156,
		accuracyScore: 96.8
	}}
>
	<!-- Main Dashboard Content -->
	<div class="gaming-dashboard">
		<!-- Page Header -->
		<div class="page-header">
			<h1 class="page-title">Mission Control Dashboard</h1>
			<p class="page-subtitle">Advanced Legal Document Analysis & Evidence Processing</p>
		</div>
		
		<!-- Control Grid -->
		<div class="control-grid">
			<!-- Active Case Panel -->
			<GamingPanel 
				title="Active Case"
				subtitle="Current Investigation"
				variant="primary"
				borderGlow={true}
				scanEffect={true}
			>
				<div class="case-info">
					<div class="case-header">
						<div class="case-id">{currentCase}</div>
						<div class="case-status active">ACTIVE</div>
					</div>
					
					<div class="case-details">
						<div class="detail-item">
							<span class="label">Priority:</span>
							<span class="value high">HIGH</span>
						</div>
						<div class="detail-item">
							<span class="label">Type:</span>
							<span class="value">Corporate Litigation</span>
						</div>
						<div class="detail-item">
							<span class="label">Assigned:</span>
							<span class="value">Agent Anderson</span>
						</div>
					</div>
					
					<div class="progress-section">
						<div class="progress-label">Analysis Progress</div>
						<div class="progress-bar">
							<div class="progress-fill" style="width: {analysisProgress}%"></div>
						</div>
						<div class="progress-text">{analysisProgress}% Complete</div>
					</div>
				</div>
			</GamingPanel>
			
			<!-- Document Processing Panel -->
			<GamingPanel 
				title="Document Queue"
				subtitle="AI Processing Status"
				variant="secondary"
				borderGlow={true}
			>
				<div class="document-queue">
					{#each documents as doc}
						<div class="document-item" class:processing={doc.status === 'processing'}>
							<div class="doc-icon">📄</div>
							<div class="doc-details">
								<div class="doc-name">{doc.name}</div>
								<div class="doc-status {doc.status}">
									{doc.status.toUpperCase()}
									{#if doc.confidence > 0}
										- {doc.confidence}% Confidence
									{/if}
								</div>
							</div>
							<div class="doc-actions">
								{#if doc.status === 'analyzed'}
									<div class="status-indicator success"></div>
								{:else if doc.status === 'processing'}
									<div class="status-indicator processing"></div>
								{:else}
									<div class="status-indicator queued"></div>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			</GamingPanel>
			
			<!-- AI Analysis Panel -->
			<GamingPanel 
				title="AI Analysis Engine"
				subtitle="Neural Network Status"
				variant="success"
				borderGlow={true}
				scanEffect={true}
			>
				<div class="ai-controls">
					<div class="ai-status">
						<div class="status-grid">
							<div class="status-item">
								<div class="status-label">Legal-BERT</div>
								<div class="status-value online">ONLINE</div>
							</div>
							<div class="status-item">
								<div class="status-label">Gemma3</div>
								<div class="status-value online">READY</div>
							</div>
							<div class="status-item">
								<div class="status-label">Vector DB</div>
								<div class="status-value online">CONNECTED</div>
							</div>
							<div class="status-item">
								<div class="status-label">GPU Usage</div>
								<div class="status-value">67%</div>
							</div>
						</div>
					</div>
					
					<div class="ai-actions">
						<GamingButton 
							variant="primary" 
							size="lg" 
							loading={isAnalyzing}
							glowEffect={true}
							onclick={startAnalysis}
						>
							{isAnalyzing ? 'Analyzing...' : 'Start Deep Analysis'}
						</GamingButton>
						
						<GamingButton 
							variant="secondary" 
							size="md"
							onclick={generateReport}
						>
							Generate Report
						</GamingButton>
					</div>
				</div>
			</GamingPanel>
			
			<!-- System Stats Panel -->
			<GamingPanel 
				title="System Metrics"
				subtitle="Performance Monitor"
				variant="warning"
				minimizable={true}
			>
				<div class="metrics-grid">
					<div class="metric-item">
						<div class="metric-icon">🚀</div>
						<div class="metric-content">
							<div class="metric-value">156</div>
							<div class="metric-label">Documents Processed</div>
						</div>
					</div>
					
					<div class="metric-item">
						<div class="metric-icon">⏱️</div>
						<div class="metric-content">
							<div class="metric-value">2.3s</div>
							<div class="metric-label">Avg Response Time</div>
						</div>
					</div>
					
					<div class="metric-item">
						<div class="metric-icon">🎯</div>
						<div class="metric-content">
							<div class="metric-value">96.8%</div>
							<div class="metric-label">Accuracy Score</div>
						</div>
					</div>
					
					<div class="metric-item">
						<div class="metric-icon">💾</div>
						<div class="metric-content">
							<div class="metric-value">4.2GB</div>
							<div class="metric-label">Memory Usage</div>
						</div>
					</div>
				</div>
			</GamingPanel>
		</div>
		
		<!-- Action Bar -->
		<div class="action-bar">
			<GamingButton variant="success" size="lg">
				🔍 Search Evidence
			</GamingButton>
			
			<GamingButton variant="primary" size="lg">
				📋 New Case
			</GamingButton>
			
			<GamingButton variant="warning" size="lg">
				⚖️ Legal Analysis
			</GamingButton>
			
			<GamingButton variant="danger" size="lg">
				🚨 Emergency Protocol
			</GamingButton>
		</div>
	</div>
</GamingLayout>

<style>
	.gaming-dashboard {
		max-width: 1400px;
		margin: 0 auto;
		padding: 0;
	}
	
	/* Page Header */
	.page-header {
		margin-bottom: 32px;
		text-align: center
	}
	
	.page-title {
		font-family: var(--gaming-font-primary);
		font-size: 2.5rem;
		font-weight: 900;
		text-transform: uppercase
		letter-spacing: 2px;
		color: #00ff88;
		text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
		margin: 0 0 8px 0;
	}
	
	.page-subtitle {
		font-size: 1.1rem;
		color: #888;
		margin: 0;
		letter-spacing: 1px;
	}
	
	/* Control Grid */
	.control-grid {
		display: grid
		grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
		gap: 24px;
		margin-bottom: 32px;
	}
	
	/* Case Info */
	.case-info {
		display: flex
		flex-direction: column
		gap: 16px;
	}
	
	.case-header {
		display: flex
		justify-content: space-between;
		align-items: center
	}
	
	.case-id {
		font-family: var(--gaming-font-primary);
		font-size: 1.2rem;
		font-weight: bold
		color: #00ff88;
		text-shadow: 0 0 8px rgba(0, 255, 136, 0.3);
	}
	
	.case-status {
		padding: 4px 12px;
		border-radius: 12px;
		font-size: 12px;
		font-weight: bold
		text-transform: uppercase
		letter-spacing: 1px;
	}
	
	.case-status.active {
		background: rgba(0, 255, 136, 0.2);
		border: 1px solid #00ff88;
		color: #00ff88;
	}
	
	.case-details {
		display: flex
		flex-direction: column
		gap: 8px;
	}
	
	.detail-item {
		display: flex
		justify-content: space-between;
		align-items: center
	}
	
	.detail-item .label {
		color: #888;
		font-size: 14px;
	}
	
	.detail-item .value {
		color: #fff;
		font-weight: 500;
	}
	
	.detail-item .value.high {
		color: #ffaa00;
		font-weight: bold
	}
	
	.progress-section {
		display: flex
		flex-direction: column
		gap: 8px;
	}
	
	.progress-label {
		font-size: 14px;
		color: #888;
		text-transform: uppercase
		letter-spacing: 1px;
	}
	
	.progress-bar {
		width: 100%;
		height: 8px;
		background: rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		overflow: hidden
		border: 1px solid #333;
	}
	
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, #00ff88, #00cc66);
		border-radius: 4px;
		transition: width 0.5s ease;
		box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
	}
	
	.progress-text {
		font-size: 12px;
		color: #00ff88;
		text-align: center
	}
	
	/* Document Queue */
	.document-queue {
		display: flex
		flex-direction: column
		gap: 12px;
	}
	
	.document-item {
		display: flex
		align-items: center
		gap: 12px;
		padding: 12px;
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 6px;
		transition: all 0.3s ease;
	}
	
	.document-item:hover {
		background: rgba(255, 255, 255, 0.1);
		border-color: rgba(0, 255, 136, 0.3);
	}
	
	.document-item.processing {
		border-color: #ffaa00;
		box-shadow: 0 0 15px rgba(255, 170, 0, 0.2);
		animation: processing-pulse 2s ease-in-out infinite;
	}
	
	.doc-icon {
		font-size: 20px;
	}
	
	.doc-details {
		flex: 1;
	}
	
	.doc-name {
		font-weight: 500;
		color: #fff;
		margin-bottom: 4px;
	}
	
	.doc-status {
		font-size: 12px;
		text-transform: uppercase
		letter-spacing: 0.5px;
	}
	
	.doc-status.analyzed {
		color: #00ff88;
	}
	
	.doc-status.processing {
		color: #ffaa00;
	}
	
	.doc-status.queued {
		color: #888;
	}
	
	.status-indicator {
		width: 12px;
		height: 12px;
		border-radius: 50%;
	}
	
	.status-indicator.success {
		background: #00ff88;
		box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
	}
	
	.status-indicator.processing {
		background: #ffaa00;
		box-shadow: 0 0 8px rgba(255, 170, 0, 0.5);
		animation: pulse 1s ease-in-out infinite;
	}
	
	.status-indicator.queued {
		background: #888;
	}
	
	/* AI Controls */
	.ai-controls {
		display: flex
		flex-direction: column
		gap: 20px;
	}
	
	.status-grid {
		display: grid
		grid-template-columns: repeat(2, 1fr);
		gap: 16px;
	}
	
	.status-item {
		text-align: center
	}
	
	.status-label {
		font-size: 12px;
		color: #888;
		margin-bottom: 4px;
		text-transform: uppercase
		letter-spacing: 0.5px;
	}
	
	.status-value {
		font-weight: bold
		font-size: 14px;
	}
	
	.status-value.online {
		color: #00ff88;
		text-shadow: 0 0 8px rgba(0, 255, 136, 0.3);
	}
	
	.ai-actions {
		display: flex
		flex-direction: column
		gap: 12px;
	}
	
	/* Metrics Grid */
	.metrics-grid {
		display: grid
		grid-template-columns: repeat(2, 1fr);
		gap: 16px;
	}
	
	.metric-item {
		display: flex
		align-items: center
		gap: 12px;
		padding: 12px;
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 6px;
	}
	
	.metric-icon {
		font-size: 24px;
	}
	
	.metric-content {
		text-align: center
	}
	
	.metric-value {
		font-size: 18px;
		font-weight: bold
		color: #00ff88;
		margin-bottom: 2px;
	}
	
	.metric-label {
		font-size: 11px;
		color: #888;
		text-transform: uppercase
		letter-spacing: 0.5px;
	}
	
	/* Action Bar */
	.action-bar {
		display: flex
		justify-content: center
		gap: 16px;
		margin-top: 32px;
		flex-wrap: wrap
	}
	
	/* Animations */
	@keyframes processing-pulse {
		0%, 100% {
			box-shadow: 0 0 15px rgba(255, 170, 0, 0.2);
		}
		50% {
			box-shadow: 0 0 25px rgba(255, 170, 0, 0.4);
		}
	}
	
	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
	
	/* Responsive Design */
	@media (max-width: 768px) {
		.control-grid {
			grid-template-columns: 1fr;
		}
		
		.action-bar {
			flex-direction: column
			align-items: center
		}
		
		.page-title {
			font-size: 2rem;
		}
		
		.status-grid,
		.metrics-grid {
			grid-template-columns: 1fr;
		}
	}
</style>