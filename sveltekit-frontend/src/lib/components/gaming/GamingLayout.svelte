<script lang="ts">
	import { page } from '$app/stores';
	import GamingHUD from './GamingHUD.svelte';
	import GamingPanel from './GamingPanel.svelte';
	import GamingButton from './GamingButton.svelte';
	import type { ComponentProps } from 'svelte';
	
	interface GamingLayoutProps {
		showHUD?: boolean;
		title?: string;
		subtitle?: string;
		user?: {
			level: number;
			experience: number;
			maxExperience: number;
		};
		stats?: {
			documentsAnalyzed: number;
			accuracyScore: number;
		};
		navigation?: {
			label: string;
			href: string;
			icon?: string;
		}[];
		children: any;
	}
	
	let { 
		showHUD = true,
		title = "Legal AI System",
		subtitle = "Advanced Document Analysis",
		user = {
			level: 1,
			experience: 750,
			maxExperience: 1000
		},
		stats = {
			documentsAnalyzed: 47,
			accuracyScore: 94.2
		},
		navigation = [
			{ label: 'Dashboard', href: '/', icon: '🏠' },
			{ label: 'Cases', href: '/cases', icon: '📁' },
			{ label: 'Evidence', href: '/evidence', icon: '📋' },
			{ label: 'Documents', href: '/documents', icon: '📄' },
			{ label: 'Analysis', href: '/analysis', icon: '🔍' },
			{ label: 'Reports', href: '/reports', icon: '📊' },
		],
		children
	}: GamingLayoutProps = $props();
	
	let currentPath = $derived($page.url.pathname);
	let sidebarCollapsed = $state(false);
	
	function toggleSidebar() {
		sidebarCollapsed = !sidebarCollapsed;
	}
	
	function isActiveRoute(href: string): boolean {
		return currentPath === href || (href !== '/' && currentPath.startsWith(href));
	}
</script>

<div class="gaming-layout">
	<!-- Gaming HUD -->
	{#if showHUD}
		<GamingHUD 
			userLevel={user.level}
			experience={user.experience}
			maxExperience={user.maxExperience}
			currentCase={$page.data?.currentCase || "CASE-2024-001"}
			documentsAnalyzed={stats.documentsAnalyzed}
			accuracyScore={stats.accuracyScore}
		/>
	{/if}
	
	<!-- Main Container -->
	<div class="main-container" class:hud-offset={showHUD}>
		<!-- Gaming Sidebar -->
		<div class="sidebar" class:collapsed={sidebarCollapsed}>
			<!-- Sidebar Header -->
			<div class="sidebar-header">
				<div class="logo-section">
					<div class="logo-icon">⚖️</div>
					{#if !sidebarCollapsed}
						<div class="logo-text">
							<div class="app-name">{title}</div>
							<div class="app-subtitle">{subtitle}</div>
						</div>
					{/if}
				</div>
				
				<button 
					class="collapse-button"
					onclick={toggleSidebar}
					aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
				>
					{sidebarCollapsed ? '▶' : '◀'}
				</button>
			</div>
			
			<!-- Navigation Menu -->
			<nav class="navigation">
				{#each navigation as navItem}
					<a 
						href={navItem.href}
						class="nav-item"
						class:active={isActiveRoute(navItem.href)}
						data-sveltekit-preload-data="hover"
					>
						<span class="nav-icon">{navItem.icon}</span>
						{#if !sidebarCollapsed}
							<span class="nav-label">{navItem.label}</span>
						{/if}
						
						{#if isActiveRoute(navItem.href)}
							<div class="active-indicator"></div>
						{/if}
					</a>
				{/each}
			</nav>
			
			<!-- Sidebar Footer -->
			<div class="sidebar-footer">
				{#if !sidebarCollapsed}
					<div class="system-status">
						<div class="status-item">
							<div class="status-dot online"></div>
							<span>AI Online</span>
						</div>
						<div class="status-item">
							<div class="status-dot online"></div>
							<span>DB Connected</span>
						</div>
					</div>
				{/if}
			</div>
		</div>
		
		<!-- Main Content Area -->
		<main class="content-area" class:sidebar-collapsed={sidebarCollapsed}>
			{@render children()}
		</main>
	</div>
	
	<!-- Gaming Effects -->
	<div class="scan-overlay"></div>
</div>

<style>
	.gaming-layout {
		position: relative;
		min-height: 100vh;
		font-family: var(--gaming-font-secondary);
	}
	
	.main-container {
		display: flex;
		min-height: 100vh;
		transition: padding-top 0.3s ease;
	}
	
	.main-container.hud-offset {
		padding-top: 120px; /* Adjust based on HUD height */
	}
	
	/* Gaming Sidebar */
	.sidebar {
		position: fixed;
		left: 0;
		top: 0;
		bottom: 0;
		width: 280px;
		background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
		border-right: 2px solid #00ff88;
		box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
		backdrop-filter: blur(10px);
		transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
		z-index: 900;
		overflow-y: auto;
	}
	
	.sidebar.collapsed {
		width: 80px;
	}
	
	.sidebar-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 20px;
		border-bottom: 1px solid rgba(0, 255, 136, 0.3);
		margin-top: 120px; /* Offset for HUD */
	}
	
	.logo-section {
		display: flex;
		align-items: center;
		gap: 12px;
	}
	
	.logo-icon {
		font-size: 24px;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 40px;
		height: 40px;
		background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
		border-radius: 8px;
		box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
	}
	
	.logo-text {
		flex: 1;
		overflow: hidden;
	}
	
	.app-name {
		font-family: var(--gaming-font-primary);
		font-size: 14px;
		font-weight: bold;
		color: #fff;
		text-transform: uppercase;
		letter-spacing: 1px;
		white-space: nowrap;
	}
	
	.app-subtitle {
		font-size: 11px;
		color: #888;
		white-space: nowrap;
	}
	
	.collapse-button {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 32px;
		height: 32px;
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid rgba(0, 255, 136, 0.3);
		border-radius: 6px;
		color: #00ff88;
		cursor: pointer;
		transition: all 0.3s ease;
		font-size: 12px;
	}
	
	.collapse-button:hover {
		background: rgba(0, 255, 136, 0.2);
		border-color: #00ff88;
		box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
	}
	
	/* Navigation */
	.navigation {
		flex: 1;
		padding: 20px 12px;
	}
	
	.nav-item {
		position: relative;
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 12px 16px;
		margin-bottom: 8px;
		color: #ccc;
		text-decoration: none;
		border-radius: 8px;
		border: 1px solid transparent;
		transition: all 0.3s ease;
		overflow: hidden;
	}
	
	.nav-item:hover {
		background: rgba(0, 255, 136, 0.1);
		border-color: rgba(0, 255, 136, 0.3);
		color: #00ff88;
		transform: translateX(4px);
	}
	
	.nav-item.active {
		background: linear-gradient(135deg, rgba(0, 255, 136, 0.2) 0%, rgba(0, 204, 102, 0.1) 100%);
		border-color: #00ff88;
		color: #00ff88;
		box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
	}
	
	.nav-icon {
		font-size: 18px;
		width: 24px;
		text-align: center;
	}
	
	.nav-label {
		font-weight: 500;
		letter-spacing: 0.5px;
		white-space: nowrap;
	}
	
	.active-indicator {
		position: absolute;
		right: 8px;
		top: 50%;
		transform: translateY(-50%);
		width: 4px;
		height: 20px;
		background: linear-gradient(180deg, #00ff88 0%, #00cc66 100%);
		border-radius: 2px;
		box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
	}
	
	/* Sidebar Footer */
	.sidebar-footer {
		padding: 20px;
		border-top: 1px solid rgba(0, 255, 136, 0.3);
	}
	
	.system-status {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	
	.status-item {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: #888;
	}
	
	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		animation: pulse 2s infinite;
	}
	
	.status-dot.online {
		background: #00ff88;
		box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
	}
	
	/* Content Area */
	.content-area {
		flex: 1;
		margin-left: 280px;
		padding: 24px;
		transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
		background: rgba(0, 0, 0, 0.1);
		min-height: 100vh;
	}
	
	.content-area.sidebar-collapsed {
		margin-left: 80px;
	}
	
	/* Gaming Effects */
	.scan-overlay {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		height: 2px;
		background: linear-gradient(90deg, transparent 0%, #00ff88 50%, transparent 100%);
		opacity: 0.6;
		animation: scan-horizontal 4s ease-in-out infinite;
		pointer-events: none;
		z-index: 1100;
	}
	
	/* Animations */
	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
	
	@keyframes scan-horizontal {
		0%, 100% {
			transform: translateX(-100%);
			opacity: 0;
		}
		50% {
			transform: translateX(100vw);
			opacity: 0.6;
		}
	}
	
	/* Responsive Design */
	@media (max-width: 1024px) {
		.main-container.hud-offset {
			padding-top: 100px;
		}
		
		.sidebar {
			width: 260px;
		}
		
		.sidebar.collapsed {
			width: 70px;
		}
		
		.content-area {
			margin-left: 260px;
		}
		
		.content-area.sidebar-collapsed {
			margin-left: 70px;
		}
	}
	
	@media (max-width: 768px) {
		.sidebar {
			transform: translateX(-100%);
			width: 100%;
		}
		
		.sidebar.collapsed {
			width: 100%;
		}
		
		.content-area {
			margin-left: 0;
		}
		
		.content-area.sidebar-collapsed {
			margin-left: 0;
		}
		
		/* Mobile sidebar toggle would need JavaScript implementation */
	}
</style>