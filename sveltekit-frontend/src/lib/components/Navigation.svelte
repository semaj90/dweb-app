<script lang="ts">
  import { $derived } from 'svelte';


	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { browser } from '$app/environment';
	import { cn } from '$lib/utils';
	import { 
		Shield, 
		Search, 
		Database, 
		Folder, 
		Eye, 
		Users, 
		BarChart3, 
		Settings,
		Terminal,
		Brain
	} from 'lucide-svelte';

	const navItems = [
		{ href: '/', label: 'COMMAND CENTER', icon: Database },
		{ href: '/evidence', label: 'EVIDENCE', icon: Eye },
		{ href: '/cases', label: 'CASES', icon: Folder },
		{ href: '/persons', label: 'PERSONS', icon: Users },
		{ href: '/analysis', label: 'ANALYSIS', icon: BarChart3 },
		{ href: '/search', label: 'SEARCH', icon: Search },
		{ href: '/terminal', label: 'TERMINAL', icon: Terminal }
	];

	let currentPath = $derived(browser && $page.url ? $page.url.pathname : '/');

	// Optimized navigation with instant transitions
	function handleNavigation(href: string, event?: Event) {
		event?.preventDefault();
		goto(href, { replaceState: false, noScroll: false, keepFocus: false, invalidateAll: false });
	}
</script>

<nav class="yorha-header">
	<div class="flex items-center justify-between">
		<div class="flex items-center space-x-6">
			<div class="flex items-center space-x-3">
				<Shield class="w-8 h-8 yorha-text-accent" />
				<div>
					<h1 class="yorha-header h1">YORHA DETECTIVE</h1>
					<p class="yorha-header subtitle">Investigation Interface</p>
				</div>
			</div>
			
			<nav class="flex items-center space-x-1 ml-8">
				{#each navItems as item}
					<a
						href={item.href}
						onclick={(e) => handleNavigation(item.href, e)}
						class={cn(
							"yorha-nav-item text-sm px-3 py-2 rounded-md transition-all duration-200",
							currentPath === item.href && "active"
						)}
					>
						<item.icon class="w-4 h-4" />
						<span>{item.label}</span>
					</a>
				{/each}
			</nav>
		</div>

		<div class="flex items-center space-x-4">
			<!-- AI Search Button -->
			<button
				class="yorha-btn yorha-btn-secondary"
				onclick={() => {
					// Trigger global FindModal via Ctrl+K event
					window.dispatchEvent(new KeyboardEvent('keydown', {
						key: 'k',
						ctrlKey: true,
						bubbles: true
					}));
				}}
			>
				<Search class="w-4 h-4 mr-2" />
				GLOBAL SEARCH
			</button>

			<div class="flex items-center space-x-2">
				<Brain class="w-4 h-4 yorha-text-accent" />
				<span class="yorha-badge yorha-badge-success">AI ACTIVE</span>
			</div>
			
			<div class="flex items-center space-x-2">
				<div class="yorha-status-indicator yorha-status-online"></div>
				<span class="yorha-text-primary text-sm font-medium">Online</span>
			</div>
		</div>
	</div>
</nav>

<style>
	.modern-header {
		background: var(--yorha-bg-secondary);
		border-bottom: 1px solid var(--yorha-border-primary);
		box-shadow: var(--yorha-shadow-sm);
		sticky: top;
		top: 0;
		z-index: 40;
	}
	
	.header-title {
		font-size: var(--text-lg);
		font-weight: 700;
		color: var(--yorha-accent-gold);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		margin: 0;
	}
	
	.header-subtitle {
		font-size: var(--text-xs);
		color: var(--yorha-text-secondary);
		margin: 0;
		text-transform: uppercase;
		letter-spacing: 0.025em;
	}
	
	.nav-link {
		display: flex;
		align-items: center;
		gap: var(--golden-sm);
		padding: var(--golden-sm) var(--golden-md);
		color: var(--yorha-text-secondary);
		text-decoration: none;
		border-radius: 0.375rem;
		font-weight: 500;
		font-size: var(--text-sm);
		text-transform: uppercase;
		letter-spacing: 0.025em;
		transition: all 200ms ease;
		border: 1px solid transparent;
	}
	
	.nav-link:hover {
		background-color: var(--yorha-bg-hover);
		color: var(--yorha-text-primary);
		border-color: var(--yorha-border-primary);
	}
	
	.nav-link-active {
		background-color: var(--yorha-bg-tertiary);
		color: var(--yorha-accent-gold);
		border-color: var(--yorha-border-accent);
	}
	
	.nav-link:focus-visible {
		outline: 2px solid var(--yorha-accent-gold);
		outline-offset: 2px;
	}
	
	.status-indicator {
		gap: var(--golden-xs);
	}
	
	.status-info {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	
	@media (max-width: 768px) {
		.header-title {
			font-size: var(--text-base);
		}
		
		.header-subtitle {
			display: none;
		}
	}
</style>
