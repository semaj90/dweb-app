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
		Brain,
		User,
		LogOut,
		LogIn,
		UserPlus
	} from 'lucide-svelte';
	import UniversalSearchBar from '$lib/components/search/UniversalSearchBar.svelte';
	import { authStore, useAuth } from '$lib/stores/auth-store.svelte.js';
	import { onMount } from 'svelte';

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

	// Authentication state using the auth store
	const auth = useAuth();
	let showMobileMenu = $state(false);
	let showSearchModal = $state(false);
	let userAvatarUrl = $state<string | null>(null);

	// Generate user avatar URL from MinIO or use initials
	$effect(() => {
		if (auth.user) {
			// Try to load avatar from MinIO storage
			if (auth.user.avatar_url) {
				userAvatarUrl = auth.user.avatar_url;
			} else {
				// Generate initials-based avatar URL (you could also use a service like UI Avatars)
				const initials = `${auth.user.firstName?.[0] || ''}${auth.user.lastName?.[0] || ''}`.toUpperCase();
				userAvatarUrl = `https://ui-avatars.com/api/?name=${encodeURIComponent(initials)}&background=random&color=fff&size=40`;
			}
		} else {
			userAvatarUrl = null;
		}
	});

	// Optimized navigation with instant transitions
	function handleNavigation(href: string, event?: Event) {
		event?.preventDefault();
		goto(href, { replaceState: false, noScroll: false, keepFocus: false, invalidateAll: false });
	}

	function handleSearchSelect(event: CustomEvent) {
		const { result } = event.detail;
		// Navigate to the search result
		if (result.metadata?.url) {
			handleNavigation(result.metadata.url);
		} else {
			// Generate URL based on result type
			const typeRoutes = {
				case: '/cases',
				evidence: '/evidence',
				criminal: '/persons',
				document: '/documents',
				precedent: '/analysis'
			};
			const baseRoute = typeRoutes[result.type] || '/search';
			handleNavigation(`${baseRoute}?id=${result.id}`);
		}
		showSearchModal = false;
	}

	function toggleSearchModal() {
		showSearchModal = !showSearchModal;
	}

	function handleAuth(action: 'login' | 'register' | 'logout') {
		switch (action) {
			case 'login':
				goto('/auth/login');
				break;
			case 'register':
				goto('/auth/register');
				break;
			case 'logout':
				auth.logout();
				break;
		}
	}
</script>

<nav class="nes-legal-header yorha-3d-panel">
	<div class="container-nes-main">
		<div class="nes-header-left">
			<div class="nes-logo-section">
				<Shield class="yorha-3d-button neural-sprite-active w-8 h-8" />
				<div class="nes-title-group">
					<h1 class="nes-legal-title">YORHA DETECTIVE</h1>
					<p class="nes-legal-subtitle">Investigation Interface</p>
				</div>
			</div>

			<nav class="nes-nav-section">
				{#each navItems as item}
					<a
						href={item.href}
						on:onclick={(e) => handleNavigation(item.href, e)}
						class={cn(
							"nes-legal-priority-medium yorha-3d-button",
							currentPath === item.href && "nes-legal-priority-high"
						)}
					>
						<item.icon class="w-4 h-4" />
						<span>{item.label}</span>
					</a>
				{/each}
			</nav>
		</div>

		<div class="nes-header-center">
			<!-- Universal Search Bar (shown for authenticated users) -->
			{#if auth.isAuthenticated}
				<div class="search-container">
					<UniversalSearchBar
						placeholder="Search cases, evidence, documents..."
						theme="yorha"
						showFilters={false}
						maxResults={15}
						select={handleSearchSelect}
					/>
				</div>
			{/if}
		</div>

		<div class="nes-header-right">
			<!-- Authentication Area -->
			<div class="auth-section">
				{#if auth.isAuthenticated}
					<!-- Authenticated User Section -->
					<div class="user-section">
						<!-- AI Search Button -->
						<button
							class="nes-legal-priority-high yorha-3d-button neural-sprite-active search-btn"
							on:onclick={toggleSearchModal}
							title="Advanced Search (Ctrl+K)"
						>
							<Search class="w-4 h-4" />
							<span class="hidden md:inline ml-2">SEARCH</span>
						</button>

						<!-- User Avatar & Profile -->
						<div class="user-profile-section">
							<button
								class="user-avatar-btn yorha-3d-button"
								on:onclick={() => goto('/profile')}
								title={`${auth.user?.firstName} ${auth.user?.lastName}`}
							>
								{#if userAvatarUrl}
									<img
										src={userAvatarUrl}
										alt="User Avatar"
										class="user-avatar"
										error={() => {
											// Fallback to initials if image fails to load
											userAvatarUrl = null;
										}}
									/>
								{:else}
									<User class="w-6 h-6 text-yellow-400" />
								{/if}
							</button>

							<div class="user-info hidden lg:block">
								<span class="user-name text-yellow-400">
									{auth.user?.firstName || auth.user?.email || 'User'}
								</span>
								<span class="user-role text-xs opacity-70">
									{auth.user?.role || 'User'}
								</span>
							</div>
						</div>

						<!-- Logout Button -->
						<button
							class="nes-legal-priority-medium yorha-3d-button logout-btn"
							on:onclick={() => handleAuth('logout')}
							title="Logout"
						>
							<LogOut class="w-4 h-4" />
							<span class="hidden md:inline ml-2">LOGOUT</span>
						</button>
					</div>
				{:else}
					<!-- Unauthenticated User Section -->
					<div class="guest-section">
						<!-- Quick Search (limited) -->
						<button
							class="nes-legal-priority-medium yorha-3d-button search-btn"
							on:onclick={toggleSearchModal}
							title="Public Search"
						>
							<Search class="w-4 h-4" />
							<span class="hidden md:inline ml-2">SEARCH</span>
						</button>

						<!-- Login Button -->
						<button
							class="nes-legal-priority-high yorha-3d-button login-btn"
							on:onclick={() => handleAuth('login')}
						>
							<LogIn class="w-4 h-4" />
							<span class="hidden md:inline ml-2">LOGIN</span>
						</button>

						<!-- Register Button -->
						<button
							class="nes-legal-priority-critical yorha-3d-button register-btn"
							on:onclick={() => handleAuth('register')}
						>
							<UserPlus class="w-4 h-4" />
							<span class="hidden md:inline ml-2">REGISTER</span>
						</button>
					</div>
				{/if}
			</div>

			<!-- System Status -->
			<div class="system-status">
				<div class="nes-status-group">
					<Brain class="w-4 h-4 neural-sprite-active" />
					<span class="nes-legal-priority-critical">AI ACTIVE</span>
				</div>

				<div class="nes-status-group">
					<div class="neural-sprite-loading nes-status-online"></div>
					<span class="nes-legal-priority-high">ONLINE</span>
				</div>
			</div>
		</div>
	</div>
</nav>

<!-- Advanced Search Modal -->
{#if showSearchModal}
	<div class="search-modal-overlay" on:onclick={(e) => e.target === e.currentTarget && (showSearchModal = false)}>
		<div class="search-modal yorha-3d-panel">
			<div class="search-modal-header">
				<h3 class="text-lg font-bold text-yellow-400">
					{auth.isAuthenticated ? 'Advanced Legal Search' : 'Public Search'}
				</h3>
				<button
					class="close-btn"
					on:onclick={() => showSearchModal = false}
					aria-label="Close search"
				>
					×
				</button>
			</div>

			<div class="search-modal-content">
				<UniversalSearchBar
					placeholder={auth.isAuthenticated ?
						"Search cases, evidence, documents, precedents..." :
						"Search public documents and information..."
					}
					theme="yorha"
					showFilters={auth.isAuthenticated}
					maxResults={auth.isAuthenticated ? 50 : 10}
					select={handleSearchSelect}
				/>

				{#if !auth.isAuthenticated}
					<div class="auth-prompt">
						<p class="text-sm opacity-70 mt-4">
							Sign in for full access to cases, evidence, and advanced legal search features.
						</p>
						<div class="auth-buttons mt-3">
							<button
								class="nes-legal-priority-high yorha-3d-button"
								on:onclick={() => { handleAuth('login'); showSearchModal = false; }}
							>
								Login for Full Access
							</button>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	/* Navigation Layout */
	.container-nes-main {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.75rem 1.5rem;
		gap: 1rem;
	}

	.nes-header-left {
		display: flex;
		align-items: center;
		gap: 2rem;
		flex: 1;
	}

	.nes-header-center {
		flex: 2;
		max-width: 600px;
		min-width: 0;
	}

	.nes-header-right {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex: 1;
		justify-content: flex-end;
	}

	/* Search Container */
	.search-container {
		width: 100%;
		max-width: 500px;
	}

	/* Authentication Section */
	.auth-section {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.user-section {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.guest-section {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.user-profile-section {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.user-avatar-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 2.5rem;
		height: 2.5rem;
		border-radius: 50%;
		overflow: hidden;
		border: 2px solid rgba(255, 255, 0, 0.3);
		transition: all 0.2s ease;
	}

	.user-avatar-btn:hover {
		border-color: rgba(255, 255, 0, 0.6);
		transform: scale(1.05);
	}

	.user-avatar {
		width: 100%;
		height: 100%;
		object-fit: cover;
		border-radius: 50%;
	}

	.user-info {
		display: flex;
		flex-direction: column;
		gap: 0.125rem;
	}

	.user-name {
		font-weight: 600;
		font-size: 0.875rem;
		line-height: 1.25;
	}

	.user-role {
		font-size: 0.75rem;
		opacity: 0.7;
		text-transform: uppercase;
	}

	/* System Status */
	.system-status {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-left: 1rem;
		padding-left: 1rem;
		border-left: 1px solid rgba(255, 255, 0, 0.2);
	}

	/* Search Modal */
	.search-modal-overlay {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: rgba(0, 0, 0, 0.8);
		backdrop-filter: blur(4px);
		display: flex;
		align-items: flex-start;
		justify-content: center;
		padding-top: 10vh;
		z-index: 1000;
	}

	.search-modal {
		width: 90%;
		max-width: 800px;
		max-height: 80vh;
		overflow-y: auto;
		border-radius: 8px;
	}

	.search-modal-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 1.5rem 2rem;
		border-bottom: 1px solid rgba(255, 255, 0, 0.2);
	}

	.close-btn {
		background: none;
		border: none;
		color: #ffff00;
		font-size: 1.5rem;
		cursor: pointer;
		padding: 0.25rem;
		line-height: 1;
		transition: opacity 0.2s ease;
	}

	.close-btn:hover {
		opacity: 0.7;
	}

	.search-modal-content {
		padding: 2rem;
	}

	.auth-prompt {
		text-align: center;
		padding: 1rem 0;
	}

	.auth-buttons {
		display: flex;
		justify-content: center;
		gap: 0.75rem;
	}

	/* Responsive Design */
	@media (max-width: 1024px) {
		.nes-header-center {
			display: none;
		}

		.system-status {
			display: none;
		}
	}

	@media (max-width: 768px) {
		.container-nes-main {
			padding: 0.5rem 1rem;
		}

		.nes-header-left {
			gap: 1rem;
		}

		.user-profile-section {
			gap: 0.5rem;
		}

		.search-modal {
			width: 95%;
			margin: 0 auto;
		}

		.search-modal-content {
			padding: 1rem;
		}
	}

	/* Legacy styles (keep for compatibility) */
	.modern-header {
		background: var(--yorha-bg-secondary);
		border-bottom: 1px solid var(--yorha-border-primary);
		box-shadow: var(--yorha-shadow-sm);
		position: sticky;
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
