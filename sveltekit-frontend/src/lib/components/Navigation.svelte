<script lang="ts">
	import { page } from '$app/stores';
	import { cn } from '$lib/utils';
	import Badge from '$lib/components/ui/Badge.svelte';
	import Button from '$lib/components/ui/button/Button.svelte';
	
	const navItems = [
		{ href: '/', label: 'Dashboard', icon: '🏠' },
		{ href: '/evidence/analyze', label: 'Evidence Analysis', icon: '🔍' },
		{ href: '/cases', label: 'Cases', icon: '📁' },
		{ href: '/semantic-search-demo', label: 'Search Demo', icon: '🔎' },
		{ href: '/dev/self-prompting-demo', label: 'Agent Orchestration', icon: '🤖' },
		{ href: '/dev/mcp-tools', label: 'MCP Tools', icon: '🔧' }
	];
	
	$: currentPath = $page.url.pathname;
</script>

<nav class="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
	<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
		<div class="flex justify-between h-16">
			<div class="flex items-center">
				<div class="flex-shrink-0">
					<h1 class="text-xl font-bold">⚖️ Legal AI System</h1>
				</div>
				<div class="ml-10 flex items-center space-x-1">
					{#each navItems as item}
						<Button
							variant={currentPath === item.href ? 'default' : 'ghost'}
							size="sm"
							href={item.href}
							class={cn(
								"justify-start gap-2",
								currentPath === item.href && "bg-muted"
							)}
						>
							<span>{item.icon}</span>
							{item.label}
						</Button>
					{/each}
				</div>
			</div>
			
			<div class="flex items-center space-x-4">
				<Badge variant="outline" class="gap-2">
					<span>🤖</span>
					Multi-Agent Pipeline
				</Badge>
				<div class="flex items-center gap-2">
					<div class="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" title="System Online"></div>
					<span class="text-sm text-muted-foreground">Online</span>
				</div>
			</div>
		</div>
	</div>
</nav>