<script>
	import '../app.css';
	import Navigation from '$lib/components/Navigation.svelte';
</script>

<div class="min-h-screen bg-background">
	<Navigation />
	<main class="container mx-auto py-6">
		<slot />
	</main>
</div>

<style>
	:global(html) {
		font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
	}
	
	:global(body) {
		margin: 0;
		padding: 0;
		line-height: 1.6;
		overflow-x: hidden;
	}
</style>
