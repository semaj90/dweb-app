<script>
	import '../app.css';
	import Navigation from '$lib/components/Navigation.svelte';
</script>

<svelte:window on:keydown />

<main class="min-h-screen bg-background">
	<Navigation />
	<div class="container mx-auto p-4">
		<slot />
	</div>
</main>