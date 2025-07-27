<script>
	import '../app.css';
</script>

<main>
	<slot />
</main>

<style>
	:global(html) {
		background: #1a1a1a;
		color: #f5f5f5;
	}
	
	:global(body) {
		margin: 0;
		padding: 0;
		font-family: 'JetBrains Mono', monospace;
		line-height: 1.6;
		overflow-x: hidden;
	}
</style>
