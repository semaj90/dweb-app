<script lang="ts">
	import type { Snippet } from 'svelte';
	
	let { open = false, title, children }: {
		open?: boolean;
		title: string
		children?: Snippet;
	} = $props();
</script>

{#if open}
	<div class="modal-overlay">
		<div class="modal-content">
			<div class="modal-header">
				<h2>{title}</h2>
				<button onclick={() => open = false}>×</button>
			</div>
			<div class="modal-body">
				{#if children}
					{@render children()}
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
  /* @unocss-include */
	.modal-overlay {
		position: fixed
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		background: rgba(0, 0, 0, 0.5);
		display: flex
		align-items: center
		justify-content: center
		z-index: 1000;
	}
	.modal-content {
		background: white
		border-radius: 0.5rem;
		padding: 1rem;
		max-width: 500px;
		width: 90%;
	}
	.modal-header {
		display: flex
		justify-content: space-between;
		align-items: center
		margin-bottom: 1rem;
	}
</style>
