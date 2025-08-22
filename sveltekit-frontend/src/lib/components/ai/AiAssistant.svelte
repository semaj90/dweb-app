<!--
  AiAssistant.svelte
  - Production-ready, context7-compliant, Svelte 5, XState, and global store integration
  - Handles: streaming, memoization, global state, evidence source highlighting, and save-to-DB
  - Backend: expects /api/ai/process-evidence (LangChain, Ollama, pg_vector, Neo4j, Redis)
-->
<script lang="ts">
	import { getContext } from "svelte";
	import { state, effect, derived } from "svelte/reactivity";

	// UI components
	import { Button } from "$lib/components/ui/button";
	import { Card } from "$lib/components/ui/card";
	import { aiGlobalStore, aiGlobalActions } from "$lib/stores/ai";

	// Get user from context (SSR-safe)
	const getUser = getContext("user");
	const user = typeof getUser === "function" ? getUser() : undefined;

	let {
		contextItems = [],
		caseId = "",
	}: {
		contextItems?: unknown[];
		caseId?: string;
	} = $props();

	// Reactive state derived from the global store
	const currentSnapshot = state($aiGlobalStore);
	effect(() => {
		// This ensures the local state updates whenever the global store changes
		currentSnapshot.value = $aiGlobalStore;
	});

	const isLoading = derived(currentSnapshot, ($snapshot) => $snapshot.matches("summarizing"));
	const isSaving = derived(currentSnapshot, ($snapshot) => $snapshot.matches("saving"));
	const summary = derived(currentSnapshot, ($snapshot) => $snapshot.context.summary);
	const error = derived(currentSnapshot, ($snapshot) => $snapshot.context.error);
	const stream = derived(currentSnapshot, ($snapshot) => $snapshot.context.stream);
	const sources = derived(currentSnapshot, ($snapshot) => $snapshot.context.sources);

	// Trigger summary
	function handleSummarize() {
		if (!user?.id) return;
		// Using the specified model from the user request
		aiGlobalActions.summarize(caseId, contextItems, user.id, "gemma3:legal-latest");
	}

	// Save summary to DB
	function handleSave() {
		if (!$summary || !caseId) return;
		aiGlobalActions.saveSummary();
	}
</script>

<Card class="nier-card p-6">
	<div class="nier-header mb-4">
		<h3 class="nier-title text-lg font-bold mb-2">AI Evidence Summary</h3>
		<div class="flex gap-2">
			<Button
				on:click={handleSummarize}
				disabled={!user || $isLoading || $isSaving}
				variant="default"
				class="relative overflow-hidden transition-all duration-300 hover:translate-y--0.5 hover:shadow-lg"
			>
				{#if !user}
					Sign in to Summarize
				{:else if $isLoading}
					Summarizing...
				{:else}
					Summarize Evidence
				{/if}
			</Button>
			<Button
				on:click={handleSave}
				disabled={!$summary || $isLoading || $isSaving}
				variant="secondary"
				class="relative overflow-hidden transition-all duration-300 hover:translate-y--0.5 hover:shadow-lg"
			>
				{$isSaving ? "Saving..." : "Save Summary"}
			</Button>
		</div>
	</div>

	<div class="nier-content">
		{#if $isLoading}
			<div class="nier-loading">
				<span class="nier-text-muted">Summarizing evidence...</span>
				<!-- Streaming output (if supported) -->
				{#if $stream}
					<pre class="nier-code mt-2">{$stream}</pre>
				{/if}
			</div>
		{:else if $error}
			<div class="nier-error p-3 rounded">
				<span class="text-red-600">{$error}</span>
			</div>
		{:else if $summary}
			<div class="nier-summary">
				<pre class="nier-code whitespace-pre-wrap">{$summary}</pre>
				<!-- Top 3 evidence sources (if available) -->
				{#if $sources && $sources.length > 0}
					<div class="nier-sources mt-4 pt-4 border-t border-gray-200">
						<h4 class="nier-subtitle font-semibold mb-2">Top Evidence Used:</h4>
						<ol class="nier-list space-y-1">
							{#each $sources.slice(0, 3) as item, i}
								<li class="nier-list-item">
									<span class="nier-badge">{i + 1}</span>
									{item.title || item.id || `Evidence #${i + 1}`}
								</li>
							{/each}
						</ol>
					</div>
				{/if}
			</div>
		{:else}
			<div class="nier-empty">
				<span class="nier-text-muted">No summary yet.</span>
			</div>
		{/if}
	</div>
</Card>

<style>
	/* Nier.css inspired styles */
	:global(.nier-card) {
		background: rgba(255, 255, 255, 0.95);
		border: 2px solid #000;
		box-shadow: 4px 4px 0 rgba(0, 0, 0, 0.1);
	}

	:global(.nier-title) {
		letter-spacing: 0.05em;
		text-transform: uppercase;
	}

	:global(.nier-button) {
		position: relative;
		overflow: hidden;
		transition: all 0.3s ease;
	}

	:global(.nier-button:hover) {
		transform: translateY(-2px);
		box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
	}

	:global(.nier-code) {
		background: #f4f4f4;
		border: 1px solid #ddd;
		padding: 1rem;
		font-family: "Courier New", monospace;
		font-size: 0.875rem;
	}

	:global(.nier-error) {
		background: rgba(255, 0, 0, 0.05);
		border: 2px solid #ff0000;
	}

	:global(.nier-badge) {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 24px;
		background: #000;
		color: #fff;
		border-radius: 50%;
		font-size: 0.75rem;
		margin-right: 0.5rem;
	}

	:global(.nier-text-muted) {
		color: #666;
		font-style: italic;
	}

	:global(.nier-list-item) {
		display: flex;
		align-items: center;
		padding: 0.5rem 0;
	}
</style>
