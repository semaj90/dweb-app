<script lang="ts">
	import { onMount } from 'svelte';
	// Debounce + streaming support
	let debounceMs = 400;
	let autoSearch = true;
	let lastTimer: any = null;
	let useStreaming = true;
	let streaming = false;
	let streamedCount = 0;

	let query = '';
	let mode: 'simple' | 'enhanced' = 'simple';
	let limit = 8;
	let threshold: number | null = null;
	let model = '';
	let caseId = '';
	let autoFocus = true;

	let loading = false;
	let controller: AbortController | null = null;
	let results: any[] = [];
	let responseMeta: any = null;
	let errorMsg: string | null = null;

	function reset() {
		results = [];
		responseMeta = null;
		errorMsg = null;
		streamedCount = 0;
	}

	function scheduleDebounced() {
		if (!autoSearch) return;
		if (lastTimer) clearTimeout(lastTimer);
		lastTimer = setTimeout(() => {
			if (query.trim()) runSearch();
		}, debounceMs);
	}

	async function runSearch() {
		if (!query.trim()) return;
		reset();
		loading = true;
		controller?.abort();
		controller = new AbortController();
		const body: any = { query, limit, mode };
		if (threshold !== null && threshold >= 0) body.threshold = threshold;
		if (model.trim()) body.model = model.trim();
		if (caseId.trim()) body.caseId = caseId.trim();
		if (useStreaming) {
			await runStreaming(body);
			return;
		}
		try {
			const res = await fetch('/api/ai/vector-search', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
				signal: controller.signal
			});
			if (!res.ok) {
				errorMsg = `Request failed (${res.status})`;
			} else {
				const data = await res.json();
				results = data.results || [];
				responseMeta = data;
			}
		} catch (e: any) {
			if (e?.name !== 'AbortError') errorMsg = e?.message || String(e);
		} finally {
			loading = false;
		}
	}

	async function runStreaming(body: any) {
		streaming = true;
		try {
			const params = new URLSearchParams({ query: body.query, limit: String(body.limit || 8), mode: body.mode || 'simple' });
			if (body.threshold != null) params.set('threshold', String(body.threshold));
			if (body.model) params.set('model', body.model);
			if (body.caseId) params.set('caseId', body.caseId);
			const url = `/api/ai/vector-search/stream?${params.toString()}`;
			const res = await fetch(url, { signal: controller!.signal });
			if (!res.ok || !res.body) { errorMsg = `Stream failed (${res.status})`; return; }
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				let idx;
				while ((idx = buffer.indexOf('\n\n')) !== -1) {
					const raw = buffer.slice(0, idx).trim();
					buffer = buffer.slice(idx + 2);
					if (!raw) continue;
					const lines = raw.split('\n');
					let event = 'message';
					let dataStr = '';
					for (const line of lines) {
						if (line.startsWith('event:')) event = line.slice(6).trim();
						else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
					}
					if (dataStr) {
						try { handleStreamEvent(event, JSON.parse(dataStr)); } catch {}
					}
				}
			}
		} catch (e: any) {
			if (e?.name !== 'AbortError') errorMsg = e?.message || String(e);
		} finally {
			streaming = false;
			loading = false;
		}
	}

	function handleStreamEvent(event: string, data: any) {
		if (event === 'meta') {
			responseMeta = { ...(responseMeta || {}), ...data };
		} else if (event === 'result') {
			results = [...results, data];
			streamedCount = results.length;
		} else if (event === 'error') {
			errorMsg = data.message || 'Stream error';
		} else if (event === 'done') {
			responseMeta = { ...(responseMeta || {}), ...data, count: results.length };
		}
	}

	function submit(e: Event) {
		e.preventDefault();
		runSearch();
	}

	function abort() {
		controller?.abort();
		loading = false;
		streaming = false;
	}

	onMount(() => {
		if (autoFocus) {
			const el = document.getElementById('query-input');
			el?.focus();
		}
	});

	function scoreClass(score: number){
		if (score == null) return 'bg-neutral-200 dark:bg-neutral-700 text-neutral-700 dark:text-neutral-200';
		if (score >= 0.90) return 'score-top';
		if (score >= 0.80) return 'score-high';
		if (score >= 0.65) return 'score-mid';
		return 'score-low';
	}
</script>

<div class="mx-auto max-w-5xl p-6 space-y-6">
	<header class="space-y-2">
		<h1 class="text-2xl font-semibold tracking-tight">Enhanced Vector Search</h1>
		<p class="text-sm text-neutral-500 dark:text-neutral-400">Interact with the unified pgvector + (stub) enhanced RAG pipeline. Choose simple (direct similarity) or enhanced (RAG fallback) mode.</p>
	</header>

	<form class="grid gap-4 md:grid-cols-7 items-end bg-neutral-50 dark:bg-neutral-900/40 p-4 rounded-lg border border-neutral-200 dark:border-neutral-700" onsubmit={(e) => { e.preventDefault(); submit(); }}>
		<div class="md:col-span-3 flex flex-col gap-1">
			<label for="query-input" class="text-xs font-medium uppercase tracking-wide">Query</label>
			<input id="query-input" bind:value={query} oninput={scheduleDebounced} class="px-3 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Find clauses about indemnification..." />
		</div>
		<div class="flex flex-col gap-1">
			<label for="streaming-toggle" class="text-xs font-medium uppercase tracking-wide">Streaming</label>
			<select id="streaming-toggle" bind:value={useStreaming} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800">
				<option value={true}>on</option>
				<option value={false}>off</option>
			</select>
		</div>
			<div class="flex flex-col gap-1">
				<label for="mode-select" class="text-xs font-medium uppercase tracking-wide">Mode</label>
				<select id="mode-select" bind:value={mode} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800">
				<option value="simple">simple</option>
				<option value="enhanced">enhanced</option>
			</select>
		</div>
		<div class="flex flex-col gap-1">
				<label for="limit-input" class="text-xs font-medium uppercase tracking-wide">Limit</label>
				<input id="limit-input" type="number" min="1" max="50" bind:value={limit} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800" />
		</div>
		<div class="flex flex-col gap-1">
				<label for="threshold-input" class="text-xs font-medium uppercase tracking-wide">Threshold (0-1)</label>
				<input id="threshold-input" type="number" step="0.01" min="0" max="1" bind:value={threshold} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800" placeholder="optional" />
		</div>
		<div class="flex flex-col gap-1">
				<label for="model-input" class="text-xs font-medium uppercase tracking-wide">Model</label>
				<input id="model-input" bind:value={model} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800" placeholder="(auto)" />
		</div>
		<div class="flex flex-col gap-1">
				<label for="case-input" class="text-xs font-medium uppercase tracking-wide">Case ID</label>
				<input id="case-input" bind:value={caseId} class="px-2 py-2 rounded border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800" placeholder="optional" />
		</div>
		<div class="md:col-span-7 flex gap-3 pt-1">
			<button type="submit" class="px-4 py-2 rounded bg-indigo-600 text-white text-sm font-medium disabled:opacity-50" disabled={loading}>{loading ? (useStreaming ? (streaming ? 'Streaming…' : 'Starting…') : 'Searching…') : 'Search'}</button>
			{#if loading}
				<button type="button" onclick={abort} class="px-3 py-2 rounded bg-neutral-200 dark:bg-neutral-700 text-sm">Abort</button>
			{/if}
			<button type="button" onclick={reset} class="px-3 py-2 rounded border border-neutral-300 dark:border-neutral-600 text-sm">Clear</button>
		</div>
	</form>

	{#if errorMsg}
		<div class="p-3 rounded border border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900/30 text-sm text-red-700 dark:text-red-300">{errorMsg}</div>
	{/if}

	{#if responseMeta}
		<section class="space-y-4">
			<div class="flex flex-wrap gap-4 text-xs text-neutral-600 dark:text-neutral-400">
				<span><strong>Mode:</strong> {responseMeta.mode}</span>
				<span><strong>Source:</strong> {responseMeta.source}</span>
				<span><strong>Count:</strong> {responseMeta.count}</span>
				{#if responseMeta.timings}<span><strong>Total:</strong> {responseMeta.timings.totalMs}ms</span>{/if}
				{#if responseMeta.health}
					<span><strong>Go:</strong> {responseMeta.health.goService ? 'up' : '—'}</span>
					<span><strong>Summarizer:</strong> {responseMeta.health.summarizer ? 'up' : '—'}</span>
				{/if}
			</div>
			{#if responseMeta.errors && (responseMeta.errors.primary || responseMeta.errors.enhanced)}
				<details class="text-xs">
					<summary class="cursor-pointer">Errors</summary>
					<pre class="mt-2 p-2 bg-neutral-100 dark:bg-neutral-900 rounded overflow-auto text-[11px]">{JSON.stringify(responseMeta.errors, null, 2)}</pre>
				</details>
			{/if}
		</section>
	{/if}

	{#if results.length > 0}
		<section class="space-y-3">
			<h2 class="text-lg font-semibold">Results</h2>
			<ul class="space-y-3">
				{#each results as r (r.id)}
					<li class="p-4 rounded border border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-800 shadow-sm">
						<div class="flex justify-between items-start gap-4">
							<div class="text-sm font-mono truncate max-w-[60%]" title={r.id}>{r.id}</div>
							<div class="text-xs px-2 py-0.5 rounded font-semibold {scoreClass(r.score)}">{(r.score ?? 0).toFixed(3)}</div>
						</div>
						<p class="mt-2 text-sm leading-snug whitespace-pre-wrap break-words">{r.content}</p>
						{#if r.metadata}
							<details class="mt-2 text-xs">
								<summary class="cursor-pointer">Metadata</summary>
								<pre class="mt-1 p-2 bg-neutral-50 dark:bg-neutral-900 rounded overflow-auto max-h-48 text-[11px]">{JSON.stringify(r.metadata, null, 2)}</pre>
							</details>
						{/if}
					</li>
				{/each}
			</ul>
		</section>
	{/if}

	{#if !loading && !results.length && !errorMsg && query}
		<p class="text-sm italic text-neutral-500 dark:text-neutral-400">No results returned.</p>
	{/if}

	<footer class="pt-8 text-[11px] text-neutral-500 dark:text-neutral-500 space-y-2">
		<p>API: <code class="bg-neutral-100 dark:bg-neutral-800 px-1 rounded">POST /api/ai/vector-search</code></p>
		<p>Body fields: <code>{'{ query, limit?, threshold?, model?, mode?, caseId? }'}</code></p>
		<p>Streaming API: <code class="bg-neutral-100 dark:bg-neutral-800 px-1 rounded">GET /api/ai/vector-search/stream?query=...</code></p>
		{#if useStreaming}
			<p class="text-[10px] italic">Streaming {streamedCount} result(s){streaming ? '…' : ''}</p>
		{/if}
	</footer>
</div>

<style>
	:global(body){background:var(--background,transparent);}
	.score-low { background:#fee2e2; color:#991b1b; }
	.score-mid { background:#fef3c7; color:#92400e; }
	.score-high { background:#dcfce7; color:#065f46; }
	.score-top { background:#dbeafe; color:#1e40af; }
</style>