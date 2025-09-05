<script lang="ts">
	/** Optimized MinIO Upload Component
	 *  - Native Windows friendly (no fs dependencies)
	 *  - Uses Svelte 5 $state stores for local reactive state
	 *  - Integrates with Bits-UI (optional graceful fallback)
	 *  - Supports:
	 *      * Single & multi-file selection
	 *      * Parallel chunked uploads (configurable)
	 *      * Auto–retry with exponential backoff
	 *      * Presigned URL flow (preferred) with fallback to `/api/upload`
	 *  - Emits custom events: success, error, progress, complete
	 */
	import { onMount, createEventDispatcher } from 'svelte';

	// Optional Bits UI components (guarded dynamic import to avoid build break if missing)
	let Button: any = 'button';
	let Progress: any = null;
	onMount(async () => {
		try {
			const mod = await import('@bits-ui/svelte');
			Button = (mod as any).Button ?? 'button';
			Progress = (mod as any).Progress ?? null;
		} catch (e) {
			// Silent fallback
		}
	});

	const dispatch = createEventDispatcher();

	// Component props
	export let multiple: boolean = true;
	export let parallel: number = 3;                // Max concurrent uploads
	export let chunkSize: number = 5 * 1024 * 1024; // 5MB default
	export let usePresigned: boolean = true;        // Prefer presigned MinIO URLs
	export let accept: string = '*/*';
	export let autoStart: boolean = true;
	export let maxFiles: number = 25;

	// Reactive state ($state for Svelte 5)
	let files = $state<Array<InternalFile>>([]);
	let overallProgress = $state(0);
	let uploading = $state(false);
	let errors = $state<string[]>([]);
	let completed = $state(false);
	let cancelController: AbortController | null = null;

	interface InternalFile {
		id: string
		file: File
		status: 'pending' | 'uploading' | 'error' | 'done' | 'cancelled';
		progress: number // 0..1
		uploadedBytes: number
		totalBytes: number
		attempts: number
		presigned?: string; // presigned upload URL
		etags?: string[];   // for multi-part (future)
		error?: string;
	}

	function selectFiles(e: Event) {
		const input = e.target as HTMLInputElement;
		if (!input.files) return;
		const list = Array.from(input.files);
		const newItems: InternalFile[] = list.slice(0, Math.max(0, maxFiles - files.length)).map(f => ({
			id: crypto.randomUUID(),
			file: f,
			status: 'pending',
			progress: 0,
			uploadedBytes: 0,
			totalBytes: f.size,
			attempts: 0
		}));
		files = [...files, ...newItems];
		if (autoStart) startUpload();
	}

	async function fetchPresigned(f: InternalFile): Promise<string | null> {
		try {
			const res = await fetch(`/api/upload/presign?filename=${encodeURIComponent(f.file.name)}&size=${f.file.size}`);
			if (!res.ok) return null;
			const data = await res.json();
			return data?.url || null;
		} catch {
			return null;
		}
	}

	async function startUpload() {
		if (uploading) return;
		uploading = true;
		completed = false;
		errors = [];
		cancelController = new AbortController();
		const queue = files.filter(f => f.status === 'pending' || f.status === 'error');
		let active = 0;
		let index = 0;
		const runNext = () => {
			if (index >= queue.length) {
				if (active === 0) finalize();
				return;
			}
			while (active < parallel && index < queue.length) {
				const f = queue[index++];
				active++;
				uploadFile(f).finally(() => {
					active--;
					runNext();
				});
			}
		};
		runNext();
	}

	async function uploadFile(f: InternalFile) {
		f.status = 'uploading';
		if (usePresigned && !f.presigned) {
			f.presigned = await fetchPresigned(f) || undefined;
		}
		try {
			if (f.file.size <= chunkSize) {
				await uploadChunk(f, 0, f.file.size, 0);
			} else {
				let offset = 0; let part = 0;
				while (offset < f.file.size) {
					const end = Math.min(offset + chunkSize, f.file.size);
						await uploadChunk(f, offset, end, part++);
					offset = end;
					if (f.status === 'cancelled') return;
				}
			}
			f.status = f.status !== 'cancelled' ? 'done' : 'cancelled';
			f.progress = 1;
			updateOverallProgress();
			if (f.status === 'done') dispatch('success', { id: f.id, name: f.file.name });
		} catch (e: any) {
			f.status = 'error';
			f.error = e?.message || 'Upload failed';
			errors = [...errors, `${f.file.name}: ${f.error}`];
			dispatch('error', { id: f.id, error: f.error });
		}
	}

	async function uploadChunk(f: InternalFile, start: number, end: number, part: number) {
		const blob = f.file.slice(start, end);
		const attemptUpload = async (attempt: number): Promise<Response> => {
			const signal = cancelController!.signal;
			if (f.presigned) {
				return fetch(f.presigned, { method: 'PUT', body: blob, signal });
			}
			const form = new FormData();
			form.append('file', blob, f.file.name);
			form.append('part', String(part));
			form.append('start', String(start));
			form.append('end', String(end));
			return fetch('/api/upload', { method: 'POST', body: form, signal });
		};
		let attempt = 0;
		while (attempt < 5) {
			try {
				const res = await attemptUpload(attempt);
				if (!res.ok) throw new Error(`HTTP ${res.status}`);
				f.uploadedBytes += blob.size;
				f.progress = f.uploadedBytes / f.totalBytes;
				updateOverallProgress();
				dispatch('progress', { id: f.id, progress: f.progress });
				return;
			} catch (err) {
				attempt++;
				await new Promise(r => setTimeout(r, 200 * 2 ** attempt));
				if (attempt >= 5) throw err;
			}
		}
	}

	function updateOverallProgress() {
		const total = files.reduce((acc, f) => acc + f.totalBytes, 0);
		const uploaded = files.reduce((acc, f) => acc + f.uploadedBytes, 0);
		overallProgress = total ? uploaded / total : 0;
	}

	function cancelAll() {
		cancelController?.abort();
		files = files.map(f => f.status === 'uploading' ? { ...f, status: 'cancelled' } : f);
		uploading = false;
		dispatch('cancel');
	}

	function finalize() {
		uploading = false;
		completed = true;
		dispatch('complete', { errors: errors.length, total: files.length });
	}

	function removeFile(id: string) {
		files = files.filter(f => f.id !== id);
		updateOverallProgress();
	}
</script>

<div class="upload-container" data-state={uploading ? 'uploading' : 'idle'}>
	<div class="header">
		<h3>Optimized MinIO Upload</h3>
		{#if uploading}
			<span class="status uploading">Uploading…</span>
		{:else if completed}
			<span class="status done">Done</span>
		{/if}
	</div>

	<div class="controls">
		<label class="file-select">
			<input type="file" {accept} {multiple} on:change={selectFiles} />
			<Button class="select-btn" type="button">Select Files</Button>
		</label>
		<Button type="button" disabled={uploading || files.length===0} on:click={startUpload}>Start</Button>
		<Button type="button" disabled={!uploading} on:click={cancelAll}>Cancel</Button>
	</div>

	{#if files.length > 0}
		<div class="file-list">
			{#each files as f}
				<div class="file-row" data-status={f.status}>
					<div class="meta">
						<strong>{f.file.name}</strong>
						<small>{(f.file.size/1024).toFixed(1)} KB</small>
					</div>
					<div class="progress-wrap">
						{#if Progress}
							<Progress value={Math.round(f.progress*100)} max={100} />
						{:else}
							<div class="bar"><span style={`width:${f.progress*100}%`}></span></div>
						{/if}
						<small>{Math.round(f.progress*100)}%</small>
					</div>
					<div class="actions">
						{#if f.status === 'error'}<span class="err" title={f.error}>⚠</span>{/if}
						<button class="remove" on:click={() => removeFile(f.id)} disabled={f.status==='uploading'}>✕</button>
					</div>
				</div>
			{/each}
		</div>
	{/if}

	<div class="overall">
		<div class="overall-bar">
			<div class="bar"><span style={`width:${overallProgress*100}%`}></span></div>
			<small>{Math.round(overallProgress*100)}%</small>
		</div>
		{#if errors.length > 0}
			<details class="errors">
				<summary>{errors.length} error(s)</summary>
				<ul>{#each errors as e}<li>{e}</li>{/each}</ul>
			</details>
		{/if}
	</div>
</div>

<style>
	.upload-container { border: 1px solid var(--border,#333); padding: 1rem; border-radius: 8px; background: var(--panel,#111); color: var(--fg,#eee); font-family: system-ui, sans-serif; }
	.header { display: flex align-items: center gap:.75rem; justify-content:space-between; margin-bottom:.5rem; }
	.status { font-size:.75rem; padding:.25rem .5rem; border-radius:4px; background:#222; }
	.status.uploading { background:#1e3a8a; }
	.status.done { background:#065f46; }
	.controls { display: flex gap:.5rem; flex-wrap: wrap margin-bottom:.75rem; }
	.file-select input { display: none }
	.select-btn { cursor: pointer }
	.file-list { display: flex flex-direction: column gap:.5rem; max-height:260px; overflow: auto }
	.file-row { display: grid grid-template-columns: minmax(0,1fr) 220px 40px; gap:.75rem; align-items: center font-size:.85rem; padding:.4rem .5rem; border:1px solid #222; border-radius:6px; background:#181818; }
	.file-row[data-status='error'] { border-color:#b91c1c; }
	.file-row[data-status='done'] { border-color:#065f46; }
	.meta { display: flex flex-direction: column gap:2px; overflow: hidden }
	.meta strong { font-weight:600; white-space: nowrap overflow: hidden text-overflow: ellipsis }
	.progress-wrap { display: flex align-items: center gap:.5rem; }
	.bar { position: relative flex:1; height:8px; background:#222; border-radius:4px; overflow: hidden }
	.bar span { position: absolute left:0; top:0; bottom:0; background:linear-gradient(90deg,#2563eb,#10b981); box-shadow:0 0 0 1px #0006 inset; }
	.actions { display: flex align-items: center gap:.35rem; }
	.remove { background: none border: none color:#999; cursor: pointer font-size:.85rem; }
	.remove:hover { color:#fff; }
	.err { color:#dc2626; cursor: help }
	.overall { margin-top:.75rem; display: flex flex-direction: column gap:.5rem; }
	.overall-bar { display: flex align-items: center gap:.5rem; }
	details.errors { font-size:.75rem; }
	details.errors ul { margin:.25rem 0 0 .75rem; padding:0; list-style: disc display: flex flex-direction: column gap:.25rem; }
	button, .select-btn { background:#1f2937; color:#eee; border:1px solid #374151; padding:.45rem .75rem; border-radius:6px; font-size:.8rem; line-height:1; font-weight:500; }
	button:hover:enabled { background:#334155; }
	button:disabled { opacity:.45; cursor:not-allowed; }
</style>
