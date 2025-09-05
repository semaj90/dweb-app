<script lang="ts">
	import { browser } from '$app/environment';
	import {
	  AlertCircle,
	  CheckCircle,
	  CloudUpload,
	  File as FileIcon,
	  FileText,
	  Image,
	  Upload,
	  X
	} from 'lucide-svelte';
	import { createEventDispatcher } from 'svelte';
	import { loki } from "../stores/lokiStore";
	import TagList from './TagList.svelte';

	export let reportId: string = '';
	export let acceptedTypes: string[] = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.txt', '.doc', '.docx'];
	export let maxFileSize = 10 * 1024 * 1024; // 10MB
	export let maxFiles = 5;
	export let multiple = true;

	const dispatch = createEventDispatcher<{
		upload: { files: globalThis.File[]; tags: string[] };
		filesChanged: FileUpload[];
		error: string;
	}>();

	interface FileUpload {
		id: string;
		file: globalThis.File;
		preview?: string;
		tags: string[];
		progress: number;
		status: 'pending' | 'uploading' | 'success' | 'error';
		error?: string;
		hash?: string;
	}

	let fileInput: HTMLInputElement;
	let dragActive = false;
	let uploads: FileUpload[] = [];
	let availableTags: string[] = [];

	// Load available tags
	$: if (browser) {
		loadAvailableTags();
	}

	async function loadAvailableTags() {
		try {
			const evidence = loki.evidence.getAll();
			const allTags = evidence.flatMap((e: unknown) => e.tags || []);
			availableTags = [...new Set(allTags as string[])].sort();
		} catch (error) {
			console.error('Failed to load available tags:', error);
		}
	}

	function getFileIcon(file: globalThis.File) {
		const type = file.type.toLowerCase();
		const name = file.name.toLowerCase();
		
		if (type.startsWith('image/')) {
			return Image;
		} else if (type === 'application/pdf' || name.endsWith('.pdf')) {
			return FileText; // Using FileText for PDF since lucide doesn't have a specific PDF icon
		} else if (type.startsWith('text/') || name.endsWith('.txt') || name.endsWith('.doc') || name.endsWith('.docx')) {
			return FileText;
		}
		
		return FileIcon;
	}

	function formatFileSize(bytes: number): string {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

	function isFileValid(file: globalThis.File): { valid: boolean; error?: string } {
		// Check file size
		if (file.size > maxFileSize) {
			return {
				valid: false,
				error: `File size exceeds ${formatFileSize(maxFileSize)} limit`
			};
		}

		// Check file type
		const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
		if (!acceptedTypes.some(type => type.toLowerCase() === fileExtension)) {
			return {
				valid: false,
				error: `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`
			};
		}

		return { valid: true };
	}

	async function createFilePreview(file: globalThis.File): Promise<string | undefined> {
		if (!file.type.startsWith('image/')) return undefined;

		return new Promise((resolve) => {
			const reader = new FileReader();
			reader.onload = (e) => resolve(e.target?.result as string);
			reader.onerror = () => resolve(undefined);
			reader.readAsDataURL(file);
		});
	}

	async function processFiles(files: FileList | globalThis.File[]) {
		const fileArray = Array.from(files);
		
		// Check total file limit
		if (uploads.length + fileArray.length > maxFiles) {
			dispatch('error', `Maximum ${maxFiles} files allowed`);
			return;
		}

		for (const file of fileArray) {
			const validation = isFileValid(file);
			
			if (!validation.valid) {
				dispatch('error', validation.error!);
				continue;
			}

			const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
			const preview = await createFilePreview(file);
			
			const upload: FileUpload = {
				id,
				file,
				preview,
				tags: [],
				progress: 0,
				status: 'pending'
			};

			uploads = [...uploads, upload];
		}
		
		dispatch('filesChanged', uploads);
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			processFiles(target.files);
		}
		// Reset input value to allow selecting the same file again
		target.value = '';
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		dragActive = true;
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		dragActive = false;
	}

	async function handleDrop(event: DragEvent) {
		event.preventDefault();
		dragActive = false;

		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			await processFiles(files);
		}
	}

	function removeFile(id: string) {
		uploads = uploads.filter(upload => upload.id !== id);
		dispatch('filesChanged', uploads);
	}

	function updateFileTags(id: string, tags: string[]) {
		uploads = uploads.map(upload => 
			upload.id === id ? { ...upload, tags } : upload
		);
		dispatch('filesChanged', uploads);
	}

	async function uploadFiles() {
		const pendingUploads = uploads.filter(u => u.status === 'pending');
		
		if (pendingUploads.length === 0) {
			dispatch('error', 'No files to upload');
			return;
		}

		// Start uploading files
		for (const upload of pendingUploads) {
			upload.status = 'uploading';
			uploads = [...uploads];

			try {
				// Simulate progress
				for (let progress = 0; progress <= 100; progress += 10) {
					upload.progress = progress;
					uploads = [...uploads];
					await new Promise(resolve => setTimeout(resolve, 100));
				}

				// Store in Loki.js for now (would normally upload to server)
				const evidenceData = {
					id: upload.id,
					caseId: reportId,
					criminalId: null,
					title: upload.file.name,
					description: `Uploaded file: ${upload.file.name}`,
					evidenceType: upload.file.type.includes('image') ? 'photo' : 
						upload.file.type.includes('video') ? 'video' : 
						upload.file.type.includes('audio') ? 'audio' : 'document',
					fileType: upload.file.type,
					subType: null,
					fileUrl: upload.preview || `file://${upload.file.name}`,
					fileName: upload.file.name,
					fileSize: upload.file.size,
					mimeType: upload.file.type,
					hash: upload.hash || null,
					tags: upload.tags,
					chainOfCustody: [],
					collectedAt: null,
					collectedBy: null,
					location: null,
					labAnalysis: {},
					aiAnalysis: {},
					aiTags: [],
					aiSummary: null,
					summary: null,
					isAdmissible: true,
					confidentialityLevel: 'standard',
					canvasPosition: {},
					uploadedBy: 'current-user', // TODO: get from auth context
					uploadedAt: new Date(),
					updatedAt: new Date()
				};

				loki.evidence.add(evidenceData);

				upload.status = 'success';
				upload.progress = 100;
			} catch (error) {
				upload.status = 'error';
				upload.error = error instanceof Error ? error.message : 'Upload failed';
			}
		}

		uploads = [...uploads];
		
		// Notify parent component
		const successfulFiles = uploads
			.filter(u => u.status === 'success')
			.map(u => u.file);
		
		if (successfulFiles.length > 0) {
			const allTags = uploads
				.filter(u => u.status === 'success')
				.flatMap(u => u.tags);
			
			dispatch('upload', { files: successfulFiles, tags: [...new Set(allTags)] });
		}
	}

	function clearCompleted() {
		uploads = uploads.filter(u => u.status !== 'success');
		dispatch('filesChanged', uploads);
	}

	function triggerFileSelect() {
		fileInput?.click();
	}
</script>

<div class="mx-auto px-4 max-w-7xl">
	<input
		bind:this={fileInput}
		type="file"
		{multiple}
		accept={acceptedTypes.join(',')}
		onchange={handleFileSelect}
		class="mx-auto px-4 max-w-7xl"
		aria-label="Select files for upload"
	/>

	<!-- Drop Zone -->
	<div
		class="mx-auto px-4 max-w-7xl"
		class:drag-active={dragActive}
		class:has-files={uploads.length > 0}
		on:dragover={handleDragOver}
		on:dragleave={handleDragLeave}
		on:drop={handleDrop}
		role="button"
		tabindex={0}
		onclick={() => triggerFileSelect()}
		onkeydown={(e) => e.key === 'Enter' && triggerFileSelect()}
	>
		{#if uploads.length === 0}
			<div class="mx-auto px-4 max-w-7xl">
				<Upload class="mx-auto px-4 max-w-7xl" size={48} />
				<h3 class="mx-auto px-4 max-w-7xl">Upload Evidence Files</h3>
				<p class="mx-auto px-4 max-w-7xl">
					Drag and drop files here, or click to browse
				</p>
				<p class="mx-auto px-4 max-w-7xl">
					Supports: {acceptedTypes.join(', ')} • Max {formatFileSize(maxFileSize)} per file • Up to {maxFiles} files
				</p>
			</div>
		{:else}
			<div class="mx-auto px-4 max-w-7xl">
				<CloudUpload size={24} />
				<span>{uploads.length} file{uploads.length !== 1 ? 's' : ''} ready</span>
				<button type="button" class="mx-auto px-4 max-w-7xl" onclick|stopPropagation={triggerFileSelect}>
					Add more
				</button>
			</div>
		{/if}
	</div>

	<!-- File List -->
	{#if uploads.length > 0}
		<div class="mx-auto px-4 max-w-7xl">
			{#each uploads as upload (upload.id)}
				<div class="mx-auto px-4 max-w-7xl" class:uploading={upload.status === 'uploading'}>
					<div class="mx-auto px-4 max-w-7xl">
						<div class="mx-auto px-4 max-w-7xl">
							{#if upload.preview}
								<img src={upload.preview} alt="File preview" class="mx-auto px-4 max-w-7xl" />
							{:else}
								<svelte:component this={getFileIcon(upload.file)} size={24} />
							{/if}
						</div>
						
						<div class="mx-auto px-4 max-w-7xl">
							<div class="mx-auto px-4 max-w-7xl">{upload.file.name}</div>
							<div class="mx-auto px-4 max-w-7xl">
								{formatFileSize(upload.file.size)}
								{#if upload.status === 'uploading'}
									• Uploading... {upload.progress}%
								{:else if upload.status === 'success'}
									• Uploaded
								{:else if upload.status === 'error'}
									• Error: {upload.error}
								{/if}
							</div>
						</div>

						<div class="mx-auto px-4 max-w-7xl">
							{#if upload.status === 'uploading'}
								<div class="mx-auto px-4 max-w-7xl">
									<div class="mx-auto px-4 max-w-7xl" style="--progress: {upload.progress}%"></div>
								</div>
							{:else if upload.status === 'success'}
								<CheckCircle size={20} class="mx-auto px-4 max-w-7xl" />
							{:else if upload.status === 'error'}
								<AlertCircle size={20} class="mx-auto px-4 max-w-7xl" />
							{/if}
						</div>

						{#if upload.status !== 'uploading'}
							<button
								type="button"
								class="mx-auto px-4 max-w-7xl"
								onclick={() => removeFile(upload.id)}
								aria-label="Remove {upload.file.name}"
							>
								<X size={16} />
							</button>
						{/if}
					</div>

					{#if upload.status === 'pending' || upload.status === 'error'}
						<div class="mx-auto px-4 max-w-7xl">
							<TagList
								bind:tags={upload.tags}
								{availableTags}
								placeholder="Add tags for this file..."
								maxTags={5}
								onchange={(e) => updateFileTags(upload.id, e.detail)}
							/>
						</div>
					{/if}

					{#if upload.status === 'uploading'}
						<div class="mx-auto px-4 max-w-7xl">
							<div class="mx-auto px-4 max-w-7xl" style="width: {upload.progress}%"></div>
						</div>
					{/if}
				</div>
			{/each}
		</div>

		<!-- Upload Actions -->
		<div class="mx-auto px-4 max-w-7xl">
			<button
				type="button"
				class="mx-auto px-4 max-w-7xl"
				onclick={() => uploadFiles()}
				disabled={uploads.every(u => u.status !== 'pending')}
			>
				Upload Files
				{#if uploads.filter(u => u.status === 'pending').length > 0}
					({uploads.filter(u => u.status === 'pending').length})
				{/if}
			</button>

			{#if uploads.some(u => u.status === 'success')}
				<button
					type="button"
					class="mx-auto px-4 max-w-7xl"
					onclick={() => clearCompleted()}
				>
					Clear Completed
				</button>
			{/if}
		</div>
	{/if}
</div>

<style>
	.file-upload-section {
		width: 100%;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.hidden {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	.drop-zone {
		border: 2px dashed #d1d5db;
		border-radius: 0.5rem;
		padding: 1.5rem;
		text-align: center;
		cursor: pointer;
		transition: all 0.2s;
		background-color: white;
	}

	.drop-zone:hover {
		border-color: #60a5fa;
		background-color: rgba(239, 246, 255, 0.5);
	}

	.drop-zone:focus {
		outline: none;
		box-shadow: 0 0 0 2px #3b82f6;
	}

	.drop-zone.drag-active {
		border-color: #3b82f6;
		background-color: #eff6ff;
	}

	.drop-zone.has-files {
		padding: 1rem;
	}

	.drop-zone-content {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.drop-title {
		font-size: 1.125rem;
		font-weight: 600;
		color: #111827;
	}

	.drop-description {
		color: #4b5563;
	}

	.drop-details {
		font-size: 0.875rem;
		color: #6b7280;
	}

	.upload-summary {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
		color: #374151;
	}

	.btn-link {
		color: #2563eb;
		text-decoration: underline;
		font-weight: 500;
		border-radius: 0.25rem;
		padding: 0.25rem;
	}

	.btn-link:hover {
		color: #1d4ed8;
	}

	.btn-link:focus {
		outline: none;
		box-shadow: 0 0 0 2px #3b82f6;
	}

	.file-list {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		max-height: 24rem;
		overflow-y: auto;
	}

	.file-item {
		background-color: white;
		border: 1px solid #e5e7eb;
		border-radius: 0.5rem;
		padding: 1rem;
		transition: all 0.2s;
	}

	.file-item.uploading {
		background-color: #eff6ff;
		border-color: #bfdbfe;
	}

	.file-info {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.file-icon-wrapper {
		flex-shrink: 0;
	}

	.file-preview {
		width: 3rem;
		height: 3rem;
		object-fit: cover;
		border-radius: 0.25rem;
		border: 1px solid #e5e7eb;
	}

	.file-details {
		flex: 1;
		min-width: 0;
	}

	.file-name {
		font-weight: 500;
		color: #111827;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.file-meta {
		font-size: 0.875rem;
		color: #6b7280;
	}

	.file-status {
		flex-shrink: 0;
	}

	.progress-ring {
		width: 1.25rem;
		height: 1.25rem;
		position: relative;
	}

	.progress-fill {
		position: absolute;
		inset: 0;
		border-radius: 50%;
		border: 2px solid #3b82f6;
		background: conic-gradient(#3b82f6 var(--progress), #e5e7eb var(--progress));
	}

	:global(.status-icon.success) {
		color: #059669;
	}

	:global(.status-icon.error) {
		color: #dc2626;
	}

	.remove-file {
		flex-shrink: 0;
		padding: 0.25rem;
		color: #9ca3af;
		border-radius: 0.25rem;
		background: none;
		border: none;
		cursor: pointer;
		transition: color 0.15s;
	}

	.remove-file:hover {
		color: #dc2626;
	}

	.remove-file:focus {
		outline: none;
		box-shadow: 0 0 0 2px #ef4444;
	}

	.file-tags {
		margin-top: 0.75rem;
		padding-top: 0.75rem;
		border-top: 1px solid #e5e7eb;
	}

	.progress-bar {
		margin-top: 0.5rem;
		width: 100%;
		background-color: #e5e7eb;
		border-radius: 9999px;
		height: 0.5rem;
		overflow: hidden;
	}

	.progress-bar .progress-fill {
		height: 100%;
		background-color: #3b82f6;
		transition: all 0.3s;
	}

	.upload-actions {
		display: flex;
		gap: 0.5rem;
		justify-content: flex-end;
	}

	.btn {
		padding: 1rem 1rem;
		border-radius: 0.375rem;
		font-weight: 500;
		border: none;
		cursor: pointer;
		transition: all 0.15s;
	}

	.btn:focus {
		outline: none;
		box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5);
	}

	.btn-primary {
		background-color: #2563eb;
		color: white;
	}

	.btn-primary:hover {
		background-color: #1d4ed8;
	}

	.btn-primary:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-secondary {
		background-color: #e5e7eb;
		color: #111827;
	}

	.btn-secondary:hover {
		background-color: #d1d5db;
	}
</style>
