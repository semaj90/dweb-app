<script lang="ts">
	import { onMount } from 'svelte';
	import { avatarStore } from "../stores/avatarStore";
	
	export let size: 'small' | 'medium' | 'large' = 'medium';
	export let clickable = false;
	export let showUploadButton = false;
	
	let fileInput: HTMLInputElement;
	let dragOver = false;
	
	$: avatarSize = {
		small: '32px',
		medium: '48px', 
		large: '80px'
	}[size];
	
	onMount(() => {
		avatarStore.loadAvatar();
	});
	
	function handleAvatarClick() {
		if (clickable && fileInput) {
			fileInput.click();
		}
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		const file = target.files?.[0];
		if (file) {
			uploadFile(file);
		}
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		dragOver = false;
		
		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			uploadFile(files[0]);
		}
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		dragOver = true;
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		dragOver = false;
	}

	async function uploadFile(file: File) {
		// Validate file type
		const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/svg+xml', 'image/webp'];
		if (!allowedTypes.includes(file.type)) {
			alert('Please select a valid image file (JPEG, PNG, GIF, SVG, WebP)');
			return;
		}

		// Validate file size (5MB)
		if (file.size > 5 * 1024 * 1024) {
			alert('File too large. Maximum size: 5MB');
			return;
		}

		const result = await avatarStore.uploadAvatar(file);
		if (!result.success) {
			alert(result.error || 'Upload failed');
		}
	}

	function handleRemoveAvatar() {
		if (confirm('Remove your avatar?')) {
			avatarStore.removeAvatar();
		}
	}
</script>

<div class="mx-auto px-4 max-w-7xl" class:clickable class:drag-over={dragOver}>
	<div 
		class="mx-auto px-4 max-w-7xl" 
		style="width: {avatarSize}; height: {avatarSize};"
		onclick={() => handleAvatarClick()}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ' ') {
				e.preventDefault();
				handleAvatarClick();
			}
		}}
		on:drop={handleDrop}
		on:dragover={handleDragOver}
		on:dragleave={handleDragLeave}
		role="button"
		tabindex={clickable ? 0 : -1}
		aria-label="Upload or change avatar"
	>
		{#if $avatarStore.isUploading}
			<div class="mx-auto px-4 max-w-7xl">
				<div class="mx-auto px-4 max-w-7xl"></div>
			</div>
		{:else}
			<img 
				src={$avatarStore.url || '/images/default-avatar.svg'} 
				alt="User Avatar"
				class="mx-auto px-4 max-w-7xl"
				loading="lazy"
			/>
		{/if}
		
		{#if clickable}
			<div class="mx-auto px-4 max-w-7xl">
				<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
					<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
					<polyline points="7,10 12,15 17,10"/>
					<line x1="12" y1="15" x2="12" y2="3"/>
				</svg>
			</div>
		{/if}
	</div>
	
	{#if showUploadButton}
		<div class="mx-auto px-4 max-w-7xl">
			<button 
				type="button" 
				class="mx-auto px-4 max-w-7xl"
				onclick={() => fileInput?.click()}
				disabled={$avatarStore.isUploading}
			>
				{$avatarStore.isUploading ? 'Uploading...' : 'Change Avatar'}
			</button>
			
			{#if $avatarStore.url && $avatarStore.url !== '/images/default-avatar.svg'}
				<button 
					type="button" 
					class="mx-auto px-4 max-w-7xl"
					onclick={() => handleRemoveAvatar()}
				>
					Remove
				</button>
			{/if}
		</div>
	{/if}
	
	{#if $avatarStore.error}
		<div class="mx-auto px-4 max-w-7xl">
			{$avatarStore.error}
			<button type="button" onclick={() => avatarStore.clearError()} class="mx-auto px-4 max-w-7xl">×</button>
		</div>
	{/if}
</div>

<input 
	bind:this={fileInput}
	type="file" 
	accept="image/jpeg,image/png,image/gif,image/svg+xml,image/webp"
	onchange={handleFileSelect}
	style="display: none;"
/>

<style>
	.avatar-container {
		position: relative;
		display: inline-block;
	}

	.avatar {
		position: relative;
		border-radius: 50%;
		overflow: hidden;
		border: 2px solid #e5e7eb;
		transition: all 0.2s ease;
		background: #f9fafb;
	}

	.clickable .avatar:hover {
		border-color: #3b82f6;
		cursor: pointer;
		transform: scale(1.05);
	}

	.drag-over .avatar {
		border-color: #10b981;
		background: #ecfdf5;
	}

	.avatar-image {
		width: 100%;
		height: 100%;
		object-fit: cover;
		transition: opacity 0.2s ease;
	}

	.loading-spinner {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 100%;
		height: 100%;
		background: #f9fafb;
	}

	.spinner {
		width: 24px;
		height: 24px;
		border: 2px solid #e5e7eb;
		border-top: 2px solid #3b82f6;
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}

	.upload-overlay {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
		opacity: 0;
		transition: opacity 0.2s ease;
		color: white;
	}

	.clickable:hover .upload-overlay {
		opacity: 1;
	}

	.upload-controls {
		margin-top: 12px;
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}

	.upload-btn, .remove-btn {
		padding: 8px 16px;
		border: none;
		border-radius: 6px;
		font-size: 14px;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.upload-btn {
		background: #3b82f6;
		color: white;
	}

	.upload-btn:hover:not(:disabled) {
		background: #2563eb;
	}

	.upload-btn:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.remove-btn {
		background: #ef4444;
		color: white;
	}

	.remove-btn:hover {
		background: #dc2626;
	}

	.error-message {
		position: absolute;
		top: 100%;
		left: 0;
		right: 0;
		margin-top: 8px;
		padding: 8px 12px;
		background: #fef2f2;
		border: 1px solid #fecaca;
		border-radius: 6px;
		color: #dc2626;
		font-size: 14px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		z-index: 10;
	}

	.close-error {
		background: none;
		border: none;
		color: #dc2626;
		cursor: pointer;
		font-size: 18px;
		line-height: 1;
		padding: 0;
		margin-left: 8px;
	}
</style>