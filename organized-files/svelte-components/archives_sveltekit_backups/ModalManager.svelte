<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { quintOut } from 'svelte/easing';
	import { fade, fly } from 'svelte/transition';
	import { modals } from "../../stores/modal";

	// Built-in modal components
	function ConfirmModal({ props, onConfirm, onClose }: unknown) {
		return {
			title: 'Confirm Action',
			content: props.message,
			actions: [
				{
					label: props.cancelText || 'Cancel',
					variant: 'ghost',
					action: onClose
				},
				{
					label: props.confirmText || 'Confirm',
					variant: 'primary',
					action: onConfirm
}
			]
		};
}
	function AlertModal({ props, onClose }: unknown) {
		return {
			title: 'Alert',
			content: props.message,
			actions: [
				{
					label: props.buttonText || 'OK',
					variant: 'primary',
					action: onClose
}
			]
		};
}
	function PromptModal({ props, onConfirm, onClose }: unknown) {
		let inputValue = props.defaultValue || '';

		return {
			title: 'Input Required',
			content: `
				<div class="mx-auto px-4 max-w-7xl">
					<p class="mx-auto px-4 max-w-7xl">${props.message}</p>
					<input
						type="text"
						class="mx-auto px-4 max-w-7xl"
						placeholder="${props.placeholder || ''}"
						value="${inputValue}"
						autofocus
					/>
				</div>
			`,
			actions: [
				{
					label: props.cancelText || 'Cancel',
					variant: 'ghost',
					action: onClose
				},
				{
					label: props.confirmText || 'OK',
					variant: 'primary',
					action: () => onConfirm?.(inputValue)
}
			]
		};
}
	const builtInComponents = {
		ConfirmModal,
		AlertModal,
		PromptModal
	};

	function getSizeClasses(size: string) {
		const sizeMap = {
			sm: 'max-w-md',
			md: 'max-w-lg',
			lg: 'max-w-2xl',
			xl: 'max-w-4xl',
			full: 'max-w-[95vw] max-h-[95vh]'
		};
		return sizeMap[size as keyof typeof sizeMap] || sizeMap.md;
}
	function handleBackdropClick(event: MouseEvent, modal: unknown) {
		if (event.target === event.currentTarget && !modal.persistent) {
			modals.close(modal.id);
}}
	function handleKeydown(event: KeyboardEvent, modal: unknown) {
		if (event.key === 'Escape' && modal.closable) {
			modals.close(modal.id);
}}
</script>

<!-- Render all active modals -->
{#each $modals.modals as modal (modal.id)}
	<div
		class="mx-auto px-4 max-w-7xl"
		onclick={(e) => handleBackdropClick(e, modal)}
		onkeydown={(e) => handleKeydown(e, modal)}
		role="dialog"
		aria-modal="true"
		aria-labelledby="{modal.id}-title"
		tabindex={-1}
		in:fade={{ duration: 200 "
		out:fade={{ duration: 150 "
	>
		<!-- Backdrop -->
		<div
			class="mx-auto px-4 max-w-7xl"
			aria-hidden="true "
		></div>

		<!-- Modal Content -->
		<div
			class={`
				relative bg-white dark:bg-gray-900 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-800
				w-full ${getSizeClasses(modal.size || 'md')} max-h-[90vh] overflow-hidden flex flex-col
			`}
			in:fly={{
				y: 30,
				duration: 300,
				easing: quintOut
			"
			out:fly={{
				y: -30,
				duration: 200,
				easing: quintOut
			 "
		>
			<!-- Header -->
			{#if modal.title || modal.closable !== false}
				<div class="mx-auto px-4 max-w-7xl">
					<div class="mx-auto px-4 max-w-7xl">
						{#if modal.title}
							<h2
								id="{modal.id}-title"
								class="mx-auto px-4 max-w-7xl"
							>
								{modal.title}
							</h2>
						{/if}
					</div>

					{#if modal.closable !== false}
						<button
							class="mx-auto px-4 max-w-7xl"
							onclick={() => modals.close(modal.id)}
							aria-label="Close modal"
						>
							<iconify-icon data-icon="${1}" class="mx-auto px-4 max-w-7xl"></iconify-icon>
						</button>
					{/if}
				</div>
			{/if}

			<!-- Content -->
			<div class="mx-auto px-4 max-w-7xl">
				{#if modal.component && typeof modal.component === 'string' && modal.component in builtInComponents}
					{@const builtInModal = (builtInComponents as any)[modal.component]({
						props: modal.props,
						onConfirm: modal.onConfirm,
						onClose: () => modals.close(modal.id)
					})}

					<!-- Built-in component content -->
					<div>
						{#if builtInModal.content}
							{@html builtInModal.content}
						{/if}
					</div>

					<!-- Built-in component actions -->
					{#if builtInModal.actions}
						<div class="mx-auto px-4 max-w-7xl">
							{#each builtInModal.actions as action}
								<Button
									variant={action.variant}
									onclick={() => action.action()}
								>
									{action.label}
								</Button>
							{/each}
						</div>
					{/if}
				{:else if modal.component}
					<!-- Custom Svelte component -->
					<svelte:component
						this={modal.component}
						{...modal.props}
						on:close={() => modals.close(modal.id)}
						on:confirm={modal.onConfirm}
					/>
				{:else}
					<!-- Default slot content -->
					<div class="mx-auto px-4 max-w-7xl">
						Modal content goes here
					</div>
				{/if}
			</div>
		</div>
	</div>
{/each}

<style>
	/* Smooth scrolling for modal content */
	.overflow-y-auto {
		scrollbar-width: thin;
		scrollbar-color: #9ca3af transparent;
}
	.overflow-y-auto::-webkit-scrollbar {
		width: 6px;
}
	.overflow-y-auto::-webkit-scrollbar-track {
		background: transparent;
}
	.overflow-y-auto::-webkit-scrollbar-thumb {
		background-color: #9ca3af;
		border-radius: 3px;
}
	.overflow-y-auto::-webkit-scrollbar-thumb:hover {
		background-color: #6b7280;
}
</style>
