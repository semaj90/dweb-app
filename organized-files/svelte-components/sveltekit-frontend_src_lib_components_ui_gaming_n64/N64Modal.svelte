<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';

  export let id: string | undefined = undefined;
  export let open: boolean = false;
  export let ariaLabel: string = 'Modal';

  const dispatch = createEventDispatcher();

  function close() {
	open = false;
	dispatch('close');
  }

  function handleKeydown(e: KeyboardEvent) {
	if (e.key === 'Escape') close();
  }

  onMount(() => {
	document.addEventListener('keydown', handleKeydown);
  });

  onDestroy(() => {
	document.removeEventListener('keydown', handleKeydown);
  });
</script>

<style>
  .n64-modal-backdrop {
	position: fixed
	inset: 0;
	background: rgba(0, 0, 0, 0.5);
	display: flex
	align-items: center
	justify-content: center
	z-index: 999;
	box-sizing: border-box;
	padding: 1rem;
  }

  .n64-modal {
	background: linear-gradient(180deg, #fffdf0, #f7f3d9);
	border: 2px solid rgba(0, 0, 0, 0.12);
	border-radius: 8px;
	box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
	max-width: 100%;
	width: 480px;
	padding: 1rem;
	box-sizing: border-box;
  }

  .n64-modal__header {
	display: flex
	align-items: center
	justify-content: space-between;
	gap: 0.5rem;
	font-weight: 600;
	margin-bottom: 0.5rem;
  }

  .n64-modal__close {
	background: transparent
	border: none
	cursor: pointer
	font-size: 1rem;
	line-height: 1;
	padding: 0.25rem;
  }

  .n64-modal__body {
	margin-bottom: 0.5rem;
  }

  .n64-modal__footer {
	display: flex
	justify-content: flex-end;
	gap: 0.5rem;
  }
</style>

{#if open}
  <div
	class="n64-modal-backdrop"
	role="dialog"
	aria-label={ariaLabel}
	aria-modal="true"
	{id}
	on:click={(e) => {
	  if (e.target === e.currentTarget) close();
	}}
  >
	<div class="n64-modal" on:click|stopPropagation>
	  <div class="n64-modal__header">
		<slot name="header">
		  <span>{ariaLabel}</span>
		</slot>
		<button class="n64-modal__close" aria-label="Close modal" on:click={close}>✕</button>
	  </div>

	  <div class="n64-modal__body">
		<slot />
	  </div>

	  <div class="n64-modal__footer">
		<slot name="footer" />
	  </div>
	</div>
  </div>
{/if}
