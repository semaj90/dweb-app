<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import N64ToastStore from './N64ToastStore';
  import type { N64Toast } from './N64ToastStore';

  let toasts: N64Toast[] = [];
  let unsubscribe: () => void = () => {};

  onMount(() => {
	unsubscribe = N64ToastStore.subscribe((v) => (toasts = v));
  });

  onDestroy(() => {
	unsubscribe();
  });

  function remove(id: string) {
	N64ToastStore.remove(id);
  }
</script>

<style>
  .n64-toaster {
	position: fixed
	right: 16px;
	bottom: 16px;
	display: flex
	flex-direction: column
	gap: 8px;
	z-index: 1000;
  }
  .n64-toast {
	color: #fff;
	padding: 8px 12px;
	border-radius: 6px;
	min-width: 200px;
	box-shadow: 0 4px 12px rgba(0,0,0,0.2);
	display: flex
	justify-content: space-between;
	align-items: center
  }
  .n64-toast.info { background: #2b2f77; }
  .n64-toast.success { background: #2b7a2b; }
  .n64-toast.warning { background: #b06a00; }
  .n64-toast.error { background: #8b1e2f; }
  .close {
	background: transparent
	border: none
	color: inherit
	cursor: pointer
	margin-left: 12px;
	font-size: 14px;
	line-height: 1;
  }
</style>

<div class="n64-toaster" aria-live="polite" aria-atomic="true">
  {#each toasts as t (t.id)}
	<div class="n64-toast {t.type}">
	  <div class="message">{t.message}</div>
	  <button class="close" on:click={() => remove(t.id)} aria-label="Dismiss">✕</button>
	</div>
  {/each}
</div>
