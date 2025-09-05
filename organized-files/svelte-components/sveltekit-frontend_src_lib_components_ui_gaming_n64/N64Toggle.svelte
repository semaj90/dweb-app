<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { shouldEnableRetroEffects } from './retroPerformanceGuard';

  export let checked: boolean = false;
  export let disabled: boolean = false;
  export let name: string | undefined;

  const dispatch = createEventDispatcher<{ change: { checked: boolean } }>();

  let retroEnabled = false;
  $: thumbX = checked ? 18 : 0;

  onMount(() => {
	try {
	  retroEnabled = shouldEnableRetroEffects();
	} catch {
	  retroEnabled = false;
	}
  });

  function toggle() {
	if (disabled) return;
	checked = !checked;
	dispatch('change', { checked });
  }

  function handleKey(e: KeyboardEvent) {
	if (e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') {
	  e.preventDefault();
	  toggle();
	}
  }
</script>

<style>
  .n64-toggle {
	display: inline-flex;
	align-items: center
	cursor: pointer
	user-select: none
	outline: none
	border: none
	background: transparent
	padding: 4px;
  }
  .n64-toggle[aria-checked="true"] .thumb {
	transform: translateX(18px);
  }
  .switch {
	width: 36px;
	height: 20px;
	background: #ddd;
	border-radius: 10px;
	position: relative
	transition: background .2s;
  }
  .thumb {
	width: 16px;
	height: 16px;
	background: #fff;
	border-radius: 50%;
	position: absolute
	top: 2px;
	left: 2px;
	transition: transform .18s;
	box-shadow: 0 1px 2px rgba(0,0,0,.2);
  }
  .n64-toggle[aria-disabled="true"] { opacity: .5; cursor: default }
  :global(.n64-retro--enabled) .n64-toggle .switch {
	background: linear-gradient(180deg,#ffd27a,#ff9a3c);
  }
</style>

<button
  class="n64-toggle"
  role="switch"
  aria-checked={checked}
  aria-disabled={disabled}
  on:click={toggle}
  on:keydown={handleKey}
  disabled={disabled}
  {name}
>
  <span class="switch" aria-hidden="true">
	<span class="thumb" style="transform: translateX({thumbX}px)"></span>
  </span>
</button>
