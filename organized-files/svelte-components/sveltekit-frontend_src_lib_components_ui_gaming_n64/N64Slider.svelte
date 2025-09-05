<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let id: string | undefined = undefined;
  export let value: number = 0;
  export let min: number = 0;
  export let max: number = 100;
  export let step: number = 1;
  export let disabled: boolean = false;
  export let vertical: boolean = false;
  export let ariaLabel: string | undefined = undefined;

  const dispatch = createEventDispatcher();

  function handleInput(e: Event) {
	value = Number((e.target as HTMLInputElement).value);
	dispatch('input', { value });
  }

  function handleChange(e: Event) {
	value = Number((e.target as HTMLInputElement).value);
	dispatch('change', { value });
  }

  function handleBlur() {
	dispatch('blur');
  }

  function handleFocus() {
	dispatch('focus');
  }
</script>

<style>
  .n64-slider {
	width: 100%;
	display: flex
	align-items: center
	padding: 4px 0;
  }

  .n64-slider.vertical {
	transform: rotate(-90deg);
  }

  input[type="range"].n64-range {
	-webkit-appearance: none
	appearance: none
	width: 100%;
	height: 10px;
	background: transparent
	cursor: pointer
  }

  /* Track */
  input[type="range"].n64-range::-webkit-slider-runnable-track {
	height: 6px;
	background: rgba(0, 0, 0, 0.18);
	border-radius: 6px;
	border: 1px solid rgba(255, 255, 255, 0.06);
  }
  input[type="range"].n64-range::-moz-range-track {
	height: 6px;
	background: rgba(0, 0, 0, 0.18);
	border-radius: 6px;
	border: 1px solid rgba(255, 255, 255, 0.06);
  }

  /* Fill for webkit: simulate filled part with gradient */
  input[type="range"].n64-range::-webkit-slider-thumb {
	-webkit-appearance: none
	appearance: none
	margin-top: -6px;
	width: 18px;
	height: 18px;
	border-radius: 50%;
	background: var(--n64-accent, #ffd400);
	border: 2px solid rgba(0,0,0,0.3);
	box-shadow: 0 2px 6px rgba(0,0,0,0.35);
  }

  input[type="range"].n64-range::-moz-range-thumb {
	width: 18px;
	height: 18px;
	border-radius: 50%;
	background: var(--n64-accent, #ffd400);
	border: 2px solid rgba(0,0,0,0.3);
	box-shadow: 0 2px 6px rgba(0,0,0,0.35);
  }

  input[type="range"].n64-range:focus {
	outline: none
  }

  input[type="range"].n64-range:disabled {
	opacity: 0.6;
	cursor: not-allowed;
  }
</style>

<div class="n64-slider {vertical ? 'vertical' : ''}">
  <input
	id={id}
	class="n64-range"
	type="range"
	bind:value
	{min}
	{max}
	{step}
	{disabled}
	{ariaLabel}
	aria-label={ariaLabel}
	on:input={handleInput}
	on:change={handleChange}
	on:blur={handleBlur}
	on:focus={handleFocus}
  />
</div>
