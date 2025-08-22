<script lang="ts">
  import { $props, $derived } from 'svelte';

import { cn } from '$lib/utils';

  interface Props {
    value?: unknown;
    max?: unknown;
    class?: string;
  }
  let {
    value = 0,
    max = 100,
    class: className = ''
  }: Props = $props();

	const percentage = $derived(Math.min((value / max) * 100, 100));
</script>

<div
	class={cn(
		"relative h-2 w-full overflow-hidden rounded-full bg-secondary",
		class
	)}
	role="progressbar"
	aria-valuenow={value}
	aria-valuemax={max}
	aria-valuemin="0"
>
	<div
		class="h-full w-full flex-1 bg-primary transition-all duration-300 ease-in-out"
		style="transform: translateX(-{100 - percentage}%)"
	></div>
</div>