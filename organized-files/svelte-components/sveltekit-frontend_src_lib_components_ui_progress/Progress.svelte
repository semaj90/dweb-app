<script lang="ts">
  interface Props {
    value?: any;
    max?: any;
  }
  let {
    value = 0,
    max = 100
  } = $props();



	import { cn } from '$lib/utils';

	let className = $state('');

	const percentage = $derived(Math.min((value / max) * 100, 100));
</script>

<div
	class={cn(
		"relative h-2 w-full overflow-hidden rounded-full bg-secondary",
		className
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