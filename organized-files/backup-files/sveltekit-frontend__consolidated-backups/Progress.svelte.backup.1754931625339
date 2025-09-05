<script lang="ts">
	import { cn } from '$lib/utils';
	
	let className = '';
	export { className as class };
	export let value = 0;
	export let max = 100;
	
	$: percentage = Math.min((value / max) * 100, 100);
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