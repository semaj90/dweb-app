<script lang="ts">
	import { ScrollArea as ScrollAreaPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";
	import type { ScrollAreaRootProps } from "bits-ui";

	type Props = ScrollAreaRootProps & {
		class?: string;
	};

	let { class: classNameVar, children, ...rest }: Props & { children?: any } = $props();
</script>

<ScrollAreaPrimitive.Root
	class={cn("relative overflow-hidden", classNameVar)}
	{...rest}
>
	<ScrollAreaPrimitive.Viewport class="h-full w-full rounded-[inherit]">
		{#if children}
			{@render children()}
		{/if}
	</ScrollAreaPrimitive.Viewport>
	<ScrollAreaPrimitive.Scrollbar orientation="vertical" />
	<ScrollAreaPrimitive.Scrollbar orientation="horizontal" />
	<ScrollAreaPrimitive.Corner />
</ScrollAreaPrimitive.Root>
