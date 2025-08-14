<script lang="ts">
	import { ScrollArea as ScrollAreaPrimitive } from "bits-ui";
	import { cn } from "$lib/utils.js";
	import type { ScrollAreaRootProps } from "bits-ui";

	type Props = ScrollAreaRootProps & {
		class?: string;
	};

	let { children, class: className, ...rest }: Props = $props();
</script>

<ScrollAreaPrimitive.Root
	class={cn("relative overflow-hidden", className)}
	{...rest}
>
	<ScrollAreaPrimitive.Viewport class="h-full w-full rounded-[inherit]">
		{@render children?.()}
	</ScrollAreaPrimitive.Viewport>
	<ScrollAreaPrimitive.Scrollbar orientation="vertical" />
	<ScrollAreaPrimitive.Scrollbar orientation="horizontal" />
	<ScrollAreaPrimitive.Corner />
</ScrollAreaPrimitive.Root>
