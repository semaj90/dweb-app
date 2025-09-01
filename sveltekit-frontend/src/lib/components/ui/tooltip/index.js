// Export a simple API compatible with named imports used in pages
export { default as Tooltip } from "../Tooltip.svelte";
export { default } from "../Tooltip.svelte";

// Provide fallbacks for Content/Trigger/Provider to avoid TS import errors
export const TooltipContent = undefined;
export const TooltipTrigger = undefined;
export const TooltipProvider = undefined;
;