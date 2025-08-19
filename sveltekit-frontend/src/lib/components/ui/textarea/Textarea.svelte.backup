<script lang="ts">
  import { cn } from "../../../utils";

  export let value: string = "";
  export let placeholder: string = "";
  export let disabled: boolean = false;
  export let readonly: boolean = false;
  export let rows: number = 3;
  export let cols: number | undefined = undefined;
  export let maxlength: number | undefined = undefined;
  export let resize: "none" | "both" | "horizontal" | "vertical" = "vertical";

  let className: string = "";
  export { className as class };
</script>

<textarea
  bind:value
  {placeholder}
  {disabled}
  {readonly}
  {rows}
  {cols}
  {maxlength}
  class={cn(
    "flex min-h-[80px] w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm ring-offset-white placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-800 dark:bg-slate-950 dark:ring-offset-slate-950 dark:placeholder:text-slate-400 dark:focus-visible:ring-slate-300",
    resize === "none" && "resize-none",
    resize === "horizontal" && "resize-x",
    resize === "vertical" && "resize-y",
    resize === "both" && "resize",
    className
  )}
  {...$$restProps}
></textarea>
