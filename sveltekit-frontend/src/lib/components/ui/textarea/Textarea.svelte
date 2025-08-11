<script lang="ts">
  interface Props {
    value: string ;
    placeholder: string ;
    disabled: boolean ;
    readonly: boolean ;
    rows: number ;
    cols: number | undefined ;
    maxlength: number | undefined ;
    resize: "none" | "both" | "horizontal" | "vertical" ;
  }
  let {
    value = "",
    placeholder = "",
    disabled = false,
    readonly = false,
    rows = 3,
    cols = undefined,
    maxlength = undefined,
    resize = "vertical"
  }: Props = $props();



  import { cn } from "../../../utils";

                
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
