<script lang="ts">
  interface Props {
    checked: boolean ;
    disabled: boolean ;
    id: string | undefined ;
    name: string | undefined ;
    value: string | undefined ;
    ariaLabel: string | undefined ;
  }
  let {
    checked = false,
    disabled = false,
    id = undefined,
    name = undefined,
    value = undefined,
    ariaLabel = undefined
  }: Props = $props();



  // Minimal, accessible checkbox component compatible with bind:checked
  export let checked: boolean = false;
  export let disabled: boolean = false;
  export let id: string | undefined = undefined;
  export let name: string | undefined = undefined;
  export let value: string | undefined = undefined;
  export let ariaLabel: string | undefined = undefined;
  let className: string = '';
  export { className as class };
</script>

<input
  type="checkbox"
  {id}
  {name}
  {value}
  bind:checked
  {disabled}
  aria-label={ariaLabel}
  class="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-600 disabled:opacity-50 disabled:cursor-not-allowed {className}"
  {...$$restProps}
/>
