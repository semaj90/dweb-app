<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import type { Props } from "$lib/types/global";
  let {
    value = $bindable([0.5]),
    min = 0,
    max = 1,
    step = 0.01,
    disabled = false,
    id = undefined,
    class: klass = '',
  }: Props = $props();

  function onInput(e: Event) {
    const t = e.target as HTMLInputElement;
    const v = Number(t.value);
    // single-thumb emulation
    value = [Math.min(max, Math.max(min, v))];
  }
</script>

<input
  type="range"
  {id}
  {min}
  {max}
  {step}
  value={value?.[0] ?? 0}
  class={klass}
  {disabled}
  oninput={onInput as any} />

<style>
  input[type='range'] {
    width: 100%;
  }
  /* simple slider styling, can be replaced with Bits UI later */
</style>



<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';
interface Props extends CommonProps {}
</script>
