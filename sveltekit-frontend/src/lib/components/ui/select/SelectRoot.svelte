<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  import { Select } from 'bits-ui';

  interface Props extends CommonProps {
    value?: string;
    onValueChange?: (value: string) => void;
    children?: any;
  }

  let { value = $bindable(), onValueChange, children, ...restProps }: Props = $props();
</script>

<Select.Root bind:value {onValueChange} {...restProps}>
  {@render children?.()}
</Select.Root>