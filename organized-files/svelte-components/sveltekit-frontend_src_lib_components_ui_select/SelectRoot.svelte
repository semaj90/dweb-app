<script lang="ts">
  import { Select } from 'bits-ui';

  interface Props {
    value?: string;
    onValueChange?: (value: string) => void;
    children?: any;
  }

  let { value = $bindable(), onValueChange, children, ...restProps } = $props();
</script>

<Select.Root bind:value {onValueChange} {...restProps}>
  {@render children?.()}
</Select.Root>