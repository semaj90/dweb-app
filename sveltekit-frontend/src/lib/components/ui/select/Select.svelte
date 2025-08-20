<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';

  // Basic Select wrapper that uses enhanced-bits Select
  import EnhancedSelect from '../enhanced-bits/Select.svelte';
  
  interface SelectOption {
    value: string;
    label: string;
    description?: string;
    disabled?: boolean;
    category?: string;
  }
  
  let { 
    value = $bindable(''),
    options = [] as SelectOption[],
    placeholder = 'Select...',
    ...props 
  }: {
    value?: string;
    options?: SelectOption[];
    placeholder?: string;
    [key: string]: any;
  } = $props();
</script>

<EnhancedSelect {value} {options} {placeholder} {...props} />
<script lang="ts">
import type { CommonProps } from '$lib/types/common-props';
interface Props extends CommonProps {}
</script>
