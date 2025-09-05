<script lang="ts">
  import { Select } from "bits-ui";
  
  interface SelectOption {
    value: string
    label: string
    disabled?: boolean;
  }
  
  interface Props {
    value?: string;
    options: SelectOption[];
    placeholder?: string;
    disabled?: boolean;
    class?: string;
    onchange?: (value: string) => void;
  }
  
  let {
    value = $bindable(),
    options,
    placeholder = "Select option",
    disabled = false,
    class: className = "",
    onchange
  } = $props();
  
  // Helper to get selected option label
  const selectedLabel = $derived(options.find(option => option.value === value)?.label || placeholder
  );
  
  function handleValueChange(newValue: string | undefined) {
    if (newValue !== undefined) {
      value = newValue;
      onchange?.(newValue);
    }
  }
</script>

<Select.Root type="single" onValueChange={handleValueChange} {disabled}>
  <Select.Trigger 
    class="flex h-10 w-full items-center justify-between rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:cursor-not-allowed disabled:opacity-50 {className}"
  >
    <span class="truncate">
      {selectedLabel}
    </span>
  </Select.Trigger>
  
  <Select.Portal>
    <Select.Content 
      class="relative z-50 min-w-[8rem] overflow-hidden rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-950 dark:text-gray-50 shadow-md"
      sideOffset={4}
    >
      <Select.Viewport class="p-1">
        {#each options as option}
          <Select.Item 
            value={option.value} 
            disabled={option.disabled}
            class="relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none hover:bg-gray-100 dark:hover:bg-gray-700 focus:bg-gray-100 dark:focus:bg-gray-700 data-[disabled]:pointer-events-none data-[disabled]:opacity-50"
          >
            <div class="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
              {#if value === option.value}
                ✓
              {/if}
            </div>
            <span>
              {option.label}
            </span>
          </Select.Item>
        {/each}
      </Select.Viewport>
    </Select.Content>
  </Select.Portal>
</Select.Root>