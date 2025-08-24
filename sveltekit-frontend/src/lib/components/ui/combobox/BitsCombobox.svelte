<!-- Combobox Component for Legal AI App -->
<script lang="ts">
  import { Combobox } from 'bits-ui';
  import { Check, ChevronDown, Search, X } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { $props, $derived, $state } from 'svelte';

  export interface ComboboxOption {
    value: string;
    label: string;
    description?: string;
    category?: string;
    disabled?: boolean;
    metadata?: Record<string, any>;
  }

  export interface ComboboxProps {
    options: ComboboxOption[];
    value?: string;
    placeholder?: string;
    searchPlaceholder?: string;
    emptyMessage?: string;
    disabled?: boolean;
    required?: boolean;
    multiple?: boolean;
    creatable?: boolean;
    categories?: boolean;
    label?: string;
    description?: string;
    error?: string;
    class?: string;
    onValueChange?: (value: string | string[] | undefined) => void;
    onCreateOption?: (inputValue: string) => ComboboxOption | Promise<ComboboxOption>;
  }

  let {
    options = [],
    value = $bindable(multiple ? [] : undefined),
    placeholder = 'Select option...',
    searchPlaceholder = 'Search options...',
    emptyMessage = 'No options found',
    disabled = false,
    required = false,
    multiple = false,
    creatable = false,
    categories = false,
    label,
    description,
    error,
    class: className = '',
    onValueChange,
    onCreateOption
  }: ComboboxProps = $props();

  let inputValue = $state('');
  let open = $state(false);

  // Filter options based on search input
  const filteredOptions = $derived(() => {
    if (!inputValue) return options;
    
    const query = inputValue.toLowerCase();
    return options.filter(option =>
      option.label.toLowerCase().includes(query) ||
      option.description?.toLowerCase().includes(query) ||
      option.category?.toLowerCase().includes(query)
    );
  });

  // Group options by category if categories are enabled
  const groupedOptions = $derived(() => {
    if (!categories) return [{ category: null, options: filteredOptions }];

    const grouped = filteredOptions.reduce((acc, option) => {
      const category = option.category || 'Other';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(option);
      return acc;
    }, {} as Record<string, ComboboxOption[]>);

    return Object.entries(grouped).map(([category, options]) => ({
      category: category === 'Other' ? null : category,
      options
    }));
  });

  // Find selected option(s) for display
  const selectedOptions = $derived(() => {
    if (multiple && Array.isArray(value)) {
      return options.filter(option => value.includes(option.value));
    } else if (!multiple && typeof value === 'string') {
      return options.find(option => option.value === value);
    }
    return null;
  });

  // Check if option can be created
  const canCreateOption = $derived(() => {
    return creatable && 
           inputValue.trim() && 
           !filteredOptions.some(opt => opt.label.toLowerCase() === inputValue.toLowerCase());
  });

  function handleValueChange(newValue: string | undefined) {
    if (multiple) {
      const currentValues = Array.isArray(value) ? value : [];
      if (newValue && !currentValues.includes(newValue)) {
        value = [...currentValues, newValue];
      }
    } else {
      value = newValue;
    }
    onValueChange?.(value);
  }

  function removeValue(valueToRemove: string) {
    if (multiple && Array.isArray(value)) {
      value = value.filter(v => v !== valueToRemove);
      onValueChange?.(value);
    }
  }

  async function handleCreateOption() {
    if (!canCreateOption || !onCreateOption) return;
    
    try {
      const newOption = await onCreateOption(inputValue);
      options = [...options, newOption];
      handleValueChange(newOption.value);
      inputValue = '';
    } catch (error) {
      console.error('Failed to create option:', error);
    }
  }

  // Generate unique ID for accessibility
  const inputId = `combobox-${Math.random().toString(36).substr(2, 9)}`;
</script>

<div class="legal-combobox-container w-full space-y-2">
  <!-- Label -->
  {#if label}
    <label 
      for={inputId}
      class="block text-sm font-medium text-yorha-text-primary font-mono"
    >
      {label}
      {#if required}
        <span class="text-yorha-accent ml-1">*</span>
      {/if}
    </label>
  {/if}

  <Combobox.Root 
    bind:inputValue 
    bind:open
    {disabled}
    {multiple}
    onSelectedChange={handleValueChange}
  >
    <div class="relative">
      <Combobox.Input
        id={inputId}
        placeholder={open ? searchPlaceholder : placeholder}
        class={cn(
          'flex h-9 w-full items-center justify-between rounded-md border bg-transparent px-3 py-2 text-sm font-mono ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
          'border-yorha-border bg-yorha-bg-tertiary text-yorha-text-primary',
          'focus:ring-yorha-primary',
          error && 'border-red-500 focus:ring-red-500',
          className
        )}
        {required}
      />

      <Combobox.Trigger
        class="absolute inset-y-0 right-0 flex h-full w-9 items-center justify-center"
      >
        <ChevronDown class="h-4 w-4 shrink-0 opacity-50" />
      </Combobox.Trigger>
    </div>

    <Combobox.Content
      class="relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md border-yorha-border bg-yorha-bg-secondary"
      sideOffset={4}
    >
      <div class="p-1">
        <!-- Create option -->
        {#if canCreateOption}
          <Combobox.Item
            value={inputValue}
            onSelect={handleCreateOption}
            class="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-yorha-bg-tertiary data-[highlighted]:bg-yorha-primary data-[highlighted]:text-yorha-bg-primary font-mono"
          >
            <div class="flex items-center gap-2">
              <Search class="w-4 h-4" />
              <span>Create "{inputValue}"</span>
            </div>
          </Combobox.Item>
          <div class="border-b border-yorha-border my-1"></div>
        {/if}

        <!-- Options -->
        {#if groupedOptions.length === 0}
          <div class="py-6 text-center text-sm text-yorha-text-secondary font-mono">
            {emptyMessage}
          </div>
        {:else}
          {#each groupedOptions as group}
            {#if group.category}
              <div class="px-2 py-1.5 text-xs font-medium text-yorha-text-secondary uppercase font-mono">
                {group.category}
              </div>
            {/if}
            
            {#each group.options as option}
              <Combobox.Item
                value={option.value}
                disabled={option.disabled}
                class="relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-yorha-bg-tertiary data-[highlighted]:bg-yorha-primary data-[highlighted]:text-yorha-bg-primary data-[disabled]:pointer-events-none data-[disabled]:opacity-50 font-mono"
              >
                <div class="flex items-center justify-between w-full">
                  <div class="flex flex-col">
                    <span class="font-medium">{option.label}</span>
                    {#if option.description}
                      <span class="text-xs text-yorha-text-secondary">
                        {option.description}
                      </span>
                    {/if}
                  </div>
                  
                  {#if (multiple && Array.isArray(value) && value.includes(option.value)) || (!multiple && value === option.value)}
                    <Check class="h-4 w-4" />
                  {/if}
                </div>
              </Combobox.Item>
            {/each}
          {/each}
        {/if}
      </div>
    </Combobox.Content>

    <!-- Hidden input for form submission -->
    <Combobox.HiddenInput />
  </Combobox.Root>

  <!-- Selected items display for multiple selection -->
  {#if multiple && Array.isArray(value) && value.length > 0}
    <div class="flex flex-wrap gap-2 mt-2">
      {#each selectedOptions as option}
        <div class="inline-flex items-center gap-1 bg-yorha-primary/10 text-yorha-primary text-xs font-mono px-2 py-1 rounded border border-yorha-primary/20">
          {option.label}
          <button
            type="button"
            onclick={() => removeValue(option.value)}
            class="hover:bg-yorha-primary/20 rounded p-0.5"
          >
            <X class="w-3 h-3" />
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <!-- Description -->
  {#if description}
    <p class="text-xs text-yorha-text-secondary font-mono">
      {description}
    </p>
  {/if}

  <!-- Error Message -->
  {#if error}
    <p class="text-xs text-red-500 font-mono">
      {error}
    </p>
  {/if}
</div>

<style>
  :global(.legal-combobox-container input) {
    transition: all 0.2s ease;
  }

  :global(.legal-combobox-container input:focus) {
    box-shadow: 0 0 0 1px rgb(var(--yorha-primary) / 0.5);
  }
</style>