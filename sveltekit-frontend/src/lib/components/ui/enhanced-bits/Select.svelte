<script lang="ts">
  import { Select as BitsSelect } from 'bits-ui';
  import { cn } from '$lib/utils/cn';
  import { ChevronDown, Check } from 'lucide-svelte';

  interface SelectOption {
    value: string;
    label: string;
    description?: string;
    disabled?: boolean;
    category?: string;
  }

  interface SelectProps {
    /** Selected value */
    value?: string;
    /** Callback when value changes */
    onValueChange?: (value: string) => void;
    /** Available options */
    options: SelectOption[];
    /** Placeholder text */
    placeholder?: string;
    /** Disabled state */
    disabled?: boolean;
    /** Legal context styling */
    legal?: boolean;
    /** Evidence category selection */
    evidenceCategory?: boolean;
    /** Case type selection */
    caseType?: boolean;
    /** AI confidence for recommendations */
    aiRecommendations?: boolean;
    /** Select size */
    size?: 'sm' | 'md' | 'lg';
    /** Error state */
    error?: boolean;
    /** Error message */
    errorMessage?: string;
    /** Full width */
    fullWidth?: boolean;
    /** Custom trigger class */
    triggerClass?: string;
    /** Custom content class */
    contentClass?: string;
  }

  let {
    value = $bindable(),
    onValueChange,
    options = [],
    placeholder = 'Select an option...',
    disabled = false,
    legal = false,
    evidenceCategory = false,
    caseType = false,
    aiRecommendations = false,
    size = 'md',
    error = false,
    errorMessage = '',
    fullWidth = false,
    triggerClass = '',
    contentClass = ''
  }: SelectProps = $props();

  // Group options by category if they have categories
  const groupedOptions = $derived(() => {
    const hasCategories = options.some(option => option.category);
    
    if (!hasCategories) {
      return { '': options };
    }

    return options.reduce((acc, option) => {
      const category = option.category || 'Other';
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(option);
      return acc;
    }, {} as Record<string, SelectOption[]>);
  });

  // Reactive trigger classes using $derived
  const triggerClasses = $derived(() => {
    const base = 'bits-select-trigger';
    
    const sizes = {
      sm: 'h-8 px-3 text-xs',
      md: 'h-10 px-3 text-sm',
      lg: 'h-12 px-4 text-base'
    };

    return cn(
      base,
      sizes[size],
      {
        'w-full': fullWidth,
        'nier-bits-select': legal,
        'yorha-input': evidenceCategory || caseType,
        'border-red-500 bg-red-50': error,
        'border-green-500 bg-green-50': aiRecommendations && value,
        'font-gothic tracking-wide': legal,
        'cursor-not-allowed opacity-50': disabled
      },
      triggerClass
    );
  });

  // Reactive content classes using $derived
  const selectContentClasses = $derived(() => {
    return cn(
      'bits-select-content',
      {
        'nier-panel-elevated shadow-xl': legal,
        'border-2 border-nier-border-primary': evidenceCategory,
        'yorha-card': caseType,
        'bg-gradient-to-b from-nier-bg-primary to-nier-bg-secondary': legal
      },
      contentClass
    );
  });

  // Handle value change
  function handleValueChange(newValue: string) {
    value = newValue;
    onValueChange?.(newValue);
  }

  // Get selected option label
  const selectedLabel = $derived(() => {
    const selectedOption = options.find(option => option.value === value);
    return selectedOption?.label || placeholder;
  });
</script>

<div class="select-wrapper" class:w-full={fullWidth}>
  <BitsSelect.Root {value} onValueChange={handleValueChange} {disabled} type="single">
    <BitsSelect.Trigger class={triggerClasses()}>
      <div class="select-value">
        {selectedLabel}
      </div>
      <div class="select-icon">
        <ChevronDown class="h-4 w-4 opacity-50" />
      </div>
    </BitsSelect.Trigger>

    <BitsSelect.Portal>
      <BitsSelect.Content class={selectContentClasses()}>
        <BitsSelect.Viewport class="p-1">
          {#each Object.entries(groupedOptions) as [category, categoryOptions]}
            {#if category && Object.keys(groupedOptions).length > 1}
              <BitsSelect.Group>
                <BitsSelect.Label class="px-2 py-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  {category}
                </BitsSelect.Label>
                {#each categoryOptions as option (option.value)}
                  {@render selectItem(option)}
                {/each}
              </BitsSelect.Group>
              {#if category !== Object.keys(groupedOptions)[Object.keys(groupedOptions).length - 1]}
                <BitsSelect.Separator class="h-px bg-border my-1" />
              {/if}
            {:else}
              {#each categoryOptions as option (option.value)}
                {@render selectItem(option)}
              {/each}
            {/if}
          {/each}
        </BitsSelect.Viewport>
      </BitsSelect.Content>
    </BitsSelect.Portal>
  </BitsSelect.Root>

  {#if error && errorMessage}
    <div class="mt-1 text-xs text-red-600 font-medium">
      {errorMessage}
    </div>
  {/if}
</div>

{#snippet selectItem(option: SelectOption)}
  <BitsSelect.Item
    value={option.value}
    disabled={option.disabled}
    class={cn(
      'bits-select-item',
      {
        'yorha-priority-high': evidenceCategory && option.value.includes('critical'),
        'yorha-priority-medium': evidenceCategory && option.value.includes('evidence'),
        'opacity-50 cursor-not-allowed': option.disabled,
        'font-gothic': legal
      }
    )}
  >
    <BitsSelect.ItemIndicator class="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <Check class="h-4 w-4" />
    </BitsSelect.ItemIndicator>
    
    <div class="pl-6">
      <BitsSelect.ItemText class="font-medium">
        {option.label}
      </BitsSelect.ItemText>
      {#if option.description}
        <div class="text-xs text-muted-foreground mt-0.5">
          {option.description}
        </div>
      {/if}
    </div>
  </BitsSelect.Item>
{/snippet}

<script lang="ts" context="module">
  export { BitsSelect as Select };
  
  // Re-export commonly used sub-components (only those that exist)
  export const SelectTrigger = BitsSelect.Trigger;
  export const SelectPortal = BitsSelect.Portal;
  export const SelectContent = BitsSelect.Content;
  export const SelectViewport = BitsSelect.Viewport;
  export const SelectItem = BitsSelect.Item;
  export const SelectGroup = BitsSelect.Group;
  
  // For compatibility, create fallback components for missing exports
  export const SelectValue = 'div';
  export const SelectIcon = 'div'; 
  export const SelectItemText = 'span';
  export const SelectItemIndicator = 'div';
  export const SelectLabel = 'div';
  export const SelectSeparator = 'div';
</script>

<style>
  /* @unocss-include */
  .select-wrapper {
    position: relative;
  }

  /* Enhanced select animations for legal AI context */
  :global(.bits-select-content) {
    animation: select-content-show 200ms cubic-bezier(0.16, 1, 0.3, 1);
  }

  @keyframes select-content-show {
    from {
      opacity: 0;
      transform: scale(0.96) translateY(-2px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  /* Legal AI specific styling */
  :global(.nier-bits-select) {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
    border: 2px solid var(--color-nier-border-secondary);
    transition: all 0.2s ease;
  }

  :global(.nier-bits-select:focus) {
    border-color: var(--color-nier-border-primary);
    box-shadow: 0 0 0 1px var(--color-nier-border-primary);
  }

  :global(.nier-panel-elevated) {
    box-shadow: 
      0 10px 15px -3px rgba(0, 0, 0, 0.1),
      0 4px 6px -2px rgba(0, 0, 0, 0.05),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
  }

  /* Evidence category specific styling */
  :global([data-evidence-category] .bits-select-item) {
    position: relative;
  }

  :global([data-evidence-category] .bits-select-item::before) {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 4px;
    height: 60%;
    background: var(--color-nier-accent-cool);
    border-radius: 2px;
    opacity: 0;
    transition: opacity 0.2s ease;
  }

  :global([data-evidence-category] .bits-select-item[data-highlighted]::before) {
    opacity: 1;
  }

  /* Case type specific styling */
  :global([data-case-type] .bits-select-content) {
    background-image: 
      radial-gradient(circle at 20% 80%, rgba(58, 55, 47, 0.05) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(58, 55, 47, 0.05) 0%, transparent 50%);
  }

  /* AI recommendations styling */
  :global([data-ai-recommendations] .bits-select-item) {
    transition: all 0.2s ease;
  }

  :global([data-ai-recommendations] .bits-select-item:hover) {
    background: linear-gradient(
      90deg,
      rgba(16, 185, 129, 0.1) 0%,
      transparent 100%
    );
  }

  /* Enhanced focus states for accessibility */
  :global(.bits-select-trigger:focus-visible) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: 2px;
  }

  :global(.bits-select-item:focus-visible) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: -2px;
  }

  /* Responsive adjustments */
  @media (max-width: 640px) {
    :global(.bits-select-content) {
      max-height: 60vh;
    }
  }
</style>