<!-- Date Picker Component for Legal AI App -->
<script lang="ts">
  import { DatePicker } from 'bits-ui';
  import { Calendar, ChevronLeft, ChevronRight } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { $props, $derived } from 'svelte';

  export interface DatePickerProps {
    value?: Date;
    placeholder?: string;
    disabled?: boolean;
    required?: boolean;
    minDate?: Date;
    maxDate?: Date;
    label?: string;
    description?: string;
    error?: string;
    variant?: 'default' | 'legal' | 'deadline';
    showTime?: boolean;
    format?: string;
    class?: string;
    onValueChange?: (value: Date | undefined) => void;
  }

  let {
    value = $bindable(undefined),
    placeholder = 'Select date...',
    disabled = false,
    required = false,
    minDate,
    maxDate,
    label,
    description,
    error,
    variant = 'default',
    showTime = false,
    format = showTime ? 'MMM dd, yyyy HH:mm' : 'MMM dd, yyyy',
    class: className = '',
    onValueChange
  }: DatePickerProps = $props();

  const variantStyles = {
    default: 'border-yorha-border bg-yorha-bg-tertiary text-yorha-text-primary',
    legal: 'border-yorha-primary/30 bg-yorha-bg-secondary text-yorha-text-primary ring-yorha-primary',
    deadline: 'border-red-500/30 bg-red-500/5 text-yorha-text-primary ring-red-500/20'
  };

  // Format date for display
  const formattedDate = $derived(() => {
    if (!value) return placeholder;
    
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    };

    if (showTime) {
      options.hour = '2-digit';
      options.minute = '2-digit';
    }

    return value.toLocaleDateString('en-US', options);
  });

  // Check if date is a legal deadline (within 30 days)
  const isUpcomingDeadline = $derived(() => {
    if (!value || variant !== 'deadline') return false;
    const now = new Date();
    const diffTime = value.getTime() - now.getTime();
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays <= 30 && diffDays >= 0;
  });

  function handleValueChange(newValue: Date | undefined) {
    value = newValue;
    onValueChange?.(newValue);
  }

  // Generate unique ID for accessibility
  const inputId = `datepicker-${Math.random().toString(36).substr(2, 9)}`;
</script>

<div class="legal-datepicker-container w-full space-y-2">
  <!-- Label -->
  {#if label}
    <label 
      for={inputId}
      class={cn(
        "block text-sm font-medium font-mono",
        variant === 'deadline' ? 'text-red-400' : 'text-yorha-text-primary'
      )}
    >
      {label}
      {#if required}
        <span class="text-yorha-accent ml-1">*</span>
      {/if}
      {#if isUpcomingDeadline}
        <span class="ml-2 text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
          DEADLINE
        </span>
      {/if}
    </label>
  {/if}

  <DatePicker.Root 
    bind:value
    onValueChange={handleValueChange}
    {disabled}
    {minValue: minDate}
    {maxValue: maxDate}
  >
    <!-- Trigger Button -->
    <DatePicker.Trigger
      id={inputId}
      class={cn(
        'flex h-9 w-full items-center justify-between whitespace-nowrap rounded-md border px-3 py-2 text-sm font-mono shadow-sm ring-offset-background focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
        variantStyles[variant],
        error && 'border-red-500 focus:ring-red-500',
        className
      )}
      {required}
    >
      <span class={cn(
        value ? 'text-yorha-text-primary' : 'text-yorha-text-secondary'
      )}>
        {formattedDate}
      </span>
      <Calendar class="h-4 w-4 opacity-50" />
    </DatePicker.Trigger>

    <!-- Calendar Content -->
    <DatePicker.Content
      class="z-50 min-w-[320px] overflow-hidden rounded-md border bg-popover p-0 text-popover-foreground shadow-md border-yorha-border bg-yorha-bg-secondary"
      sideOffset={4}
    >
      <!-- Calendar Header -->
      <DatePicker.Header class="flex items-center justify-between p-3 border-b border-yorha-border">
        <DatePicker.PrevButton
          class="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
        >
          <ChevronLeft class="h-4 w-4" />
        </DatePicker.PrevButton>

        <DatePicker.Heading class="text-sm font-medium font-mono text-yorha-text-primary" />

        <DatePicker.NextButton
          class="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100"
        >
          <ChevronRight class="h-4 w-4" />
        </DatePicker.NextButton>
      </DatePicker.Header>

      <!-- Calendar Grid -->
      <DatePicker.Grid class="p-3">
        <!-- Days of Week Header -->
        <DatePicker.GridHead>
          <DatePicker.GridRow class="flex">
            {#each ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'] as day}
              <DatePicker.HeadCell 
                class="text-muted-foreground w-9 rounded-md text-[0.8rem] font-normal font-mono text-center text-yorha-text-secondary"
              >
                {day}
              </DatePicker.HeadCell>
            {/each}
          </DatePicker.GridRow>
        </DatePicker.GridHead>

        <!-- Calendar Body -->
        <DatePicker.GridBody>
          {#each Array(6) as _, weekIndex}
            <DatePicker.GridRow class="flex w-full mt-2">
              {#each Array(7) as _, dayIndex}
                <DatePicker.Cell class="text-center text-sm relative p-0 focus-within:relative focus-within:z-20">
                  <DatePicker.Day
                    class={cn(
                      'inline-flex h-9 w-9 items-center justify-center rounded-md text-sm font-normal font-mono ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
                      'hover:bg-yorha-bg-tertiary',
                      'data-[selected]:bg-yorha-primary data-[selected]:text-yorha-bg-primary data-[selected]:font-medium',
                      'data-[today]:bg-yorha-accent/10 data-[today]:text-yorha-accent',
                      'data-[outside-month]:text-yorha-text-secondary data-[outside-month]:opacity-50'
                    )}
                  />
                </DatePicker.Cell>
              {/each}
            </DatePicker.GridRow>
          {/each}
        </DatePicker.GridBody>
      </DatePicker.Grid>

      <!-- Time Picker (if enabled) -->
      {#if showTime}
        <div class="border-t border-yorha-border p-3">
          <div class="flex items-center justify-center gap-2">
            <DatePicker.TimeField>
              <DatePicker.TimeSegment 
                segment="hour"
                class="px-2 py-1 text-sm font-mono bg-yorha-bg-tertiary rounded focus:bg-yorha-primary focus:text-yorha-bg-primary outline-none"
              />
              <span class="text-yorha-text-secondary">:</span>
              <DatePicker.TimeSegment 
                segment="minute"
                class="px-2 py-1 text-sm font-mono bg-yorha-bg-tertiary rounded focus:bg-yorha-primary focus:text-yorha-bg-primary outline-none"
              />
            </DatePicker.TimeField>
          </div>
        </div>
      {/if}
    </DatePicker.Content>
  </DatePicker.Root>

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

  <!-- Legal Deadline Warning -->
  {#if variant === 'deadline' && value && isUpcomingDeadline}
    <div class="mt-2 p-2 bg-red-500/10 border border-red-500/20 rounded text-xs font-mono text-red-400">
      ⚠️ Deadline approaching: {Math.ceil((value.getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24))} days remaining
    </div>
  {/if}
</div>

<style>
  :global(.legal-datepicker-container) {
    --date-picker-bg: rgb(var(--yorha-bg-secondary));
    --date-picker-border: rgb(var(--yorha-border));
    --date-picker-text: rgb(var(--yorha-text-primary));
  }
</style>