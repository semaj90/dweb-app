<script lang="ts">
  import type { HTMLInputAttributes } from 'svelte/elements';
  import { cn } from '$lib/utils/cn';
  import { Search, Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-svelte';

  interface InputProps extends Omit<HTMLInputAttributes, 'size'> {
    /** Input variant */
    variant?: 'default' | 'search' | 'password' | 'email' | 'legal' | 'evidence';
    /** Input size */
    size?: 'sm' | 'md' | 'lg';
    /** Error state */
    error?: boolean;
    /** Success state */
    success?: boolean;
    /** Loading state */
    loading?: boolean;
    /** Error message */
    errorMessage?: string;
    /** Help text */
    helpText?: string;
    /** Label */
    label?: string;
    /** Required field indicator */
    required?: boolean;
    /** Legal context styling */
    legal?: boolean;
    /** Evidence search specific styling */
    evidenceSearch?: boolean;
    /** Case number input styling */
    caseNumber?: boolean;
    /** AI confidence for auto-completed fields */
    aiAssisted?: boolean;
    /** Full width */
    fullWidth?: boolean;
    /** Show character count */
    showCharCount?: boolean;
    /** Maximum character count */
    maxlength?: number;
    /** Icon to display */
    icon?: any;
    /** Icon position */
    iconPosition?: 'left' | 'right';
    class?: string;
  }

  let {
    variant = 'default',
    size = 'md',
    error = false,
    success = false,
    loading = false,
    errorMessage = '',
    helpText = '',
    label = '',
    required = false,
    legal = false,
    evidenceSearch = false,
    caseNumber = false,
    aiAssisted = false,
    fullWidth = false,
    showCharCount = false,
    maxlength,
    icon,
    iconPosition = 'left',
    class: className = '',
    type = 'text',
    value = $bindable(''),
    ...restProps
  } = $props();

  // Password visibility toggle for password inputs
  let showPassword = $state(false);
  let inputElement: HTMLInputElement | undefined = $state();

  // Determine if this is a password input
  const isPassword = $derived(variant === 'password' || type === 'password')
  
  // Actual input type to use
  const inputType = $derived(isPassword ? (showPassword ? 'text' : 'password') : type
  );

  // Character count
  const charCount = $derived(typeof value === 'string' ? value.length : 0)

  // Reactive input classes using $derived
  const inputClasses = $derived(cn(
    'bits-input',
    {
      'pl-10': variant === 'search' || (icon && iconPosition === 'left'),
      'pr-10': variant === 'password' || (icon && iconPosition === 'right'),
      'yorha-input font-gothic tracking-wide': variant === 'legal',
      'yorha-input border-2 border-nier-border-secondary': variant === 'evidence',
      'h-8 px-3 text-xs': size === 'sm',
      'h-10 px-3 text-sm': size === 'md',
      'h-12 px-4 text-base': size === 'lg',
      'w-full': fullWidth,
      'border-red-500 bg-red-50 focus:border-red-500 focus:ring-red-500': error,
      'border-green-500 bg-green-50 focus:border-green-500 focus:ring-green-500': success,
      'nier-bits-input': legal,
      'vector-search-input': evidenceSearch,
      'border-gothic-border-primary bg-gothic-bg-primary': caseNumber,
      'border-blue-500 bg-blue-50': aiAssisted,
      'animate-pulse': loading,
      'pr-16': showCharCount && maxlength,
      'pr-20': isPassword || (icon && iconPosition === 'right' && showCharCount)
    },
    className
  ));

  // Focus the input programmatically
  export function focus() {
    inputElement?.focus();
  }

  // Clear the input
  export function clear() {
    value = '';
    inputElement?.focus();
  }
</script>

<div class="input-wrapper" class:w-full={fullWidth}>
  {#if label}
    <label 
      for={restProps.id} 
      class={cn(
        'bits-label block text-sm font-medium mb-2',
        {
          'text-red-600': error,
          'text-green-600': success,
          'font-gothic tracking-wide': legal
        }
      )}
    >
      {label}
      {#if required}
        <span class="text-red-500 ml-1">*</span>
      {/if}
    </label>
  {/if}

  <div class="relative">
    <!-- Left icon -->
    {#if icon && iconPosition === 'left'}
      <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        {#if icon}
          {@const IconComponent = icon}
          <IconComponent class="h-4 w-4 text-muted-foreground" />
        {/if}
      </div>
    {/if}

    <!-- Search icon for search variant -->
    {#if variant === 'search'}
      <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search class="h-4 w-4 text-muted-foreground" />
      </div>
    {/if}

    <!-- Input field -->
    <input
      bind:this={inputElement}
      bind:value
      type={inputType}
      class={inputClasses}
      {maxlength}
      {...restProps}
    />

    <!-- Password visibility toggle -->
    {#if isPassword}
      <button
        type="button"
        class="absolute inset-y-0 right-0 pr-3 flex items-center"
        onclick={() => showPassword = !showPassword}
        tabindex="-1"
      >
        {#if showPassword}
          <EyeOff class="h-4 w-4 text-muted-foreground hover:text-foreground transition-colors" />
        {:else}
          <Eye class="h-4 w-4 text-muted-foreground hover:text-foreground transition-colors" />
        {/if}
      </button>
    {/if}

    <!-- Right icon -->
    {#if icon && iconPosition === 'right' && !isPassword}
      <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        {#if icon}
          {@const IconComponent = icon}
          <IconComponent class="h-4 w-4 text-muted-foreground" />
        {/if}
      </div>
    {/if}

    <!-- Status indicators -->
    {#if error}
      <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        <AlertCircle class="h-4 w-4 text-red-500" />
      </div>
    {:else if success}
      <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        <CheckCircle class="h-4 w-4 text-green-500" />
      </div>
    {/if}

    <!-- Character count -->
    {#if showCharCount && maxlength}
      <div class={cn(
        'absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none text-xs',
        {
          'text-red-500': charCount > maxlength * 0.9,
          'text-yellow-600': charCount > maxlength * 0.8,
          'text-muted-foreground': charCount <= maxlength * 0.8,
          'pr-12': isPassword
        }
      )}>
        {charCount}/{maxlength}
      </div>
    {/if}

    <!-- Loading indicator -->
    {#if loading}
      <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
        <div class="ai-status-indicator ai-status-processing w-4 h-4"></div>
      </div>
    {/if}
  </div>

  <!-- Help text or error message -->
  {#if errorMessage && error}
    <div class="mt-1 text-xs text-red-600 font-medium flex items-center gap-1">
      <AlertCircle class="h-3 w-3" />
      {errorMessage}
    </div>
  {:else if helpText}
    <div class="mt-1 text-xs text-muted-foreground">
      {helpText}
    </div>
  {/if}

  <!-- AI assistance indicator -->
  {#if aiAssisted}
    <div class="mt-1 text-xs text-blue-600 font-medium flex items-center gap-1">
      <div class="ai-status-indicator ai-status-online w-2 h-2"></div>
      AI-assisted field
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .input-wrapper {
    position: relative
  }

  /* Enhanced input styling for legal AI context */
  :global(.nier-bits-input) {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
    border: 2px solid var(--color-nier-border-secondary);
    transition: all 0.2s ease;
  }

  :global(.nier-bits-input:focus) {
    border-color: var(--color-nier-border-primary);
    box-shadow: 
      0 0 0 1px var(--color-nier-border-primary),
      0 0 0 3px rgba(58, 55, 47, 0.1);
  }

  /* Evidence search specific styling */
  :global(.vector-search-input) {
    background: linear-gradient(
      to right,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 50%,
      var(--color-nier-bg-primary) 100%
    );
    background-size: 200% 100%;
    animation: search-gradient 3s ease-in-out infinite;
  }

  @keyframes search-gradient {
    0%, 100% {
      background-position: 0% 50%;
    }
    50% {
      background-position: 100% 50%;
    }
  }

  /* Case number input styling */
  :global([data-case-number] .bits-input) {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase
  }

  /* AI-assisted field styling */
  :global([data-ai-assisted] .bits-input) {
    position: relative
  }

  :global([data-ai-assisted] .bits-input::before) {
    content: '';
    position: absolute
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(
      90deg,
      transparent,
      var(--color-ai-status-online),
      transparent
    );
    animation: ai-assistance 2s ease-in-out infinite;
  }

  @keyframes ai-assistance {
    0%, 100% {
      opacity: 0;
    }
    50% {
      opacity: 1;
    }
  }

  /* Enhanced focus states for accessibility */
  :global(.bits-input:focus-visible) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: 2px;
  }

  /* Password visibility button styling */
  :global(.bits-input + button) {
    transition: all 0.2s ease;
  }

  :global(.bits-input + button:hover) {
    background-color: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
  }

  /* Legal context enhancements */
  :global(.font-gothic) {
    font-family: var(--font-gothic);
  }

  /* Error state animations */
  :global(.bits-input[aria-invalid="true"]) {
    animation: input-error 0.3s ease-in-out;
  }

  @keyframes input-error {
    0%, 100% {
      transform: translateX(0);
    }
    25% {
      transform: translateX(-2px);
    }
    75% {
      transform: translateX(2px);
    }
  }

  /* Success state animations */
  :global(.bits-input[data-success="true"]) {
    animation: input-success 0.5s ease-in-out;
  }

  @keyframes input-success {
    0% {
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
    }
    70% {
      box-shadow: 0 0 0 4px rgba(16, 185, 129, 0);
    }
    100% {
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
  }
</style>