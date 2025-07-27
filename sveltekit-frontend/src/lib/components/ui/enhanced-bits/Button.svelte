<script lang="ts">
  import { Button as BitsButton } from 'bits-ui';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import { cn } from '$lib/utils/cn';

  interface ButtonProps extends HTMLButtonAttributes {
    /** Button variant styling */
    variant?: 'default' | 'primary' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link' | 'yorha' | 'crimson' | 'gold';
    /** Button size */
    size?: 'sm' | 'md' | 'lg' | 'icon';
    /** Loading state */
    loading?: boolean;
    /** AI confidence level for legal analysis buttons */
    confidence?: 'high' | 'medium' | 'low';
    /** Legal context specific styling */
    legal?: boolean;
    /** Full width button */
    fullWidth?: boolean;
    /** Priority for evidence-related actions */
    priority?: 'critical' | 'high' | 'medium' | 'low';
    class?: string;
  }

  let {
    variant = 'default',
    size = 'md',
    loading = false,
    confidence,
    legal = false,
    fullWidth = false,
    priority,
    class: className = '',
    children,
    ...restProps
  }: ButtonProps = $props();

  // Reactive class computation using $derived
  const buttonClasses = $derived(() => {
    const base = 'bits-btn';
    
    const variants = {
      default: 'bits-btn-default',
      primary: 'yorha-button-primary',
      destructive: 'bits-btn-destructive',
      outline: 'bits-btn-outline yorha-button',
      secondary: 'bits-btn-secondary',
      ghost: 'bits-btn-ghost',
      link: 'bits-btn-link',
      yorha: 'yorha-button',
      crimson: 'yorha-button bg-red-600 hover:bg-red-700 text-white border-red-600',
      gold: 'yorha-button bg-yellow-600 hover:bg-yellow-700 text-white border-yellow-600'
    };

    const sizes = {
      sm: 'bits-btn-sm h-8 px-3 text-xs',
      md: 'h-10 px-4 py-2',
      lg: 'bits-btn-lg h-12 px-6 text-base',
      icon: 'bits-btn-icon'
    };

    const confidenceStyles = {
      high: 'ai-confidence-90 border-green-500 bg-green-50 hover:bg-green-100',
      medium: 'ai-confidence-70 border-yellow-500 bg-yellow-50 hover:bg-yellow-100',
      low: 'ai-confidence-40 border-red-500 bg-red-50 hover:bg-red-100'
    };

    const priorityStyles = {
      critical: 'yorha-priority-critical shadow-red-200',
      high: 'yorha-priority-high shadow-orange-200',
      medium: 'yorha-priority-medium shadow-yellow-200',
      low: 'yorha-priority-low shadow-gray-200'
    };

    return cn(
      base,
      variants[variant],
      sizes[size],
      {
        'w-full': fullWidth,
        'opacity-50 cursor-not-allowed': loading,
        'nier-bits-button': legal,
        'animate-pulse': loading,
        'font-gothic tracking-wider': variant === 'yorha',
        'shadow-lg hover:shadow-xl transition-all duration-200': !loading
      },
      confidence && confidenceStyles[confidence],
      priority && priorityStyles[priority],
      className
    );
  });
</script>

<BitsButton.Root
  class={buttonClasses()}
  disabled={loading}
  {...restProps}
>
  {#if loading}
    <div class="ai-status-indicator ai-status-processing w-4 h-4 mr-2"></div>
  {/if}
  {#if children}
    {@render children()}
  {:else}
    <slot />
  {/if}
</BitsButton.Root>

<style>
  /* @unocss-include */
  /* Enhanced button animations for legal AI context */
  :global(.bits-btn) {
    position: relative;
    overflow: hidden;
  }

  :global(.bits-btn::before) {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.2),
      transparent
    );
    transition: left 0.5s ease;
  }

  :global(.bits-btn:hover::before) {
    left: 100%;
  }

  /* Legal AI specific styling */
  :global(.nier-bits-button) {
    font-family: var(--font-gothic);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* Confidence indicators */
  :global(.ai-confidence-90) {
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
  }

  :global(.ai-confidence-70) {
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.3);
  }

  :global(.ai-confidence-40) {
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.3);
  }
</style>