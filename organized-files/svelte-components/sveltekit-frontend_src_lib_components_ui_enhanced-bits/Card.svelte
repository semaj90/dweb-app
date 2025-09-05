<script lang="ts">
  import { cn } from '$lib/utils/cn';

  interface CardProps {
    /** Card variant */
    variant?: 'default' | 'elevated' | 'outline' | 'ghost' | 'yorha' | 'evidence' | 'case';
    /** Card size */
    size?: 'sm' | 'md' | 'lg' | 'xl';
    /** Hover effects */
    hoverable?: boolean;
    /** Clickable card */
    clickable?: boolean;
    /** Loading state */
    loading?: boolean;
    /** Selected state */
    selected?: boolean;
    /** Legal context styling */
    legal?: boolean;
    /** Evidence card specific styling */
    evidenceCard?: boolean;
    /** Case card specific styling */
    caseCard?: boolean;
    /** AI analysis card */
    aiAnalysis?: boolean;
    /** Priority level for evidence cards */
    priority?: 'critical' | 'high' | 'medium' | 'low';
    /** Confidence level for AI analysis */
    confidence?: 'high' | 'medium' | 'low';
    /** Full width */
    fullWidth?: boolean;
    /** Custom class */
    class?: string;
    /** Click handler */
    onclick?: () => void;
    /** Children content */
    children?: import('svelte').Snippet;
  }

  let {
    variant = 'default',
    size = 'md',
    hoverable = false,
    clickable = false,
    loading = false,
    selected = false,
    legal = false,
    evidenceCard = false,
    caseCard = false,
    aiAnalysis = false,
    priority,
    confidence,
    fullWidth = false,
    class: className = '',
    onclick,
    children
  } = $props();

  // Reactive card classes using $derived
  const cardClasses = $derived(cn(
    'shadcn-card',
    {
      'yorha-card-elevated shadow-lg': variant === 'elevated',
      'border-2': variant === 'outline',
      'border-none shadow-none bg-transparent': variant === 'ghost',
      'yorha-card': variant === 'yorha',
      'yorha-evidence-item': variant === 'evidence',
      'yorha-card yorha-panel': variant === 'case',
      'p-4': size === 'sm',
      'p-6': size === 'md',
      'p-8': size === 'lg',
      'p-10': size === 'xl',
      'w-full': fullWidth,
      'cursor-pointer': clickable,
      'hover:shadow-lg hover:-translate-y-1 transition-all duration-200': hoverable,
      'yorha-evidence-item-selected': selected && evidenceCard,
      'border-2 border-nier-border-primary': selected,
      'nier-bits-card': legal,
      'animate-pulse': loading,
      'font-gothic': legal,
      'vector-result-item': evidenceCard,
      'agent-card': aiAnalysis,
      'result-card': aiAnalysis,
      'recommendation-card': aiAnalysis && confidence,
      'yorha-priority-critical shadow-red-200 border-red-500': priority === 'critical',
      'yorha-priority-high shadow-orange-200 border-orange-500': priority === 'high',
      'yorha-priority-medium shadow-yellow-200 border-yellow-500': priority === 'medium',
      'yorha-priority-low shadow-gray-200 border-gray-300': priority === 'low',
      'ai-confidence-90 border-green-500': confidence === 'high',
      'ai-confidence-70 border-yellow-500': confidence === 'medium',
      'ai-confidence-40 border-red-500': confidence === 'low'
    },
    className));

  // Handle click events
  function handleClick() {
    if (clickable && onclick) {
      onclick();
    }
  }

  // Handle keyboard events for accessibility
  function handleKeydown(event: KeyboardEvent) {
    if (clickable && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      handleClick();
    }
  }
</script>

<div
  class={cardClasses}
  role={clickable ? 'button' : undefined}
  tabindex={clickable ? 0 : undefined}
  onclick={handleClick}
  onkeydown={handleKeydown}
  data-evidence-card={evidenceCard}
  data-case-card={caseCard}
  data-ai-analysis={aiAnalysis}
  data-selected={selected}
  data-loading={loading}
>
  {#if loading}
    <div class="processing-overlay">
      <div class="ai-status-indicator ai-status-processing w-8 h-8"></div>
    </div>
  {/if}

  {@render children?.()}

  <!-- Priority indicator for evidence cards -->
  {#if evidenceCard && priority}
    <div class="absolute top-2 right-2">
      <div class={cn(
        'w-3 h-3 rounded-full',
        {
          'bg-red-500': priority === 'critical',
          'bg-orange-500': priority === 'high',
          'bg-yellow-500': priority === 'medium',
          'bg-gray-400': priority === 'low'
        }
      )}></div>
    </div>
  {/if}

  <!-- Confidence indicator for AI analysis cards -->
  {#if aiAnalysis && confidence}
    <div class="absolute top-2 right-2">
      <div class={cn(
        'vector-confidence-badge',
        {
          'vector-confidence-high': confidence === 'high',
          'vector-confidence-medium': confidence === 'medium',
          'vector-confidence-low': confidence === 'low'
        }
      )}>
        {confidence.toUpperCase()}
      </div>
    </div>
  {/if}

  <!-- Selection indicator -->
  {#if selected}
    <div class="absolute inset-0 pointer-events-none">
      <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-nier-accent-warm to-nier-accent-cool"></div>
    </div>
  {/if}
</div>



<style>
  /* @unocss-include */
  /* Enhanced card animations for legal AI context */
  :global(.shadcn-card) {
    position: relative
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Legal AI specific styling */
  :global(.nier-bits-card) {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
    border: 1px solid var(--color-nier-border-secondary);
  }

  :global(.nier-bits-card::before) {
    content: '';
    position: absolute
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(
      90deg,
      var(--color-nier-accent-warm),
      var(--color-nier-accent-cool),
      var(--color-nier-accent-warm)
    );
    opacity: 0;
    transition: opacity 0.3s ease;
  }

  :global(.nier-bits-card:hover::before) {
    opacity: 1;
  }

  /* Evidence card specific styling */
  :global([data-evidence-card="true"]) {
    background-image: 
      linear-gradient(45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(-45deg, transparent 25%, rgba(0,0,0,0.02) 25%),
      linear-gradient(45deg, rgba(0,0,0,0.02) 75%, transparent 75%),
      linear-gradient(-45deg, rgba(0,0,0,0.02) 75%, transparent 75%);
    background-size: 12px 12px;
    background-position: 0 0, 0 6px, 6px -6px, -6px 0px;
  }

  /* Case card specific styling */
  :global([data-case-card="true"]) {
    border-left: 4px solid var(--color-nier-accent-cool);
  }

  :global([data-case-card="true"]::after) {
    content: '';
    position: absolute
    bottom: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent,
      var(--color-nier-border-primary),
      transparent
    );
  }

  /* AI analysis card styling */
  :global([data-ai-analysis="true"]) {
    background: linear-gradient(
      135deg,
      rgba(59, 130, 246, 0.05) 0%,
      rgba(16, 185, 129, 0.05) 100%
    );
    border: 1px solid rgba(59, 130, 246, 0.2);
  }

  :global([data-ai-analysis="true"]::before) {
    content: '';
    position: absolute
    top: 8px;
    left: 8px;
    width: 8px;
    height: 8px;
    background: var(--color-ai-status-online);
    border-radius: 50%;
    animation: ai-pulse 2s ease-in-out infinite;
  }

  @keyframes ai-pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(1.2);
    }
  }

  /* Selected state styling */
  :global([data-selected="true"]) {
    box-shadow: 
      0 0 0 2px var(--color-nier-border-primary),
      0 4px 12px rgba(58, 55, 47, 0.15);
    transform: translateY(-2px);
  }

  /* Loading state styling */
  :global([data-loading="true"]) {
    overflow: hidden
  }

  :global([data-loading="true"]::after) {
    content: '';
    position: absolute
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
      90deg,
      transparent,
      rgba(255, 255, 255, 0.4),
      transparent
    );
    animation: loading-shimmer 1.5s ease-in-out infinite;
  }

  @keyframes loading-shimmer {
    0% {
      left: -100%;
    }
    100% {
      left: 100%;
    }
  }

  /* Hover effects for clickable cards */
  :global(.shadcn-card[role="button"]:hover) {
    transform: translateY(-4px);
    box-shadow: 
      0 10px 25px -5px rgba(0, 0, 0, 0.1),
      0 4px 6px -2px rgba(0, 0, 0, 0.05);
  }

  :global(.shadcn-card[role="button"]:active) {
    transform: translateY(-2px);
    transition-duration: 0.1s;
  }

  /* Focus states for accessibility */
  :global(.shadcn-card[role="button"]:focus-visible) {
    outline: 2px solid var(--color-nier-border-primary);
    outline-offset: 2px;
  }

  /* Priority styling enhancements */
  :global(.yorha-priority-critical) {
    border-left-width: 6px;
    background: linear-gradient(
      90deg,
      rgba(239, 68, 68, 0.05) 0%,
      transparent 20%
    );
  }

  :global(.yorha-priority-high) {
    border-left-width: 4px;
    background: linear-gradient(
      90deg,
      rgba(245, 158, 11, 0.05) 0%,
      transparent 20%
    );
  }

  /* Confidence styling enhancements */
  :global(.ai-confidence-90) {
    box-shadow: inset 0 0 0 1px rgba(16, 185, 129, 0.3);
  }

  :global(.ai-confidence-70) {
    box-shadow: inset 0 0 0 1px rgba(245, 158, 11, 0.3);
  }

  :global(.ai-confidence-40) {
    box-shadow: inset 0 0 0 1px rgba(239, 68, 68, 0.3);
  }

  /* Responsive adjustments */
  @media (max-width: 640px) {
    :global(.shadcn-card) {
      margin: 0.5rem;
    }
  }
</style>