<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    variant?: 'default' | 'secondary' | 'destructive' | 'outline';
    size?: 'default' | 'sm' | 'lg';
    class?: string;
    children?: Snippet;
  }

  let {
    variant = 'default',
    size = 'default',
    class: className = '',
    children
  }: Props = $props();

  const variants = {
    default: 'bg-primary text-primary-foreground',
    secondary: 'bg-secondary text-secondary-foreground',
    destructive: 'bg-destructive text-destructive-foreground',
    outline: 'border border-input'
  };

  const sizes = {
    default: 'px-2.5 py-0.5 text-xs',
    sm: 'px-2 py-0.5 text-xs',
    lg: 'px-3 py-1 text-sm'
  };

  let classes = $derived([
    'inline-flex items-center rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
    variants[variant],
    sizes[size],
    className
  ].filter(Boolean).join(' '));
</script>

<span class={classes}>
  {@render children?.()}
</span>
