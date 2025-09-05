<script lang="ts">
  interface Props {
    variant?: 'default' | 'secondary' | 'destructive' | 'outline';
    size?: 'default' | 'sm' | 'lg';
    class?: string;
    children?: any;
  }

  let {
    variant = 'default',
    size = 'default',
    class: className = '',
    children
  } = $props();

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

  const classes = $derived([
    'mx-auto px-4 max-w-7xl',
    variants[variant],
    sizes[size],
    className
  ].filter(Boolean).join(' '));
</script>

<span class={classes}>
  {@render children?.()}
</span>
