<script lang="ts">
  let {
    variant = 'default',
    class: className = '',
    children,
    ...restProps
  } = $props();

  const variants = {
    default: 'bg-background text-foreground',
    destructive: 'border-destructive/50 text-destructive dark:border-destructive [&>svg]:text-destructive',
    warning: 'border-warning/50 text-warning dark:border-warning [&>svg]:text-warning',
  };
</script>

<div
  class="relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground {variants[variant]} {className}"
  role="alert"
  {...restProps}
>
  <slot></slot>
</div>
