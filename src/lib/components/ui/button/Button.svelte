<script lang="ts">
  import type { HTMLAttributes } from 'svelte/elements';
  
  interface Props extends HTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'outline' | 'ghost' | 'destructive' | 'secondary';
    size?: 'sm' | 'default' | 'lg';
    disabled?: boolean;
    className?: string;
    onclick?: () => void;
  }
  
  export let variant: 'default' | 'outline' | 'ghost' | 'destructive' | 'secondary' = 'default';
  export let size: 'sm' | 'default' | 'lg' = 'default';
  export let disabled: boolean = false;
  export let className: string = '';
  export let onclick: (() => void) | undefined = undefined;
  
  function getVariantClasses(variant: string) {
    const variants = {
      default: 'bg-blue-600 text-white hover:bg-blue-700 border-blue-600',
      outline: 'border border-gray-300 bg-transparent hover:bg-gray-50 text-gray-900',
      ghost: 'bg-transparent hover:bg-gray-100 text-gray-900',
      destructive: 'bg-red-600 text-white hover:bg-red-700 border-red-600',
      secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300 border-gray-200'
    };
    return variants[variant] || variants.default;
  }
  
  function getSizeClasses(size: string) {
    const sizes = {
      sm: 'px-3 py-1.5 text-sm',
      default: 'px-4 py-2 text-base',
      lg: 'px-6 py-3 text-lg'
    };
    return sizes[size] || sizes.default;
  }
</script>

<button
  {disabled}
  onclick={onclick}
  class="inline-flex items-center justify-center font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed {getVariantClasses(variant)} {getSizeClasses(size)} {className}"
>
  <slot />
</button>