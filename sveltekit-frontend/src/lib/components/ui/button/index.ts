// Button Component Barrel Export
export { default as Button } from './Button.svelte';
export { buttonVariants, type ButtonVariants } from '../enhanced/button-variants';

// TypeScript types
export type ButtonProps = {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link' | 'nier' | 'crimson' | 'gold';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  disabled?: boolean;
  class?: string;
};
