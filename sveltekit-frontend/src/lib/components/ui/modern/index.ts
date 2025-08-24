/**
 * Modern UI Components - SvelteKit 2 + Svelte 5 + Bits-UI + Melt-UI
 * Golden ratio spacing, CSS Grid/Flexbox, no prop drilling
 */

export { default as ModernCard } from './ModernCard.svelte';
export { default as ModernButton } from './ModernButton.svelte';
export { default as ModernDialog } from './ModernDialog.svelte';

// Type exports for better TypeScript support (Svelte 5 compatible)
import type ModernCard from './ModernCard.svelte';
import type ModernButton from './ModernButton.svelte';
import type ModernDialog from './ModernDialog.svelte';

export type ModernCardProps = InstanceType<typeof ModernCard>['$$props'];
export type ModernButtonProps = InstanceType<typeof ModernButton>['$$props'];
export type ModernDialogProps = InstanceType<typeof ModernDialog>['$$props'];

// Component variants for easier usage
export const cardVariants = [
  'default',
  'elevated', 
  'outline',
  'ghost'
] as const;

export const buttonVariants = [
  'primary',
  'secondary',
  'ghost',
  'outline',
  'danger',
  'success'
] as const;

export const sizes = [
  'xs',
  'sm', 
  'md',
  'lg',
  'xl'
] as const;

// Utility functions for golden ratio calculations
export const goldenRatio = 1.618;

export function goldenSpacing(multiplier: number = 1): string {
  return `calc(1rem * ${goldenRatio * multiplier})`;
}

export function goldenScale(base: number, level: number): number {
  return base * Math.pow(goldenRatio, level);
}

// CSS custom property helpers
export const cssVars = {
  // Golden ratio spacing
  goldenXs: 'var(--golden-xs)',
  goldenSm: 'var(--golden-sm)',
  goldenMd: 'var(--golden-md)',
  goldenLg: 'var(--golden-lg)',
  goldenXl: 'var(--golden-xl)',
  golden2xl: 'var(--golden-2xl)',
  golden3xl: 'var(--golden-3xl)',
  
  // Typography
  textXs: 'var(--text-xs)',
  textSm: 'var(--text-sm)',
  textBase: 'var(--text-base)',
  textLg: 'var(--text-lg)',
  textXl: 'var(--text-xl)',
  text2xl: 'var(--text-2xl)',
  text3xl: 'var(--text-3xl)',
  text4xl: 'var(--text-4xl)',
  text5xl: 'var(--text-5xl)',
  
  // YoRHa colors
  bgPrimary: 'var(--yorha-bg-primary)',
  bgSecondary: 'var(--yorha-bg-secondary)',
  bgTertiary: 'var(--yorha-bg-tertiary)',
  bgCard: 'var(--yorha-bg-card)',
  bgHover: 'var(--yorha-bg-hover)',
  
  textPrimary: 'var(--yorha-text-primary)',
  textSecondary: 'var(--yorha-text-secondary)',
  textMuted: 'var(--yorha-text-muted)',
  
  accentGold: 'var(--yorha-accent-gold)',
  accentBlue: 'var(--yorha-accent-blue)',
  accentGreen: 'var(--yorha-accent-green)',
  
  borderPrimary: 'var(--yorha-border-primary)',
  borderSecondary: 'var(--yorha-border-secondary)',
  borderAccent: 'var(--yorha-border-accent)'
} as const;