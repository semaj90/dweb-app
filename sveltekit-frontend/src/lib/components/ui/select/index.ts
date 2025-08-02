// Select Component Barrel Export
export { default as Select } from './Select.svelte';

// Re-export from enhanced-bits for better integration  
export { default as EnhancedSelect } from '../enhanced-bits/Select.svelte';

// Standard Bits UI Select components
export { Select as BitsSelect } from 'bits-ui';

// TypeScript interface definition
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}