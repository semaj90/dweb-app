// Barrel exports to match usage in forms
export { default as Select } from "./Select.svelte";
export { default as SelectValue } from "./SelectValue.svelte";

// Re-export common Bits UI components under expected names
import { Select as BitsSelect } from "bits-ui";
export const SelectContent = BitsSelect.Content;
export const SelectItem = BitsSelect.Item;
export const SelectTrigger = BitsSelect.Trigger;
export const SelectPortal = BitsSelect.Portal;
export const SelectGroup = BitsSelect.Group;
export const SelectViewport = BitsSelect.Viewport;

// TypeScript interface definition
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}
