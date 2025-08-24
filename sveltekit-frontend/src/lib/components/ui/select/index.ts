// Barrel exports to match usage in forms
export { default as Select } from "./Select.svelte";
export { default as SelectValue } from "./SelectValue.svelte";

// Re-export common Bits UI components under expected names
import { Select as BitsSelect } from "bits-ui";
export const SelectContent = BitsSelect.Content;
export const SelectItem = BitsSelect.Item;
export const SelectTrigger = BitsSelect.Trigger;
export const SelectPortal = BitsSelect.Content; // Portal is typically part of Content in Bits UI v2
export const SelectGroup = BitsSelect.Group || BitsSelect.Content; // Group may be part of Content in v2
export const SelectViewport = BitsSelect.Content; // Viewport is typically part of Content

// TypeScript interface definition
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}
