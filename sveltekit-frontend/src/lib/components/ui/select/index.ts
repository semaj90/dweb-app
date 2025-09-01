// Barrel exports to match usage in forms
export { default as Select } from "./Select.svelte";
export { default as SelectValue } from "./SelectValue.svelte";

// Re-export common Bits UI components under expected names
import { Select } from "bits-ui";
export const SelectContent = Select.Content;
export const SelectItem = Select.Item;
export const SelectTrigger = Select.Trigger;
// Note: Portal and Viewport may not exist in bits-ui Select
export const SelectGroup = Select.Group;
;
// TypeScript interface definition
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}
