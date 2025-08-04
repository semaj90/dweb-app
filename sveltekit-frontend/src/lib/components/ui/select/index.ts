// Select Component Barrel Export
export { default as Select } from "./Select.svelte";

// Re-export from enhanced-bits for better integration
export { default as EnhancedSelect } from "../enhanced-bits/Select.svelte";

// Standard Bits UI Select components
export { Select as BitsSelect } from "bits-ui";

// Individual Select components from bits-ui - using correct component access pattern
import { Select } from "bits-ui";

export const SelectRoot = Select.Root;
export const SelectContent = Select.Content;
export const SelectItem = Select.Item;
export const SelectTrigger = Select.Trigger;
export const SelectGroup = Select.Group;
export const SelectGroupHeading = Select.GroupHeading;
export const SelectViewport = Select.Viewport;
export const SelectPortal = Select.Portal;
export const SelectScrollUpButton = Select.ScrollUpButton;
export const SelectScrollDownButton = Select.ScrollDownButton;

// Local components that aren't in bits-ui
export { default as SelectValue } from "./SelectValue.svelte";

// TypeScript interface definition
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}
