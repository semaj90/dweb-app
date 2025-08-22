// Switch Component Barrel Export
import { Switch } from "bits-ui";

export const SwitchRoot = Switch.Root;
export const SwitchThumb = Switch.Thumb;

// Re-export the whole Switch module
export { Switch };

// TypeScript interface for Switch props
export interface SwitchProps extends CommonProps {
  checked?: boolean;
  disabled?: boolean;
  onCheckedChange?: (checked: boolean) => void;
  size?: "sm" | "md" | "lg";
  variant?: "default" | "destructive";
}
