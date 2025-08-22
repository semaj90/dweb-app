import type { Writable } from "svelte/store";


export interface SelectContext {
  selected: Writable<any>;
  open: Writable<boolean>;
  onSelect: (value: any) => void;
  onToggle: () => void;
}
export interface SelectItemProps extends CommonProps {
  value: any;
  class_?: string;
  selected?: boolean;
}
export interface SelectProps {
  value?: unknown;
  onValueChange?: (value: any) => void;
  disabled?: boolean;
  class_?: string;
}
