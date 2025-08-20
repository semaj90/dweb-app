import type { CommonProps } from '$lib/types/common-props';
import type { Writable } from "svelte/store";
// @ts-nocheck

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
  value?: any;
  onValueChange?: (value: any) => void;
  disabled?: boolean;
  class_?: string;
}
