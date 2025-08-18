// Drawer component barrel export
export { default as Drawer } from './drawer/drawer.svelte';
export { default as DrawerContent } from './drawer/drawer-content.svelte';
export { default as DrawerDescription } from './drawer/drawer-description.svelte';
export { default as DrawerFooter } from './drawer/drawer-footer.svelte';
export { default as DrawerHeader } from './drawer/drawer-header.svelte';
export { default as DrawerTitle } from './drawer/drawer-title.svelte';
export { default as DrawerTrigger } from './drawer/drawer-trigger.svelte';

// Types
export type DrawerProps = {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export type DrawerContentProps = {
  class?: string;
};

export type DrawerTriggerProps = {
  class?: string;
  asChild?: boolean;
};
