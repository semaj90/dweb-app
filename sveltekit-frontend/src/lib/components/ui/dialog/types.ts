import type {     ComponentProps, Snippet     } from 'svelte';
import type { Dialog } from 'bits-ui';

export interface DialogProps extends ComponentProps<typeof Dialog.Root> {
  children?: Snippet;
  class?: string;
}

export interface DialogContentProps extends ComponentProps<typeof Dialog.Content> {
  children?: Snippet;
  class?: string;
}

export interface DialogHeaderProps {
  children?: Snippet;
  class?: string;
}

export interface DialogTitleProps extends ComponentProps<typeof Dialog.Title> {
  children?: Snippet;
  class?: string;
}

export interface DialogDescriptionProps extends ComponentProps<typeof Dialog.Description> {
  children?: Snippet;
  class?: string;
}

export interface DialogFooterProps {
  children?: Snippet;
  class?: string;
}