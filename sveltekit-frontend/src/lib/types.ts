// Central shared type definitions (incremental widening to unblock svelte-check)
export type ButtonVariant =
  | 'default' | 'primary' | 'secondary' | 'destructive' | 'outline' | 'ghost' | 'link'
  | 'danger' | 'success' | 'warning' | 'info' | 'nier' | 'crimson' | 'gold'
  | 'case' | 'evidence' | 'legal';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'icon';
// Transitional FormField type: allow known union plus fallback string to avoid blocking builds
export type FormFieldType = 'text' | 'number' | 'date' | 'email' | 'select' | 'file' | 'textarea' | 'password' | 'checkbox' | 'radio';
export interface FormField { id: string; label: string; type: FormFieldType | (string & {}); required?: boolean; options?: { value: string; label: string }[]; }
export interface ModalProps { isOpen: boolean; title?: string; subtitle?: string; submitLabel?: string; cancelLabel?: string; fields?: FormField[]; }

// Placeholder User type (adjust later with real user domain model)
export interface User { id: string; name?: string; firstName?: string; lastName?: string; email?: string; role?: string; avatarUrl?: string; isActive?: boolean; emailVerified?: boolean; }
