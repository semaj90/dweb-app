// 🔧 Component Type Definitions (resolved duplicate export conflicts)
// Centralizes commonly used UI component prop & event types

import type { UploadedFile } from './global';

// =====================================================
// FILE UPLOAD COMPONENT FIX
// =====================================================

export interface FileUploadSectionProps {
  onFileUpload?: (files: File[]) => void;
  accept?: string;
  maxFiles?: number;
  maxFileSize?: number;
}

// =====================================================
// COMPONENT PROPS INTERFACE FIXES
// =====================================================

export interface ComponentPropsBase {
  class?: string;
  id?: string;
  children?: import('svelte').Snippet;
  [key: string]: unknown;
}

export interface ButtonProps extends ComponentPropsBase {
  variant?: 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  size?: 'default' | 'sm' | 'lg' | 'xl';
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  onclick?: (event: MouseEvent) => void;
}

export interface DialogProps extends ComponentPropsBase {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export interface DropdownMenuItemProps extends ComponentPropsBase {
  onclick?: (event: MouseEvent) => void;
  disabled?: boolean;
}

export interface FileUploadProps extends ComponentPropsBase {
  accept?: string;
  maxFileSize?: number;
  maxFiles?: number;
  multiple?: boolean;
  onUpload?: (files: File[]) => void;
  onUploadComplete?: (files: UploadedFile[]) => void;
}

// =====================================================
// ENHANCED DOCUMENT TYPE FIX
// =====================================================

export interface LegalDocumentExtended {
  id: string;
  title: string;
  type: 'contract' | 'case_law' | 'statute' | 'regulation' | 'brief' | 'evidence';
  status: 'draft' | 'reviewed' | 'approved' | 'archived';
  created: string;
  updated?: string;
  content?: string;
  metadata?: Record<string, any>;
}

// (Removed global Document augmentation to avoid DOM conflicts)

// =====================================================
// EVENT HANDLER FIXES
// =====================================================

export type EventHandler<T = Event> = (event: T) => void;
export type ClickHandler = (event: MouseEvent) => void;
export type InputHandler = (event: Event & { currentTarget: HTMLInputElement }) => void;
export type ChangeHandler = (
  event: Event & { currentTarget: HTMLSelectElement | HTMLInputElement }
) => void;
export type SubmitHandler = (event: Event & { currentTarget: HTMLFormElement }) => void;

// =====================================================
// CUSTOM EVENT TYPES
// =====================================================

export interface CustomEventDetail<T = any> {
  detail: T;
}

export type CustomEventHandler<T = any> = (event: CustomEvent<T>) => void;

// =====================================================
// FORM STATE FIXES
// =====================================================

export interface FormFieldState {
  value: any;
  error?: string;
  touched: boolean;
  dirty: boolean;
}

export interface FormValidationState {
  isValid: boolean;
  isSubmitting: boolean;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  values: Record<string, any>;
}

// =====================================================
// SIDEBAR & LAYOUT STATE FIXES
// =====================================================

export interface SidebarState {
  open: boolean;
  collapsed: boolean;
  pinned: boolean;
}

export interface LayoutState {
  sidebar: SidebarState;
  theme: 'light' | 'dark' | 'auto';
  fullscreen: boolean;
}

// =====================================================
// SEARCH COMPONENT FIXES
// =====================================================

export interface SearchFilters {
  type?: string;
  caseId?: string;
  dateRange?: {
    start: string;
    end: string;
  };
  status?: string[];
  tags?: string[];
}

export interface SearchFacets {
  types: Array<{ name: string; count: number }>;
  statuses: Array<{ name: string; count: number }>;
  dates: Array<{ name: string; count: number }>;
}

// =====================================================
// AI & ML COMPONENT FIXES
// =====================================================

export interface AIModel {
  id: string;
  name: string;
  type: 'chat' | 'embedding' | 'completion';
  available: boolean;
  config: Record<string, any>;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost?: number;
}

export interface ModelAvailability {
  ollama: boolean;
  openai: boolean;
  claude: boolean;
  local: boolean;
}

// =====================================================
// NOTIFICATION FIXES
// =====================================================

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  actions?: Array<{
    label: string;
    action: () => void;
  }>;
}

export interface Notification extends Toast {
  read: boolean;
  timestamp: string;
  category: string;
}

// (All interfaces exported individually above; no aggregate export block to avoid conflicts)
