// 🔧 Svelte 5 State Management Type Fixes
// Fixes $state(), $derived(), and runes syntax errors

import type {     Readable, Writable     } from 'svelte/store';
// Import types
import type {
  User,
  ChatSession,
  ChatMessage,
  UploadedFile,
  SearchResults,
  AIResponse,
  TokenUsage,
  ModelAvailability,
  Toast,
  PerformanceMetrics,
  AIModel,
  SearchFilters,
  SearchFacets
} from '$lib/types';

// =====================================================
// SVELTE 5 RUNES TYPES
// =====================================================

// Fix for $state() rune
declare global {
  function $state<T>(initial?: T): T;
  function $derived<T>(fn: () => T): T;
  function $effect(fn: () => void | (() => void)): void;
  function $props<T = Record<string, any>>(): T;
}

// =====================================================
// STATE STORE TYPES
// =====================================================

export interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  theme: 'light' | 'dark' | 'auto';
  sidebarOpen: boolean;
  currentPage: string;
  notifications: Notification[];
  isLoading: boolean;
  error: string | null;
}

export interface ChatState {
  sessions: ChatSession[];
  currentSession: ChatSession | null;
  isTyping: boolean;
  isConnected: boolean;
  messages: ChatMessage[];
  draft: string;
}

export interface FileUploadState {
  files: UploadedFile[];
  uploading: boolean;
  progress: number;
  error: string | null;
  completed: number;
  total: number;
}

export interface SearchState {
  query: string;
  results: SearchResults | null;
  isSearching: boolean;
  filters: SearchFilters;
  facets: SearchFacets;
  history: string[];
}

export interface AIState {
  models: AIModel[];
  currentModel: string;
  isProcessing: boolean;
  responses: AIResponse[];
  usage: TokenUsage;
  availability: ModelAvailability;
}

// =====================================================
// COMPONENT STATE TYPES
// =====================================================

export interface ComponentState {
  mounted: boolean;
  initialized: boolean;
  error: Error | null;
  loading: boolean;
  data: any;
}

export interface FormState {
  values: Record<string, any>;
  errors: Record<string, string>;
  touched: Record<string, boolean>;
  isValid: boolean;
  isSubmitting: boolean;
  isDirty: boolean;
}

export interface ModalState {
  isOpen: boolean;
  title: string;
  content: any;
  size: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  closable: boolean;
}

export interface ToastState {
  toasts: Toast[];
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  maxToasts: number;
}

// =====================================================
// STORE CREATOR UTILITIES
// =====================================================

export interface StoreOptions<T> {
  initial: T;
  persist?: boolean;
  key?: string;
  serialize?: (value: T) => string;
  deserialize?: (value: string) => T;
}

export interface AsyncStoreOptions<T> extends StoreOptions<T> {
  fetcher: () => Promise<T>;
  refetchInterval?: number;
  staleTime?: number;
}

// =====================================================
// XSTATE INTEGRATION TYPES
// =====================================================

export interface MachineState {
  value: string;
  context: any;
  matches: (value: string) => boolean;
  can: (event: string) => boolean;
  send: (event: any) => void;
}

export interface MachineConfig {
  id: string;
  initial: string;
  states: Record<string, any>;
  context?: unknown;
  on?: Record<string, any>;
}

// =====================================================
// FORM INTEGRATION TYPES
// =====================================================

export interface SuperFormsState<T = Record<string, any>> {
  form: Writable<T>;
  errors: Readable<Record<string, string[]>>;
  constraints: Readable<Record<string, any>>;
  message: Writable<any>;
  submitting: Readable<boolean>;
  delayed: Readable<boolean>;
  timeout: Readable<boolean>;
  posted: Readable<boolean>;
}

export interface ValidationConfig {
  required?: boolean;
  min?: number;
  max?: number;
  pattern?: RegExp;
  custom?: (value: any) => string | null;
}

// =====================================================
// PERFORMANCE MONITORING TYPES
// =====================================================

export interface PerformanceState {
  metrics: PerformanceMetrics;
  history: PerformanceEntry[];
  alerts: PerformanceAlert[];
  thresholds: PerformanceThresholds;
}

export interface PerformanceAlert {
  id: string;
  type: 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  metric: string;
  value: number;
  threshold: number;
}

export interface PerformanceThresholds {
  responseTime: number;
  memoryUsage: number;
  cpuUsage: number;
  errorRate: number;
}

// =====================================================
// WEBSOCKET STATE TYPES
// =====================================================

export interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  lastMessage: any;
  messageHistory: any[];
  reconnectAttempts: number;
  maxReconnectAttempts: number;
}

// =====================================================
// NAVIGATION STATE TYPES
// =====================================================

export interface NavigationState {
  currentPath: string;
  previousPath: string;
  breadcrumbs: Breadcrumb[];
  canGoBack: boolean;
  canGoForward: boolean;
  isNavigating: boolean;
}

export interface Breadcrumb {
  label: string;
  path: string;
  icon?: string;
}

// =====================================================
// IMPORT TYPE FIXES
// =====================================================

// Fix common type errors
export type { User } from './global';
// Orphaned content: export type { ChatSession, ChatMessage
export type { UploadedFile } from './global';
export type { SearchResults } from './global';
export type { AIResponse } from './global';

// Fix Svelte component types
export type SvelteComponent = import('svelte').SvelteComponent;
// Simplified component helper types (avoid generic instantiation as SvelteComponent isn't generic here)
export type ComponentProps<T extends SvelteComponent> = any;
export type ComponentEvents<T extends SvelteComponent> = any;

// Fix action types
export type Action<T = HTMLElement, P = any> = (
  node: T,
  parameters?: P
) => {
  update?: (parameters: P) => void;
  destroy?: () => void;
};

// =====================================================
// HELPER TYPES
// =====================================================

export type Prettify<T> = {
  [K in keyof T]: T[K];
} & {};

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Renamed to avoid clashing with built-in TypeScript NonNullable
export type NonNullish<T> = T extends null | undefined ? never : T;

export type NonNullable<T> = T extends null | undefined ? never : T;

export type EventHandler<T = Event> = (event: T) => void;

export type AsyncFunction<T = any> = (...args: any[]) => Promise<T>;
