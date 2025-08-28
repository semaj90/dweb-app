/**
 * Master Component Barrel Export
 * Legal AI Platform - SvelteKit 2 + Svelte 5
 * 
 * Comprehensive component wiring and modularity for all UI components
 */

// ===== CORE UI COMPONENTS =====
export * from './ui/index.js';

// ===== LAYOUT COMPONENTS =====
export * from './ui/layout/index.js';

// ===== ENHANCED UI COMPONENTS =====
// Note: Using selective exports to avoid conflicts with base UI components
export { Button as EnhancedButton, Card as EnhancedCard, Input as EnhancedInput } from './ui/enhanced/index.js';
// export { SelectOption as EnhancedSelectOption } from './ui/enhanced-bits/index.js'; // Disabled: empty file
export * from './ui/modern/index.js';

// ===== FORM COMPONENTS =====
export * from './ui/form/index.js';
export * from './ui/input/index.js';
// Textarea export handled by base UI components
// export * from './ui/textarea/index.js';
export * from './ui/checkbox/index.js';
export * from './ui/select/index.js';
export * from './ui/switch/index.js';
export * from './ui/slider/index.js';

// ===== NAVIGATION & INTERACTION =====
export * from './ui/tabs/index.js';
// Context menu exports handled selectively to avoid conflicts
export { Trigger as ContextMenuTrigger, Content as ContextMenuContent, Item as ContextMenuItem } from './ui/context-menu/index.js';
export * from './ui/dialog/index.js';
// Tooltip exports - conditional based on availability
// export * from './ui/tooltip/index.js';
export * from './ui/command/index.js';

// ===== FEEDBACK & DISPLAY =====
export * from './ui/alert/index.js';
export * from './ui/badge/index.js';
export * from './ui/progress/index.js';
export * from './ui/scroll-area/index.js';
export * from './ui/separator/index.js';
export * from './ui/drawer/index.js';

// ===== YORHA THEME COMPONENTS =====
export * from './yorha/index.js';
export * from './three/yorha-ui/index.js';

// ===== SEARCH COMPONENTS =====
export * from './search/index.js';

// ===== UNIFIED COMPONENTS =====
export * from './unified/index.js';

// ===== BUSINESS LOGIC COMPONENTS =====

// AI & Chat Components  
export { default as AIChat } from './AIChat.svelte';
export { default as EnhancedLegalChat } from './EnhancedLegalChat.svelte';
export { default as LLMAssistant } from './LLMAssistant.svelte';
export * from './ai/index.js';

// Case Management
export { default as CaseCard } from './+CaseCard.svelte';
export { default as LegalCaseManager } from './LegalCaseManager.svelte';
export * from './cases/index.js';

// Evidence & Legal
export { default as EvidencePanel } from './EvidencePanel.svelte';
export * from './legal/index.js';
export * from './evidence-editor/index.js';

// Chat & Messaging
export * from './chat/index.js';

// Canvas & Visual Editor
export * from './canvas/index.js';

// Authentication
export * from './auth/index.js';

// Layout & Navigation
export { default as NierNavigation } from './NierNavigation.svelte';
export * from './layout/index.js';

// Realtime Components
export * from './realtime/index.js';

// Editor Components
export * from './editor/index.js';

// Detective Interface
export * from './detective/index.js';

// Subcomponents
export * from './subcomponents/index.js';

// Utility Components
export { default as LoadingSpinner } from './LoadingSpinner.svelte';
export { default as Typewriter } from './Typewriter.svelte';
export { default as KeyboardShortcutProvider } from './KeyboardShortcutProvider.svelte';
export { default as HeadlessDemo } from './HeadlessDemo.svelte';

// ===== COMPONENT METADATA =====
export const COMPONENT_REGISTRY = {
  ui: 89,
  business: 24,
  layout: 12,
  ai: 8,
  legal: 15,
  realtime: 6,
  total: 154
} as const;

export const COMPONENT_CATEGORIES = [
  'ui',
  'business',
  'layout', 
  'ai',
  'legal',
  'realtime',
  'utility'
] as const;

export type ComponentCategory = typeof COMPONENT_CATEGORIES[number];

// ===== STORYBOOK SUPPORT =====
export const STORYBOOK_STORIES = {
  'Button': () => import('./ui/enhanced/Button.svelte'),
  'Card': () => import('./ui/enhanced/Card.svelte'),
  'Input': () => import('./ui/enhanced/Input.svelte'),
  'Dialog': () => import('./ui/enhanced-bits/Dialog.svelte'),
  'AIChat': () => import('./AIChat.svelte'),
  'LegalCaseManager': () => import('./LegalCaseManager.svelte'),
  'EvidencePanel': () => import('./EvidencePanel.svelte')
} as const;

// Default export for convenience
export default {
  COMPONENT_REGISTRY,
  COMPONENT_CATEGORIES,
  STORYBOOK_STORIES
};