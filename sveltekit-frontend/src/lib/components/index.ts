/**
 * Master Component Barrel Export
 * Legal AI Platform - SvelteKit 2 + Svelte 5
 *
 * Comprehensive component wiring and modularity for all UI components
 */

// ===== CORE UI COMPONENTS =====
export * from './ui/index';

// ===== LAYOUT COMPONENTS =====
export * from './ui/layout/index';

// ===== ENHANCED UI COMPONENTS =====
// Note: Using selective exports to avoid conflicts with base UI components
export { Button as EnhancedButton, Card as EnhancedCard, Input as EnhancedInput } from './ui/enhanced/index';
// export { SelectOption as EnhancedSelectOption } from './ui/enhanced-bits/index'; // Disabled: empty file
export * from './ui/modern/index';

// ===== FORM COMPONENTS =====
// Selective form exports to avoid conflicts
export { Form as ComponentForm, FormStandard as ComponentFormStandard } from './ui/form/index';
export { Input as ComponentInput } from './ui/input/index';
// Textarea export handled by base UI components
// export * from './ui/textarea/index';
export * from './ui/checkbox/index';
export * from './ui/select/index';
// Use specific exports to avoid CommonProps conflict
export { Switch, SwitchRoot, SwitchThumb } from './ui/switch/index';
export type { SwitchProps } from './ui/switch/index';
export { Slider, SliderRoot, SliderRange, SliderThumb, SliderTick } from './ui/slider/index';
export type { SliderProps } from './ui/slider/index';

// ===== NAVIGATION & INTERACTION =====
export * from './ui/tabs/index';
// Context menu exports handled selectively to avoid conflicts
export { Trigger as ContextMenuTrigger, Content as ContextMenuContent, Item as ContextMenuItem } from './ui/context-menu/index';
export * from './ui/dialog/index';
// Tooltip exports - conditional based on availability
// export * from './ui/tooltip/index';
export * from './ui/command/index';

// ===== FEEDBACK & DISPLAY =====
export * from './ui/alert/index';
export * from './ui/badge/index';
export * from './ui/progress/index';
export * from './ui/scroll-area/index';
export * from './ui/separator/index';
export * from './ui/drawer/index';

// ===== YORHA THEME COMPONENTS =====
export * from './yorha/index';
export * from './three/yorha-ui/index';

// ===== SEARCH COMPONENTS =====
export * from './search/index';

// ===== UNIFIED COMPONENTS =====
export * from './unified/index';

// ===== BUSINESS LOGIC COMPONENTS =====

// AI & Chat Components
export { default as AIChat } from './AIChat.svelte';
export { default as EnhancedLegalChat } from './EnhancedLegalChat.svelte';
export { default as LLMAssistant } from './LLMAssistant.svelte';
export * from './ai/index';

// Case Management
export { default as CaseCard } from './+CaseCard.svelte';
export { default as LegalCaseManager } from './LegalCaseManager.svelte';
export * from './cases/index';

// Evidence & Legal
export { default as EvidencePanel } from './EvidencePanel.svelte';
export * from './legal/index';
export * from './evidence-editor/index';

// Chat & Messaging
export * from './chat/index';

// Canvas & Visual Editor
export * from './canvas/index';

// Authentication
export * from './auth/index';

// Layout & Navigation
export { default as NierNavigation } from './NierNavigation.svelte';
export * from './layout/index';

// Realtime Components
export * from './realtime/index';

// Editor Components
export * from './editor/index';

// Detective Interface
export * from './detective/index';

// Subcomponents
export * from './subcomponents/index';

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