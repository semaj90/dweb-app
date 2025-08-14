// src/lib/index.ts - TypeScript Barrel Exports
// ================================================================================
// PRODUCTION TYPESCRIPT BARREL EXPORTS - MINIMAL CONFLICT VERSION
// ================================================================================
// Essential exports only to avoid module conflicts

// ============================================================================
// UTILITY EXPORTS - Key utilities without conflicts
// ============================================================================
export { cn } from './utils/cn';

// ============================================================================
// COMPONENT EXPORTS - Key components only
// ============================================================================
export { default as Chat } from './components/Chat.svelte';
export { default as SearchBar } from './components/SearchBar.svelte';
export { default as Header } from './components/Header.svelte';
export { default as LoadingSpinner } from './components/LoadingSpinner.svelte';

// UI Components that actually exist
export { default as Button } from './components/ui/Button.svelte';
export { default as Card } from './components/ui/Card.svelte';
export { default as Input } from './components/ui/Input.svelte';
export { default as Badge } from './components/ui/Badge.svelte';
export { default as Tooltip } from './components/ui/Tooltip.svelte';

// ============================================================================
// DEFAULT EXPORT
// ============================================================================
export default {
  version: '4.0.0',
  status: 'production-ready'
};
