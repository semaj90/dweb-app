// Legal Search Components
export { default as LegalSearchCombobox } from './LegalSearchCombobox.svelte';

// Export types for the components
export type {
  SearchResult,
  SearchOptions,
  SearchMetadata
} from './types';

// Re-export common search utilities
export {
  createSearchFilters,
  formatSearchResults,
  calculateRelevanceScore
} from './utils';