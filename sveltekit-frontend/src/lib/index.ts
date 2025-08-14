// Enhanced barrel exports with YoRHa Detective AI components
// SvelteKit 2.0 + Svelte 5 + bits-ui + melt-ui + shadcn-svelte

// YoRHa Detective AI Components
export { default as YorhaAIAssistant } from './components/ai/YorhaAIAssistant.svelte';

// Application Components
export { default as Header } from './components/Header.svelte';
export { default as Sidebar } from './components/Sidebar.svelte';
export { default as SearchBar } from './components/SearchBar.svelte';
export { default as Modal } from './components/Modal.svelte';
export { default as CaseCard } from './components/CaseCard.svelte';
export { default as EvidenceUpload } from './components/EvidenceUpload.svelte';
export { default as FileUploadSection } from './components/FileUploadSection.svelte';
export { default as AutomateUploadSection } from './components/AutomateUploadSection.svelte';
export { default as AddNotesSection } from './components/AddNotesSection.svelte';
export { default as Dropdown } from './components/Dropdown.svelte';
export { default as Checkbox } from './components/Checkbox.svelte';
export { default as MarkdownEditor } from './components/MarkdownEditor.svelte';

// UI Components
export * from './ui';

// Stores (without conflicting Evidence exports)
export { contextMenuStore, contextMenuActions } from './stores/ui';
export { uiStore } from './stores/ui';
export { default as modalStore } from './stores/modal';
export { notifications as notificationStore } from './stores/notification';
export { default as authStore } from './stores/auth';
export { default as userStore } from './stores/user';
export { avatarStore } from './stores/avatarStore';
export { default as casesStore } from './stores/cases';
export { default as citationsStore } from './stores/citations';
export { evidenceActions, evidenceGrid, filteredEvidence } from './stores/evidence-store';
export { evidenceStore as unifiedEvidenceStore } from './stores/evidence-unified';

// Types (specific exports to avoid conflicts)
export type { Database, API, Config, ButtonVariant, ButtonSize } from './types';

// Utils
export * from './utils';
