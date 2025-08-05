/**
 * Enhanced Store Barrel Exports - TypeScript Store Pattern
 * Centralized store management with XState integration
 */

// Core UI stores
export { contextMenuStore, contextMenuActions } from "./ui";
export { uiStore } from "./ui";
export { default as modalStore } from "./modal";
export { notifications as notificationStore } from "./notification";

// Authentication & User stores  
export { default as authStore } from "./auth";
export { default as userStore } from "./user";
export { avatarStore } from "./avatarStore";

// Data stores
export { default as casesStore } from "./cases";
export { default as citationsStore } from "./citations";
export { report as reportStore } from "./report";

// AI & Machine Learning stores
export {
  aiStore,
  parseAICommand,
  applyAIClasses,
  aiCommandService,
  recentCommands,
  isAIActive,
} from "./ai-unified";

export { aiHistory as aiHistoryStore } from "./aiHistoryStore";
export { chatStore } from "./chatStore";
export { enhancedRAGStore } from "./enhanced-rag-store";

// Evidence & Document stores
export {
  evidenceStore,
  evidenceById,
  evidenceByCase,
  type Evidence,
} from "./evidence-unified";

// Form handling stores
export { createFormStore as formStore } from "./form";

// Database & Caching stores
export { lokiStore } from "./lokiStore";
export { enhancedLokiStore } from "./enhancedLokiStore";

// XState machines and state management
export { autoTaggingMachine } from "./autoTaggingMachine";
export {
  evidenceProcessingMachine,
  evidenceProcessingStore,
  streamingStore,
} from "./enhancedStateMachines";
export { aiCommandMachine } from "./ai-command-machine";

// New XState + Go microservice integration
export * from "./machines";

// Canvas & Visual stores
export { canvasStore } from "./canvas";

// Utility stores
export { errorHandler } from "./error-handler";
export { default as savedNotesStore } from "./saved-notes";
export * from "./keyboardShortcuts";

// UI Integration utilities
export * from "./melt-ui-integration";

// Demo and testing
export {
  runPhase2Demo,
  phase2HealthCheck,
  demoEvidenceUpload,
  demoEnhancedButton,
} from "./phase2-demo";

// Legacy compatibility aliases
export { aiStore as aiCommands } from "./ai-unified";
export { evidenceStore as evidence } from "./evidence-unified";

// Types and interfaces
export type {
  StoreState,
  StoreAction,
  StoreContext,
  AsyncStoreState
} from "./types";
