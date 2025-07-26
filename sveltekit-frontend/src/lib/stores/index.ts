// Enhanced Store Barrel Exports for Phase 2 - Unified
// Prosecutor AI - Clean merged stores

// Core UI stores
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

// Unified AI Store (merged ai-commands + ai-command-parser)
export {
  aiStore,
  parseAICommand,
  applyAIClasses,
  aiCommandService,
  recentCommands,
  isAIActive
} from "./ai-unified";

// Unified Evidence Store (merged evidence + evidenceStore)
export {
  evidenceStore,
  evidenceById,
  evidenceByCase,
  type Evidence
} from "./evidence-unified";

// Canvas & Visual stores
export { canvasStore } from "./canvas";

// Form handling stores
export { createFormStore as formStore } from "./form";

// LokiJS database stores
export { lokiStore } from "./lokiStore";
export { enhancedLokiStore } from "./enhancedLokiStore";

// XState machines
export { autoTaggingMachine } from "./autoTaggingMachine";
export { evidenceProcessingMachine, evidenceProcessingStore, streamingStore } from "./enhancedStateMachines";
export { aiCommandMachine } from "./ai-command-machine";

// Error handling
export { errorHandler } from "./error-handler";

// Notes storage
export { default as savedNotesStore } from "./saved-notes";

// Phase 2: Melt UI Integration utilities
export * from "./melt-ui-integration";

// Phase 2: Demo and health check
export {
  runPhase2Demo,
  phase2HealthCheck,
  demoEvidenceUpload,
  demoEnhancedButton
} from "./phase2-demo";

// Legacy compatibility aliases
export { aiStore as aiCommands } from "./ai-unified";
export { evidenceStore as evidence } from "./evidence-unified";
export { chatStore } from "./chatStore";
export { aiHistory as aiHistoryStore } from "./aiHistoryStore";
