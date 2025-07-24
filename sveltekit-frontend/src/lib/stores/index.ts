// Enhanced Store Barrel Exports for Phase 2 - Unified
// Prosecutor AI - Clean merged stores

// Core UI stores
export { uiStore } from "./ui";
export { modalStore } from "./modal";
export { notificationStore } from "./notification";

// Authentication & User stores
export { authStore } from "./auth";
export { userStore } from "./user";
export { avatarStore } from "./avatarStore";

// Data stores
export { casesStore } from "./cases";
export { citationsStore } from "./citations";
export { reportStore } from "./report";

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
export { formStore } from "./form";

// LokiJS database stores
export { lokiStore } from "./lokiStore";
export { enhancedLokiStore } from "./enhancedLokiStore";

// XState machines
export { autoTaggingMachine } from "./autoTaggingMachine";
export { enhancedStateMachines } from "./enhancedStateMachines";
export { aiCommandMachine } from "./ai-command-machine";

// Error handling
export { errorHandler } from "./error-handler";

// Notes storage
export { savedNotesStore } from "./saved-notes";

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
export { aiHistoryStore } from "./aiHistoryStore";
