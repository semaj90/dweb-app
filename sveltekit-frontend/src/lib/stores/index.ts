// Barrel export file for all stores - SvelteKit v2 best practices
// This allows for clean imports: import { userStore, themeStore } from '$lib/stores'

// Core UI stores
export { default as uiStore } from "./ui";
export { default as modalStore } from "./modal";
export { default as notificationStore } from "./notification";

// Authentication & User stores
export { default as authStore } from "./auth";
export { default as userStore } from "./user";
export { default as avatarStore } from "./avatarStore";

// Data stores
export { default as casesStore } from "./cases";
export { default as evidenceStore } from "./evidence";
export { default as citationsStore } from "./citations";
export { default as reportStore } from "./report";

// AI & Chat stores
export { default as aiStore } from "./ai-store";
export { default as chatStore } from "./chatStore";
export { default as aiHistoryStore } from "./aiHistoryStore";

// Canvas & Visual stores
export { default as canvasStore } from "./canvas";

// Form handling stores
export { default as formStore } from "./form";

// LokiJS in-memory database stores
export { default as lokiStore } from "./lokiStore";
export { default as enhancedLokiStore } from "./enhancedLokiStore";

// XState machines
export { default as autoTaggingMachine } from "./autoTaggingMachine";
export { default as enhancedStateMachines } from "./enhancedStateMachines";

// Error handling
export { default as errorHandler } from "./error-handler";

// Notes storage
export { default as savedNotesStore } from "./saved-notes";
