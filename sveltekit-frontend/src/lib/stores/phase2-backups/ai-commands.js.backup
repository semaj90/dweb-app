/**
 * ORIGINAL AI-COMMANDS.JS BACKUP
 * =============================
 * 
 * PHASE CONTEXT: This was the Phase 1 foundation store
 * CONFLICTS: Simple store vs enhanced parsing needed for Phase 2
 * MERGE REASON: Combined with ai-command-parser.js for unified functionality
 * 
 * DIFFERENCES FROM UNIFIED VERSION:
 * - JavaScript vs TypeScript
 * - No command parsing logic
 * - No real-time AI class application
 * - Simple state management only
 * 
 * PHASE INTEGRATION:
 * Phase 1: ✅ Basic command history (this file)
 * Phase 2: ➡️ Enhanced with parsing (merged into ai-unified.ts)
 * Phase 3: 🎯 Will add LLM integration
 */

import { writable } from 'svelte/store';

/**
 * AI Commands Store
 * Manages AI command state, history, and processing
 */

// Create the AI commands store
export const aiCommands = writable({
  current: '',
  history: [],
  isProcessing: false,
  lastResult: null,
  error: null
});

// Command history management
export const addCommand = (command, result = null) => {
  aiCommands.update(store => ({
    ...store,
    history: [...store.history, { 
      command, 
      result, 
      timestamp: new Date().toISOString() 
    }],
    lastResult: result
  }));
};

// Set current command
export const setCurrentCommand = (command) => {
  aiCommands.update(store => ({
    ...store,
    current: command
  }));
};

// Set processing state
export const setProcessing = (isProcessing) => {
  aiCommands.update(store => ({
    ...store,
    isProcessing
  }));
};

// Set error state
export const setError = (error) => {
  aiCommands.update(store => ({
    ...store,
    error
  }));
};

// Clear command history
export const clearHistory = () => {
  aiCommands.update(store => ({
    ...store,
    history: [],
    current: '',
    lastResult: null,
    error: null
  }));
};

// Export default
export default aiCommands;
