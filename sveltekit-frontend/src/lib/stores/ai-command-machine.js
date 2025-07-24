/**
 * AI Command Machine for Phase 2
 * XState v5 compatible implementation
 */

import { writable } from 'svelte/store';

// Simple state machine implementation
export const aiCommandMachine = {
  id: 'aiCommand',
  initial: 'idle',
  
  states: {
    idle: {
      on: {
        PROCESS_COMMAND: 'processing'
      }
    },
    processing: {
      on: {
        SUCCESS: 'completed',
        ERROR: 'error'
      }
    },
    completed: {
      on: {
        RESET: 'idle'
      }
    },
    error: {
      on: {
        RETRY: 'processing',
        RESET: 'idle'
      }
    }
  }
};

// Machine service store
export const machineStore = writable({
  state: 'idle',
  context: {}
});
