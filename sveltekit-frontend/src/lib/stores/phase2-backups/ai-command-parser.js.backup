/**
 * ORIGINAL AI-COMMAND-PARSER.JS BACKUP
 * ====================================
 * 
 * PHASE CONTEXT: Phase 2 enhancement for AI command processing
 * CONFLICTS: Advanced parsing vs simple state from ai-commands.js
 * MERGE REASON: Combined with ai-commands.js into ai-unified.ts
 * 
 * UNIQUE FEATURES (preserved in unified version):
 * - parseAICommand() function
 * - applyAIClasses() for DOM manipulation
 * - aiCommandService state machine
 * - Real-time result store
 * 
 * PHASE INTEGRATION:
 * Phase 1: Basic foundation (ai-commands.js)
 * Phase 2: ✅ Enhanced parsing (this file) ➡️ unified
 * Phase 3: 🎯 Will integrate with LLM services
 */

import { writable } from 'svelte/store';

// Command result store
export const aiCommandResult = writable(null);

/**
 * Parse AI commands and return structured results
 * @param {string} command - The AI command to parse
 * @returns {Promise<object>} - Parsed command result
 */
export async function parseAICommand(command) {
  try {
    // Basic command parsing logic
    const words = command.toLowerCase().split(' ');
    const action = words[0];
    const target = words.find(w => ['evidence', 'case', 'document'].includes(w)) || 'general';
    const priority = words.find(w => ['high', 'medium', 'low'].includes(w)) || 'medium';
    const type = words.find(w => ['document', 'image', 'video', 'audio'].includes(w)) || 'document';

    const result = {
      action,
      target,
      priority,
      type,
      timestamp: new Date().toISOString(),
      processed: true
    };

    // Update store
    aiCommandResult.set(result);
    
    return result;
  } catch (error) {
    console.error('AI Command Parse Error:', error);
    throw error;
  }
}

/**
 * Apply AI-controlled classes to elements
 * @param {HTMLElement} element - Target element
 * @param {object} config - Class configuration
 */
export function applyAIClasses(element, config = {}) {
  const { add = [], remove = [], toggle = [] } = config;
  
  if (add.length) element.classList.add(...add);
  if (remove.length) element.classList.remove(...remove);
  if (toggle.length) toggle.forEach(cls => element.classList.toggle(cls));
}

// Simple command service for basic state management
export const aiCommandService = {
  state: writable('idle'),
  context: writable({}),
  
  send: function(event) {
    console.log('Processing event:', event);
    this.state.set('processing');
    
    setTimeout(() => {
      this.state.set('completed');
      this.context.update(ctx => ({ ...ctx, lastCommand: event }));
    }, 1000);
  },
  
  subscribe: function(callback) {
    return this.state.subscribe(callback);
  }
};
