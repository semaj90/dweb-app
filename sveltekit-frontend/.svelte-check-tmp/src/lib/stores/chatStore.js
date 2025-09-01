// Minimal chatStore stub to keep build working while full implementation is fixed.
// Exports a simple actor-like API surface used elsewhere in the app.

import { writable } from 'svelte/store';

const chatActor = {
  start: () => {},
  send: (msg) => { console.debug('chatActor.send', msg); },
};

chatActor.start();

export const useChatActor = () => ({ subscribe: (fn) => writable(null).subscribe(fn), send: chatActor.send });

export const chatActions = {
  /**
   * Send a message to the AI
   * @param {string} message - User message text
   */
  sendMessage: (message) => {
    if (!message?.trim()) return;
    chatActor.send({ type: 'SUBMIT', message });
  },

  /**
   * Reset the chat history
   */
  resetChat: () => chatActor.send({ type: 'RESET' }),

  /**
   * Change chat settings
   * @param {object} newSettings - New settings object
   */
  updateSettings: (settings) => chatActor.send({ type: 'UPDATE_SETTINGS', settings }),
};

// TODO: Add authentication integration
// TODO: Add history persistence to database
// TODO: Add context documents feature
