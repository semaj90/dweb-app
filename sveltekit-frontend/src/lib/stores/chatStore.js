import { useActor } from "@xstate/svelte";
// Orphaned content: import {

import { serviceStatus } from "$lib/services/connectivity.js";
import { createActor, , // Chat settings that can be adjusted through UI, const settings = {,   model: "gemma3-legal",,   temperature: 0.3,,   maxTokens: 500, } from

// Create the actor instance (one per application)
const chatActor = createActor(chatMachine, {
  input: {
    settings,
    actorInput: { ping: "pong" }, // Reserved for future use
  },
});

// Start the actor
chatActor.start();

// Global store references
export const useChatActor = () => useActor(chatActor);

// Exposed actions
export const chatActions = {
  /**
   * Send a message to the AI
   * @param {string} message - User message text
   */
  sendMessage: (message) => {
    if (!message?.trim()) return;
    chatActor.send({ type: "SUBMIT", message });
  },

  /**
   * Reset the chat history
   */
  resetChat: () => {
    chatActor.send({ type: "RESET" });
  },

  /**
   * Change chat settings
   * @param {object} newSettings - New settings object
   */
  updateSettings: (newSettings) => {
    chatActor.send({ type: "UPDATE_SETTINGS", settings: newSettings });
  },
};

// TODO: Add authentication integration
// TODO: Add history persistence to database
// TODO: Add context documents feature
