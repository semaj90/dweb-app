// Simple chat store for Enhanced Legal AI Chat
import { writable } from 'svelte/store';
import type { ChatMessage, ChatSession, ChatState } from '$lib/types/chat';

const initialState: ChatState = {
  messages: [],
  currentSession: null,
  isLoading: false,
  error: null
};

function createChatStore() {
  const { subscribe, set, update } = writable<ChatState>(initialState);

  return {
    subscribe,
    set,
    update,
    
    // Actions
    setMessages: (messages: ChatMessage[]) => {
      update(state => ({ ...state, messages }));
    },
    
    addMessage: (message: ChatMessage) => {
      update(state => ({
        ...state,
        messages: [...state.messages, message]
      }));
    },
    
    setCurrentSession: (session: ChatSession | null) => {
      update(state => ({ ...state, currentSession: session }));
    },
    
    setLoading: (isLoading: boolean) => {
      update(state => ({ ...state, isLoading }));
    },
    
    setError: (error: string | null) => {
      update(state => ({ ...state, error }));
    },
    
    clearMessages: () => {
      update(state => ({ ...state, messages: [] }));
    },
    
    reset: () => {
      set(initialState);
    }
  };
}

export const chatStore = createChatStore();