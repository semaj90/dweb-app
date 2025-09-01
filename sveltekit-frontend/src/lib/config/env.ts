/// <reference types="vite/client" />

// Production environment configuration
export const ENV_CONFIG = {
  OLLAMA_URL: import.meta.env.OLLAMA_URL || 'http://localhost:11434',
  OLLAMA_MODEL: import.meta.env.OLLAMA_MODEL || 'llama2',
  OPENAI_API_KEY: import.meta.env.OPENAI_API_KEY || '',
  DATABASE_URL: import.meta.env.DATABASE_URL || '',
  PUBLIC_APP_URL: import.meta.env.PUBLIC_APP_URL || 'http://localhost:5173'
} as const;

// Client-safe environment access
export const CLIENT_ENV = {
  OLLAMA_URL: typeof window !== 'undefined' 
    ? 'http://localhost:11434' 
    : ENV_CONFIG.OLLAMA_URL,
  APP_URL: ENV_CONFIG.PUBLIC_APP_URL
} as const;

export default ENV_CONFIG;
