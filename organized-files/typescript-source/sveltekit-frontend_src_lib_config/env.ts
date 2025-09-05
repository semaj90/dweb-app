// @ts-nocheck
// Production environment configuration
export const ENV_CONFIG = {
  OLLAMA_URL: process.env.OLLAMA_URL || 'http://localhost:11434',
  OLLAMA_MODEL: process.env.OLLAMA_MODEL || 'llama2',
  OPENAI_API_KEY: process.env.OPENAI_API_KEY || '',
  DATABASE_URL: process.env.DATABASE_URL || '',
  PUBLIC_APP_URL: process.env.PUBLIC_APP_URL || 'http://localhost:5173'
} as const;

// Client-safe environment access
export const CLIENT_ENV = {
  OLLAMA_URL: typeof window !== 'undefined' 
    ? 'http://localhost:11434' 
    : ENV_CONFIG.OLLAMA_URL,
  APP_URL: ENV_CONFIG.PUBLIC_APP_URL
} as const;

export default ENV_CONFIG;
