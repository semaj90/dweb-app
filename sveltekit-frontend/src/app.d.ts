/// <reference types="@sveltejs/kit" />
/// <reference types="svelte" />
/// <reference types="vite/client" />

declare global {
  namespace App {
    interface Error {
      code?: string;
      id?: string;
    }
    
    interface Locals {
      user?: {
        id: string;
        email: string;
        role: string;
      };
      session?: {
        id: string;
        expiresAt: Date;
      };
    }
    
    interface PageData {
      user?: App.Locals['user'];
    }
    
    interface Platform {
      env?: {
        REDIS_URL: string;
        RABBITMQ_URL: string;
        NEO4J_URL: string;
        DATABASE_URL: string;
      };
    }
  }
  
  interface Window {
    fs: {
      readFile: (path: string, options?: { encoding?: string }) => Promise<Uint8Array | string>;
    };
    __MATRIX_UI__: any;
  }
}

export {};
