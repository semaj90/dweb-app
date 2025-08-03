/// <reference types="@sveltejs/kit" />
/// <reference types="svelte" />
/// <reference types="vite/client" />

import type { User, UserSession } from '$lib/types/user';

declare global {
  namespace App {
    interface Error {
      code?: string;
      id?: string;
      errorId?: string;
    }

    interface Locals {
      user?: User | null;
      sessionId?: string;
      session?: UserSession | null;
    }

    interface PageData {
      user?: App.Locals["user"];
      session?: App.Locals["session"];
    }

    interface Platform {
      env?: {
        REDIS_URL: string;
        RABBITMQ_URL: string;
        NEO4J_URL: string;
        DATABASE_URL: string;
        OLLAMA_URL: string;
        NODE_ENV: string;
      };
    }
  }

  interface Window {
    fs: {
      readFile: (
        path: string,
        options?: { encoding?: string },
      ) => Promise<Uint8Array | string>;
    };
    __MATRIX_UI__: any;
    __TAURI__?: any;
    electronAPI?: any;
  }

  interface HTMLElement {
    inert?: boolean;
  }
}

export {};
