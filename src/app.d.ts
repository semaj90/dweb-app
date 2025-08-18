// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
import type { Unit, Session } from '$lib/yorha/db/schema';

declare global {
  namespace App {
    interface Error {
      message: string;
      code?: string;
    }
    
    interface Locals {
      user?: Unit;
      session?: Session;
    }
    
    interface PageData {
      user?: Unit;
    }
    
    interface Platform {}
  }
}

export {};