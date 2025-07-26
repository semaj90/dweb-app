// Enhanced TypeScript declarations for Legal AI system

// Svelte components
declare module "*.svelte" {
  import type { ComponentType, SvelteComponent } from "svelte";
  const component: ComponentType<SvelteComponent>;
  export default component;
}

// AI Services
declare module "$lib/services/ai-service" {
  export class LocalAIService {
    queryWithContext(prompt: string, caseId?: string): Promise<string>;
    embedDocument(content: string, caseId: string, type: string): Promise<void>;
  }
}

declare module "$lib/services/embedding-service" {
  export class EmbeddingService {
    embedText(text: string): Promise<number[]>;
    storeEmbedding(text: string, caseId: string, type: string): Promise<void>;
    similaritySearch(query: string, limit?: number): Promise<any[]>;
  }
}

// Legal AI types
interface LegalCase {
  id: string;
  title: string;
  description?: string;
  status: 'open' | 'closed' | 'pending';
  evidence: Evidence[];
  created_at: Date;
}

interface Evidence {
  id: string;
  case_id: string;
  title: string;
  type: 'document' | 'physical' | 'digital' | 'witness';
  content?: string;
  embedding?: number[];
}

interface AIResponse {
  response: string;
  context: any[];
  confidence: number;
}

declare global {
  namespace App {
    interface Locals {
      user: {
        id: string;
        email: string;
        name: string;
        role: 'prosecutor' | 'admin' | 'analyst';
      } | null;
    }
    
    interface PageData {
      flash?: any;
    }
    
    namespace Superforms {
      type Message = any;
    }
  }
}

export {};
