
// Enhanced Bits UI v2 + Svelte 5 + UnoCSS Integration
// Optimized for Legal AI with NieR theming

export { default as Button } from "./Button.svelte";
export { default as Dialog } from "./Dialog.svelte";
export { default as Select } from "./Select.svelte";
export { default as Input } from "./Input.svelte";
export { default as Card } from "./Card.svelte";
export { default as CardHeader } from "./CardHeader.svelte";
export { default as CardTitle } from "./CardTitle.svelte";
export { default as CardDescription } from "./CardDescription.svelte";
export { default as CardContent } from "./CardContent.svelte";
export { default as CardFooter } from "./CardFooter.svelte";

// Demo components
export { default as EnhancedBitsDemo } from "./EnhancedBitsDemo.svelte";
export { default as VectorIntelligenceDemo } from "./VectorIntelligenceDemo.svelte";

// Types and utilities
export type * from "./types";
export * from "../enhanced/button-variants";
export { cn, legalCn, confidenceClass, priorityClass } from "$lib/utils/cn";

// Legal AI specific types
export interface EvidenceItem {
  id: string;
  title: string;
  type: "document" | "image" | "video" | "audio" | "transcript";
  priority: "critical" | "high" | "medium" | "low";
  confidence: number;
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

export interface CaseData {
  id: string;
  title: string;
  type: string;
  status: "active" | "closed" | "pending";
  evidence: EvidenceItem[];
  priority: "critical" | "high" | "medium" | "low";
  assignedTo?: string;
}

export interface AIAnalysis {
  confidence: number;
  entities: Array<{
    text: string;
    type: string;
    confidence: number;
  }>;
  themes: Array<{
    topic: string;
    weight: number;
  }>;
  summary: string;
}

// Missing interfaces for compatibility
export interface SelectOption {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
  category?: string;
}

export interface VectorSearchResult {
  id: string;
  score: number;
  content: string;
  metadata?: Record<string, any>;
  embedding?: number[];
  highlights?: string[];
}

export interface SemanticEntity {
  id: string;
  type: string;
  text?: string;
  properties: Record<string, any>;
  relationships?: Array<{
    target: string;
    type: string;
    properties?: Record<string, any>;
  }>;
}

// Legal domain constants
export const EVIDENCE_TYPES = [
  "document",
  "image",
  "video",
  "audio",
  "transcript",
] as const;

export const PRIORITY_LEVELS = ["critical", "high", "medium", "low"] as const;

export const CASE_TYPES = [
  "criminal",
  "civil",
  "corporate",
  "employment",
  "intellectual_property",
] as const;
