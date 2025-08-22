
export interface EditableNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  content: string;
  type: "text" | "evidence" | "link";
  metadata?: Record<string, any>;
}

// Renamed to avoid conflict with Evidence from data/types.ts
export interface ComponentEvidence {
  id: string;
  filename: string;
  content: string;
  metadata: Record<string, any>;
  uploadedAt: string;
  userId: string;
}

export interface CanvasState {
  id: string;
  nodes: EditableNode[];
  connections: Array<{
    from: string;
    to: string;
    type: string;
  }>;
  metadata?: Record<string, any>;
}

export interface SearchResult {
  id: string;
  source: string;
  score: number;
  content: string;
  relevanceScore: number;
}

export interface WebSocketMessage {
  type: string;
  payload: any;
  userId: string;
  timestamp: number;
}
