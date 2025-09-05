/**
 * AI Chat Types - Type definitions for the AI chat system
 */

export interface ChatMessage {
  id: string;
  sessionId: string;
  session_id?: string; // Alternative naming
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  token_count?: number;
  user_id?: string;
  metadata?: {
    model?: string;
    tokens?: number;
    confidence?: number;
    sources?: string[];
  };
}

export interface ChatSession {
  id: string;
  title: string;
  created: number;
  updated: number;
  messageCount: number;
  status: 'active' | 'archived' | 'error';
  is_active?: boolean;
  start_time?: number;
  last_activity?: number;
  context?: RAGContext;
}

export interface MessageAnalysis {
  intent: string;
  entities: Array<{
    type: string;
    value: string;
    confidence: number;
  }>;
  sentiment: 'positive' | 'negative' | 'neutral';
  complexity: number;
  topics: string[];
  confidence: number;
  som_cluster?: string;
}

export interface RAGContext {
  caseId?: string;
  documents: string[];
  relevantSections: Array<{
    documentId: string;
    section: string;
    relevance: number;
  }>;
  metadata: Record<string, any>;
  recommendations?: unknown[];
  did_you_mean?: string;
}

export interface Recommendation {
  id: string;
  type: 'query' | 'document' | 'action';
  title: string;
  description: string;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface StreamingResponse {
  messageId: string;
  content: string;
  isComplete: boolean;
  timestamp: number;
  metadata?: {
    tokens?: number;
    model?: string;
  };
}

export interface UserActivity {
  userId: string;
  sessionId: string;
  isTyping: boolean;
  lastSeen: number;
  status: 'online' | 'away' | 'offline';
  type?: string;
  timestamp?: number;
}

export interface AttentionData {
  messageId: string;
  attentionWeights: number[];
  focusPoints: Array<{
    text: string;
    weight: number;
    position: [number, number];
  }>;
  focused?: boolean;
  lastActivity?: number;
  interactionCount?: number;
  scrollPosition?: number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface ChatStore {
  messages: ChatMessage[];
  currentSession: ChatSession | null;
  activeSessions: ChatSession[];
  connectionStatus: ConnectionStatus;
  isTyping: boolean;
  userActivity: UserActivity[];
  recommendations: Recommendation[];
}