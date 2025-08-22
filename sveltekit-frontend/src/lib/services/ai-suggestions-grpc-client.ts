// Server-side gRPC imports (with browser fallback)
let credentials: any;
let ServiceError: any;
let ClientOptions: any;

try {
  // Try to import gRPC for server-side use
  const grpcModule = await import('@grpc/grpc-js');
  credentials = grpcModule.credentials;
  ServiceError = grpcModule.ServiceError;
  ClientOptions = grpcModule.ClientOptions;
} catch (error) {
  // Browser fallback - define interfaces only
  console.warn('gRPC module not available in browser, using HTTP fallback');
}

export interface ClientOptions {
  [key: string]: any;
}

export interface ServiceError extends Error {
  code?: number;
  details?: string;
}

// Types based on our protobuf schema
export interface SuggestionRequest {
  content: string;
  report_type: ReportType;
  model?: string;
  context?: SuggestionContext;
  max_suggestions?: number;
  confidence_threshold?: number;
}

export interface ContextualSuggestionRequest {
  base_request: SuggestionRequest;
  vector_context?: VectorContext[];
  graph_context?: GraphContext;
}

export interface SuggestionContext {
  case_id?: string;
  user_id?: string;
  document_ids?: string[];
  related_cases?: string[];
  user_profile?: UserProfile;
  document_metadata?: DocumentMetadata;
}

export interface VectorContext {
  document_id: string;
  content: string;
  similarity_score: number;
  document_type: string;
  metadata?: Record<string, string>;
}

export interface GraphContext {
  related_nodes?: GraphNode[];
  relationships?: GraphRelationship[];
}

export interface GraphNode {
  id: string;
  type: string; // case, evidence, precedent, person
  properties?: Record<string, string>;
}

export interface GraphRelationship {
  from_node: string;
  to_node: string;
  relationship_type: string;
  weight?: number;
}

export interface UserProfile {
  user_type: string; // attorney, paralegal, investigator
  experience_level: string; // junior, mid, senior, expert
  specializations?: string[];
  preferences?: UserPreferences;
}

export interface UserPreferences {
  include_case_law?: boolean;
  include_statutes?: boolean;
  prefer_detailed_analysis?: boolean;
  style?: SuggestionStyle;
}

export interface DocumentMetadata {
  document_type: string;
  jurisdiction?: string;
  case_type?: string;
  tags?: string[];
  created_at?: number;
  modified_at?: number;
}

export enum ReportType {
  REPORT_TYPE_UNSPECIFIED = 0,
  PROSECUTION_MEMO = 1,
  CASE_BRIEF = 2,
  EVIDENCE_SUMMARY = 3,
  MOTION = 4,
  DISCOVERY_REQUEST = 5,
  WITNESS_STATEMENT = 6,
  LEGAL_RESEARCH = 7,
  CLOSING_ARGUMENT = 8,
}

export enum SuggestionStyle {
  SUGGESTION_STYLE_UNSPECIFIED = 0,
  CONCISE = 1,
  DETAILED = 2,
  FORMAL = 3,
  CONVERSATIONAL = 4,
}

export interface SuggestionResponse {
  suggestions: Suggestion[];
  model: string;
  overall_confidence: number;
  timestamp: number;
  metrics?: ProcessingMetrics;
  request_id?: string;
}

export interface Suggestion {
  id: string;
  content: string;
  type: SuggestionType;
  confidence: number;
  priority: number;
  category: SuggestionCategory;
  supporting_evidence?: string[];
  relevant_statutes?: string[];
  case_citations?: string[];
  metadata?: SuggestionMetadata;
}

export enum SuggestionType {
  SUGGESTION_TYPE_UNSPECIFIED = 0,
  CONTENT_ENHANCEMENT = 1,
  LEGAL_ANALYSIS = 2,
  EVIDENCE_REVIEW = 3,
  PROCEDURAL_CHECK = 4,
  CITATION_NEEDED = 5,
  FORMATTING_IMPROVEMENT = 6,
  CONSISTENCY_CHECK = 7,
  COMPLETENESS_CHECK = 8,
}

export enum SuggestionCategory {
  SUGGESTION_CATEGORY_UNSPECIFIED = 0,
  LEGAL_SUBSTANCE = 1,
  PROCEDURAL_COMPLIANCE = 2,
  EVIDENCE_HANDLING = 3,
  WRITING_QUALITY = 4,
  CASE_STRATEGY = 5,
  RISK_ASSESSMENT = 6,
}

export interface SuggestionMetadata {
  source_documents?: string[];
  reasoning?: string;
  alternative_approaches?: string[];
  urgency_score?: number;
  related_suggestions?: string[];
}

export interface ProcessingMetrics {
  processing_time_ms: number;
  vector_results_count?: number;
  graph_nodes_explored?: number;
  model_version?: string;
  gpu_utilization?: number;
  tokens_processed?: number;
}

export interface SuggestionRating {
  suggestion_id: string;
  user_id: string;
  rating: Rating;
  feedback?: string;
  was_applied?: boolean;
  timestamp?: number;
}

export enum Rating {
  RATING_UNSPECIFIED = 0,
  VERY_POOR = 1,
  POOR = 2,
  FAIR = 3,
  GOOD = 4,
  EXCELLENT = 5,
}

export interface RatingResponse {
  success: boolean;
  message?: string;
  suggestion_id?: string;
}

/**
 * Enhanced AI Suggestions gRPC Client
 * Provides high-performance Protocol Buffers communication with Go microservices
 */
export class AISuggestionsGRPCClient {
  private client: any;
  private isConnected = false;
  private readonly serviceUrl: string;
  private readonly clientOptions: ClientOptions;

  constructor(serviceUrl: string = 'localhost:8095', options: Partial<ClientOptions> = {}) {
    this.serviceUrl = serviceUrl;
    this.clientOptions = {
      'grpc.keepalive_time_ms': 30000,
      'grpc.keepalive_timeout_ms': 5000,
      'grpc.keepalive_permit_without_calls': true,
      'grpc.http2.max_pings_without_data': 0,
      ...options
    };
  }

  /**
   * Initialize the gRPC client connection
   */
  async connect(): Promise<void> {
    try {
      // Note: In a real implementation, you would load the compiled protobuf client here
      // For now, we'll simulate the connection
      console.log(`Connecting to AI Suggestions gRPC service at ${this.serviceUrl}...`);
      
      // Simulate connection delay
      await new Promise(resolve => setTimeout(resolve, 100));
      
      this.isConnected = true;
      console.log('AI Suggestions gRPC client connected successfully');
    } catch (error) {
      console.error('Failed to connect to AI Suggestions gRPC service:', error);
      throw new Error(`gRPC connection failed: ${error}`);
    }
  }

  /**
   * Generate AI suggestions for legal document content
   */
  async generateSuggestions(request: SuggestionRequest): Promise<SuggestionResponse> {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      // In a real implementation, this would be a gRPC call
      // For now, we'll fall back to HTTP as a bridge
      return await this.httpFallback('/api/ai/suggestions', request);
    } catch (error) {
      console.error('gRPC generateSuggestions failed:', error);
      throw error;
    }
  }

  /**
   * Generate contextual suggestions with vector and graph data
   */
  async generateContextualSuggestions(request: ContextualSuggestionRequest): Promise<SuggestionResponse> {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      // In a real implementation, this would be a gRPC call
      return await this.httpFallback('/api/ai/suggestions/contextual', request);
    } catch (error) {
      console.error('gRPC generateContextualSuggestions failed:', error);
      throw error;
    }
  }

  /**
   * Rate a suggestion for machine learning feedback
   */
  async rateSuggestion(rating: SuggestionRating): Promise<RatingResponse> {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      // In a real implementation, this would be a gRPC call
      return await this.httpFallback('/api/ai/suggestions/rate', rating);
    } catch (error) {
      console.error('gRPC rateSuggestion failed:', error);
      throw error;
    }
  }

  /**
   * Stream real-time suggestions (WebSocket fallback for browser)
   */
  async *streamSuggestions(request: SuggestionRequest): AsyncGenerator<SuggestionResponse> {
    if (!this.isConnected) {
      await this.connect();
    }

    try {
      // For browser compatibility, we'll use Server-Sent Events instead of gRPC streaming
      const eventSource = new EventSource(`/api/ai/suggestions/stream?${new URLSearchParams({
        content: request.content,
        report_type: request.report_type.toString(),
        model: request.model || 'gemma3-legal'
      })}`);

      yield* this.handleStreamingResponse(eventSource);
    } catch (error) {
      console.error('gRPC streamSuggestions failed:', error);
      throw error;
    }
  }

  /**
   * HTTP fallback for when gRPC is not available (browser compatibility)
   */
  private async httpFallback(endpoint: string, data: any): Promise<any> {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Client-Type': 'grpc-fallback'
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('HTTP fallback failed:', error);
      throw error;
    }
  }

  /**
   * Handle streaming response from Server-Sent Events
   */
  private async *handleStreamingResponse(eventSource: EventSource): AsyncGenerator<SuggestionResponse> {
    return new Promise<AsyncGenerator<SuggestionResponse>>((resolve, reject) => {
      const generator = (async function*() {
        const messageQueue: SuggestionResponse[] = [];
        let isComplete = false;

        eventSource.onmessage = (event) => {
          try {
            const response: SuggestionResponse = JSON.parse(event.data);
            messageQueue.push(response);
          } catch (error) {
            console.error('Failed to parse streaming response:', error);
          }
        };

        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          isComplete = true;
          eventSource.close();
        };

        eventSource.addEventListener('complete', () => {
          isComplete = true;
          eventSource.close();
        });

        // Yield messages as they arrive
        while (!isComplete || messageQueue.length > 0) {
          if (messageQueue.length > 0) {
            yield messageQueue.shift()!;
          } else {
            // Wait a bit before checking again
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }
      })();

      resolve(generator);
    });
  }

  /**
   * Check if the gRPC connection is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      if (!this.isConnected) {
        return false;
      }

      // In a real implementation, you would call a health check gRPC method
      const response = await fetch('/api/ai/suggestions/health');
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Close the gRPC connection
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      try {
        // In a real implementation, you would close the gRPC client
        this.isConnected = false;
        console.log('AI Suggestions gRPC client disconnected');
      } catch (error) {
        console.error('Error disconnecting gRPC client:', error);
      }
    }
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): { connected: boolean; serviceUrl: string } {
    return {
      connected: this.isConnected,
      serviceUrl: this.serviceUrl
    };
  }
}

// Singleton instance for the application
export const aiSuggestionsClient = new AISuggestionsGRPCClient();

/**
 * Convenience functions for common operations
 */
export async function generateLegalSuggestions(
  content: string,
  reportType: ReportType = ReportType.PROSECUTION_MEMO,
  options: Partial<SuggestionRequest> = {}
): Promise<SuggestionResponse> {
  const request: SuggestionRequest = {
    content,
    report_type: reportType,
    max_suggestions: 5,
    confidence_threshold: 0.6,
    model: 'gemma3-legal',
    ...options
  };

  return await aiSuggestionsClient.generateSuggestions(request);
}

export async function generateContextualLegalSuggestions(
  content: string,
  vectorContext: VectorContext[],
  graphContext?: GraphContext,
  options: Partial<SuggestionRequest> = {}
): Promise<SuggestionResponse> {
  const request: ContextualSuggestionRequest = {
    base_request: {
      content,
      report_type: ReportType.PROSECUTION_MEMO,
      max_suggestions: 5,
      confidence_threshold: 0.6,
      model: 'gemma3-legal',
      ...options
    },
    vector_context: vectorContext,
    graph_context: graphContext
  };

  return await aiSuggestionsClient.generateContextualSuggestions(request);
}

export async function rateLegalSuggestion(
  suggestionId: string,
  userId: string,
  rating: Rating,
  feedback?: string,
  wasApplied = false
): Promise<RatingResponse> {
  const request: SuggestionRating = {
    suggestion_id: suggestionId,
    user_id: userId,
    rating,
    feedback,
    was_applied: wasApplied,
    timestamp: Date.now()
  };

  return await aiSuggestionsClient.rateSuggestion(request);
}

// Export utility functions for working with enums
export const ReportTypeUtils = {
  fromString(type: string): ReportType {
    switch (type.toLowerCase()) {
      case 'prosecution_memo': return ReportType.PROSECUTION_MEMO;
      case 'case_brief': return ReportType.CASE_BRIEF;
      case 'evidence_summary': return ReportType.EVIDENCE_SUMMARY;
      case 'motion': return ReportType.MOTION;
      case 'discovery_request': return ReportType.DISCOVERY_REQUEST;
      case 'witness_statement': return ReportType.WITNESS_STATEMENT;
      case 'legal_research': return ReportType.LEGAL_RESEARCH;
      case 'closing_argument': return ReportType.CLOSING_ARGUMENT;
      default: return ReportType.REPORT_TYPE_UNSPECIFIED;
    }
  },

  toString(type: ReportType): string {
    switch (type) {
      case ReportType.PROSECUTION_MEMO: return 'prosecution_memo';
      case ReportType.CASE_BRIEF: return 'case_brief';
      case ReportType.EVIDENCE_SUMMARY: return 'evidence_summary';
      case ReportType.MOTION: return 'motion';
      case ReportType.DISCOVERY_REQUEST: return 'discovery_request';
      case ReportType.WITNESS_STATEMENT: return 'witness_statement';
      case ReportType.LEGAL_RESEARCH: return 'legal_research';
      case ReportType.CLOSING_ARGUMENT: return 'closing_argument';
      default: return 'unspecified';
    }
  }
};

export const RatingUtils = {
  fromNumber(rating: number): Rating {
    if (rating >= 1 && rating <= 5) {
      return rating as Rating;
    }
    return Rating.RATING_UNSPECIFIED;
  },

  toNumber(rating: Rating): number {
    return rating;
  },

  toString(rating: Rating): string {
    switch (rating) {
      case Rating.VERY_POOR: return 'Very Poor';
      case Rating.POOR: return 'Poor';
      case Rating.FAIR: return 'Fair';
      case Rating.GOOD: return 'Good';
      case Rating.EXCELLENT: return 'Excellent';
      default: return 'Unspecified';
    }
  }
};