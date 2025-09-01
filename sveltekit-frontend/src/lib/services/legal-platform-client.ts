/**
 * Legal AI Platform Client Service
 * TypeScript client for interacting with the centralized Legal Platform API v2
 * Provides type-safe CRUD operations for cases, evidence, criminals, and documents
 * Includes comprehensive error handling and logging
 */

import { errorHandler, handleApiError, handleNetworkError, handleValidationError, type ErrorContext } from './error-handler.js';
import crypto from "crypto";

// Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  fallback?: boolean;
}

export interface CaseData {
  id?: string;
  title: string;
  description?: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  status?: 'open' | 'investigating' | 'trial' | 'closed' | 'dismissed';
  incidentDate?: string;
  location?: string;
  userId?: string;
}

export interface EvidenceData {
  id?: string;
  caseId: string;
  title: string;
  description?: string;
  evidenceType: 'document' | 'photo' | 'video' | 'audio' | 'physical' | 'digital';
  fileUrl?: string;
  fileName?: string;
  fileSize?: number;
  mimeType?: string;
  tags?: string[];
  userId?: string;
}

export interface CriminalData {
  id?: string;
  firstName: string;
  lastName: string;
  aliases?: string[];
  dateOfBirth?: string;
  gender?: string;
  height?: string;
  weight?: string;
  eyeColor?: string;
  hairColor?: string;
  userId?: string;
}

export interface DocumentData {
  id?: string;
  caseId?: string;
  userId?: string;
  title: string;
  content: string;
  documentType?: 'brief' | 'contract' | 'evidence' | 'citation';
  status?: 'draft' | 'review' | 'published' | 'archived';
}

export interface SearchQuery {
  query: string;
  type?: 'semantic' | 'traditional' | 'hybrid';
  limit?: number;
  filters?: Record<string, any>;
}

export interface AIRequest {
  operation: 'chat' | 'analyze' | 'summarize' | 'train_som' | 'xstate_event';
  data: any;
}

export interface UploadData {
  files: File[];
  caseId?: string;
  userId?: string;
  metadata?: Record<string, any>;
}

class LegalPlatformClient {
  private baseUrl = '/api/v2/legal-platform';
  
  // Generic API call method with comprehensive error handling
  private async apiCall<T>(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'POST',
    data?: any,
    context: Partial<ErrorContext> = {}
  ): Promise<ApiResponse<T>> {
    const fullUrl = method === 'GET' && data 
      ? `${this.baseUrl}${endpoint}?${new URLSearchParams(data).toString()}`
      : `${this.baseUrl}${endpoint}`;

    try {
      // Log API request
      await errorHandler.logInfo(`API ${method} request to ${endpoint}`, {
        endpoint: fullUrl,
        method,
        hasData: !!data,
        ...context
      });

      const response = await fetch(fullUrl, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'X-Request-ID': crypto.randomUUID?.() || Date.now().toString(),
        },
        body: method !== 'GET' ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        const error = new Error(errorMessage);
        
        // Handle different types of HTTP errors
        if (response.status >= 500) {
          await handleApiError(error, fullUrl, {
            ...context,
            action: `${method} ${endpoint}`,
            httpStatus: response.status
          });
        } else if (response.status === 401 || response.status === 403) {
          await errorHandler.handleAuthError(errorMessage, {
            ...context,
            endpoint: fullUrl,
            httpStatus: response.status
          });
        } else if (response.status >= 400) {
          await handleValidationError(errorMessage, {
            ...context,
            endpoint: fullUrl,
            httpStatus: response.status
          });
        }

        return {
          success: false,
          error: errorMessage
        };
      }

      const result = await response.json();
      
      // Log successful response
      await errorHandler.logDebug(`API ${method} success for ${endpoint}`, {
        endpoint: fullUrl,
        hasResult: !!result,
        ...context
      });

      return result;

    } catch (error: any) {
      // Handle network errors, parsing errors, etc.
      if (error instanceof Error) {
        if (error.message.includes('fetch')) {
          await handleNetworkError(error, {
            ...context,
            endpoint: fullUrl,
            action: `${method} ${endpoint}`
          });
        } else {
          await handleApiError(error, fullUrl, {
            ...context,
            action: `${method} ${endpoint}`
          });
        }

        return {
          success: false,
          error: error.message
        };
      }

      // Unknown error type
      const unknownError = 'Unknown error occurred during API call';
      await errorHandler.logWarn(unknownError, {
        ...context,
        endpoint: fullUrl,
        error
      });

      return {
        success: false,
        error: unknownError
      };
    }
  }

  // Case Management Methods
  async createCase(caseData: CaseData): Promise<ApiResponse<CaseData>> {
    return this.apiCall('', 'POST', {
      action: 'create',
      entity: 'case',
      data: caseData
    });
  }

  async getCase(id: string): Promise<ApiResponse<CaseData>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'case',
      id
    });
  }

  async getAllCases(): Promise<ApiResponse<CaseData[]>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'case'
    });
  }

  async updateCase(id: string, updates: Partial<CaseData>): Promise<ApiResponse<CaseData>> {
    return this.apiCall('', 'POST', {
      action: 'update',
      entity: 'case',
      id,
      data: updates
    });
  }

  async deleteCase(id: string): Promise<ApiResponse<void>> {
    return this.apiCall('', 'POST', {
      action: 'delete',
      entity: 'case',
      id
    });
  }

  async searchCases(query: string): Promise<ApiResponse<CaseData[]>> {
    return this.apiCall('', 'POST', {
      action: 'search',
      entity: 'case',
      data: { query }
    });
  }

  // Evidence Management Methods
  async createEvidence(evidenceData: EvidenceData): Promise<ApiResponse<EvidenceData>> {
    return this.apiCall('', 'POST', {
      action: 'create',
      entity: 'evidence',
      data: evidenceData
    });
  }

  async getEvidence(id: string): Promise<ApiResponse<EvidenceData>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'evidence',
      id
    });
  }

  async getEvidenceByCase(caseId: string): Promise<ApiResponse<EvidenceData[]>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'evidence',
      filters: { caseId }
    });
  }

  async analyzeEvidence(id: string, analysisData?: any): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'analyze',
      entity: 'evidence',
      id,
      data: analysisData
    });
  }

  // Criminal Records Methods
  async createCriminal(criminalData: CriminalData): Promise<ApiResponse<CriminalData>> {
    return this.apiCall('', 'POST', {
      action: 'create',
      entity: 'criminal',
      data: criminalData
    });
  }

  async getCriminal(id: string): Promise<ApiResponse<CriminalData>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'criminal',
      id
    });
  }

  async getAllCriminals(): Promise<ApiResponse<CriminalData[]>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'criminal'
    });
  }

  // Document Management Methods
  async createDocument(documentData: DocumentData): Promise<ApiResponse<DocumentData>> {
    return this.apiCall('', 'POST', {
      action: 'create',
      entity: 'document',
      data: documentData
    });
  }

  async getDocument(id: string): Promise<ApiResponse<DocumentData>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'document',
      id
    });
  }

  async getDocumentsByCase(caseId: string): Promise<ApiResponse<DocumentData[]>> {
    return this.apiCall('', 'POST', {
      action: 'read',
      entity: 'document',
      filters: { caseId }
    });
  }

  // Search Operations
  async semanticSearch(searchQuery: SearchQuery): Promise<ApiResponse<any[]>> {
    return this.apiCall('', 'POST', {
      action: 'search',
      entity: 'search',
      data: searchQuery
    });
  }

  async vectorSearch(query: string, type: string = 'semantic'): Promise<ApiResponse<any[]>> {
    return this.semanticSearch({ query, type });
  }

  // Upload Operations
  async uploadFiles(uploadData: UploadData): Promise<ApiResponse<any>> {
    // Convert to FormData for file upload
    const formData = new FormData();
    
    uploadData.files.forEach(file => {
      formData.append('files', file);
    });
    
    if (uploadData.caseId) {
      formData.append('case_id', uploadData.caseId);
    }
    
    if (uploadData.userId) {
      formData.append('user_id', uploadData.userId);
    }

    try {
      const response = await fetch(`${this.baseUrl}`, {
        method: 'POST',
        body: JSON.stringify({
          action: 'process',
          entity: 'upload',
          data: {
            ...uploadData,
            files: uploadData.files.map(f => ({ name: f.name, size: f.size, type: f.type }))
          }
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      });

      return await response.json();
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      };
    }
  }

  // AI Operations
  async chatWithAI(message: string, context?: any): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'process',
      entity: 'ai',
      data: {
        operation: 'chat',
        data: { message, context }
      }
    });
  }

  async analyzeWithAI(content: string, analysisType?: string): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'process',
      entity: 'ai',
      data: {
        operation: 'analyze',
        data: { content, analysisType }
      }
    });
  }

  async summarizeWithAI(content: string, options?: any): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'process',
      entity: 'ai',
      data: {
        operation: 'summarize',
        data: { content, ...options }
      }
    });
  }

  async trainSOM(inputVectors: number[][], options?: any): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'process',
      entity: 'ai',
      data: {
        operation: 'train_som',
        data: { input_vectors: inputVectors, ...options }
      }
    });
  }

  async sendXStateEvent(machineId: string, eventType: string, eventData?: any): Promise<ApiResponse<any>> {
    return this.apiCall('', 'POST', {
      action: 'process',
      entity: 'ai',
      data: {
        operation: 'xstate_event',
        data: {
          machine_id: machineId,
          type: eventType,
          data: eventData
        }
      }
    });
  }

  // System Health Check
  async healthCheck(): Promise<ApiResponse<any>> {
    try {
      const response = await fetch(`${this.baseUrl}`, {
        method: 'OPTIONS'
      });

      return await response.json();
    } catch (error: any) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Health check failed'
      };
    }
  }

  // Utility Methods
  formatError(apiResponse: ApiResponse<any>): string {
    if (apiResponse.success) return '';
    return apiResponse.error || apiResponse.message || 'Unknown error occurred';
  }

  isSuccess<T>(apiResponse: ApiResponse<T>): apiResponse is ApiResponse<T> & { success: true; data: T } {
    return apiResponse.success && apiResponse.data !== undefined;
  }
}

// Export singleton instance
export const legalPlatformClient = new LegalPlatformClient();
;
// Export types for use in components
export type {
  ApiResponse,
  CaseData,
  EvidenceData,
  CriminalData,
  DocumentData,
  SearchQuery,
  AIRequest,
  UploadData
};

// Export the class for advanced usage
export { LegalPlatformClient };