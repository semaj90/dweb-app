/**
 * API Client for Legal AI Platform
 * Provides type-safe client-side API access with Lucia v3 authentication
 * Compatible with Superforms and Zod validation
 */

import {
  CreateCaseSchema,
  UpdateCaseSchema,
  CreateEvidenceSchema,
  UpdateEvidenceSchema,
  CreateReportSchema,
  UpdateReportSchema,
  CreatePersonOfInterestSchema,
  UpdatePersonOfInterestSchema,
  type CreateCaseData,
  type UpdateCaseData,
  type CreateEvidenceData,
  type UpdateEvidenceData,
  type CreateReportData,
  type UpdateReportData,
  type CreatePersonOfInterestData,
  type UpdatePersonOfInterestData,
  type PaginationOptions,
  type PaginationResult
} from '$lib/server/services/user-scoped-crud';
import { z } from 'zod';

// API Response Types
interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  pagination?: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
  meta?: {
    userId?: string;
    timestamp: string;
    [key: string]: any;
  };
}

interface APIError {
  success: false;
  message: string;
  code: string;
  details?: any;
}

// === OCR Types ===
export interface OCRResult {
  text: string;
  confidence: number;
  wordCount: number;
  processingTime: number; // ms
  format?: string;
}

export interface OCRBatchItem extends OCRResult {
  fileName: string;
  success: boolean;
  error?: string;
}

export interface OCRBatchResult {
  results: OCRBatchItem[];
  total: number;
  processed: number;
  failed: number;
  processingTime: number;
}

export interface OCRHealthStatus {
  service: 'OCR Service';
  status: 'operational' | 'degraded' | 'offline';
  port: number;
  features: string[];
  performance: {
    avgProcessingTime: number;
    documentsProcessed: number;
    errorRate: number;
  };
}

// API Client Class
export class LegalAIApiClient {
  private baseUrl: string;

  constructor(baseUrl = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  /**
   * Generic API request handler with error handling
   */
  private async apiRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'API request failed');
      }

      return data;
    } catch (error: any) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // ==== CASES API ====

  /**
   * Get all cases for the authenticated user
   */
  async getCases(options: PaginationOptions & {
    status?: 'open' | 'closed' | 'pending' | 'archived';
    priority?: 'low' | 'medium' | 'high' | 'urgent';
  } = {}): Promise<APIResponse<any[]>> {
    const params = new URLSearchParams();

    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });

    return this.apiRequest(`/cases?${params}`);
  }

  /**
   * Get a specific case by ID
   */
  async getCase(caseId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/cases/${caseId}`);
  }

  /**
   * Create a new case
   */
  async createCase(data: CreateCaseData): Promise<APIResponse<any>> {
    // Validate with Zod before sending
    const validatedData = CreateCaseSchema.parse(data);

    return this.apiRequest('/cases', {
      method: 'POST',
      body: JSON.stringify(validatedData)
    });
  }

  /**
   * Update an existing case
   */
  async updateCase(caseId: string, data: Partial<UpdateCaseData>): Promise<APIResponse<any>> {
    // Validate with Zod before sending
    const validatedData = UpdateCaseSchema.parse({ id: caseId, ...data });
    const { id, ...updateData } = validatedData;

    return this.apiRequest(`/cases/${caseId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData)
    });
  }

  /**
   * Delete a case
   */
  async deleteCase(caseId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/cases/${caseId}`, {
      method: 'DELETE'
    });
  }

  // ==== EVIDENCE API ====

  /**
   * Get all evidence for the authenticated user
   */
  async getEvidence(options: PaginationOptions & {
    caseId?: string;
    evidenceType?: string;
    isPublic?: boolean;
  } = {}): Promise<APIResponse<any[]>> {
    const params = new URLSearchParams();

    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });

    return this.apiRequest(`/evidence?${params}`);
  }

  /**
   * Get specific evidence by ID
   */
  async getEvidenceById(evidenceId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/evidence/${evidenceId}`);
  }

  /**
   * Create new evidence
   */
  async createEvidence(data: CreateEvidenceData): Promise<APIResponse<any>> {
    const validatedData = CreateEvidenceSchema.parse(data);

    return this.apiRequest('/evidence', {
      method: 'POST',
      body: JSON.stringify(validatedData)
    });
  }

  /**
   * Update existing evidence
   */
  async updateEvidence(evidenceId: string, data: Partial<UpdateEvidenceData>): Promise<APIResponse<any>> {
    const validatedData = UpdateEvidenceSchema.parse({ id: evidenceId, ...data });
    const { id, ...updateData } = validatedData;

    return this.apiRequest(`/evidence/${evidenceId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData)
    });
  }

  /**
   * Delete evidence
   */
  async deleteEvidence(evidenceId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/evidence/${evidenceId}`, {
      method: 'DELETE'
    });
  }

  // ==== REPORTS API ====

  /**
   * Get all reports for the authenticated user
   */
  async getReports(options: PaginationOptions & {
    caseId?: string;
    status?: 'draft' | 'review' | 'approved' | 'published';
    reportType?: 'analysis' | 'summary' | 'investigation' | 'final';
  } = {}): Promise<APIResponse<any[]>> {
    const params = new URLSearchParams();

    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });

    return this.apiRequest(`/reports?${params}`);
  }

  /**
   * Get specific report by ID
   */
  async getReport(reportId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/reports/${reportId}`);
  }

  /**
   * Create new report
   */
  async createReport(data: CreateReportData): Promise<APIResponse<any>> {
    const validatedData = CreateReportSchema.parse(data);

    return this.apiRequest('/reports', {
      method: 'POST',
      body: JSON.stringify(validatedData)
    });
  }

  /**
   * Update existing report
   */
  async updateReport(reportId: string, data: Partial<UpdateReportData>): Promise<APIResponse<any>> {
    const validatedData = UpdateReportSchema.parse({ id: reportId, ...data });
    const { id, ...updateData } = validatedData;

    return this.apiRequest(`/reports/${reportId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData)
    });
  }

  /**
   * Delete report
   */
  async deleteReport(reportId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/reports/${reportId}`, {
      method: 'DELETE'
    });
  }

  // ==== PERSONS OF INTEREST API ====

  /**
   * Get all persons of interest for the authenticated user
   */
  async getPersonsOfInterest(options: PaginationOptions & {
    riskLevel?: 'low' | 'medium' | 'high' | 'critical';
    status?: 'active' | 'inactive' | 'archived';
    search?: string;
  } = {}): Promise<APIResponse<any[]>> {
    const params = new URLSearchParams();

    Object.entries(options).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });

    return this.apiRequest(`/persons-of-interest?${params}`);
  }

  /**
   * Get specific person of interest by ID
   */
  async getPersonOfInterest(personId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/persons-of-interest/${personId}`);
  }

  /**
   * Create new person of interest
   */
  async createPersonOfInterest(data: CreatePersonOfInterestData): Promise<APIResponse<any>> {
    const validatedData = CreatePersonOfInterestSchema.parse(data);

    return this.apiRequest('/persons-of-interest', {
      method: 'POST',
      body: JSON.stringify(validatedData)
    });
  }

  /**
   * Update existing person of interest
   */
  async updatePersonOfInterest(personId: string, data: Partial<UpdatePersonOfInterestData>): Promise<APIResponse<any>> {
    const validatedData = UpdatePersonOfInterestSchema.parse({ id: personId, ...data });
    const { id, ...updateData } = validatedData;

    return this.apiRequest(`/persons-of-interest/${personId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData)
    });
  }

  /**
   * Delete person of interest
   */
  async deletePersonOfInterest(personId: string): Promise<APIResponse<any>> {
    return this.apiRequest(`/persons-of-interest/${personId}`, {
      method: 'DELETE'
    });
  }

  // ==== UTILITY METHODS ====

  /**
   * Health check for API
   */
  async healthCheck(): Promise<APIResponse<any>> {
    return this.apiRequest('/health');
  }

  /**
   * Get user statistics
   */
  async getUserStats(): Promise<APIResponse<{
    totalCases: number;
    totalEvidence: number;
    totalReports: number;
    totalPersonsOfInterest: number;
    lastActivity: string;
  }>> {
    return this.apiRequest('/stats');
  }

  // ==== OCR SERVICE INTEGRATION ====

  private ocrBase(): string {
    // Allow override via env; fallback to relative API proxy path
    return (globalThis as any).__OCR_BASE__ || '/api/ocr';
  }

  async processDocumentOCR(file: File): Promise<APIResponse<OCRResult>> {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${this.ocrBase()}/extract`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      return data;
    } catch (e) {
      throw e;
    }
  }

  async batchProcessOCR(files: File[]): Promise<APIResponse<OCRBatchResult>> {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    const response = await fetch(`${this.ocrBase()}/batch`, { method: 'POST', body: formData });
    const data = await response.json();
    return data;
  }

  async getOCRStatus(): Promise<APIResponse<OCRHealthStatus>> {
    const response = await fetch(`${this.ocrBase()}/status`);
    return response.json();
  }

  async createEvidenceWithOCR(caseId: string, file: File, metadata: Record<string, any> = {}): Promise<APIResponse<any>> {
    const ocr = await this.processDocumentOCR(file);
    if (!ocr.success) return ocr as any;

    return this.createEvidence({
      caseId,
      evidenceType: 'document',
      title: file.name,
      description: metadata.description || 'OCR processed document',
      contentText: (ocr.data as any)?.text,
      metadata: {
        ...metadata,
        ocr: {
          confidence: (ocr.data as any)?.confidence,
          wordCount: (ocr.data as any)?.wordCount,
          processingTime: (ocr.data as any)?.processingTime
        }
      }
    } as any);
  }
}

// Export singleton instance
export const apiClient = new LegalAIApiClient();

// Export for custom instances
export default LegalAIApiClient;