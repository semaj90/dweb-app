/**
 * TypeScript CRUD Client for Legal AI Platform
 * Type-safe client for interacting with the CRUD REST API
 */

export interface CrudResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    total?: number;
    page?: number;
    limit?: number;
    processingTime?: number;
    cached?: boolean;
    vectorSearch?: boolean;
    fallback?: boolean;
  };
}

export interface PaginationOptions {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface SearchOptions extends PaginationOptions {
  query: string;
  fields?: string[];
  vector?: boolean;
  similarity_threshold?: number;
}

// Entity type definitions based on schema
export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  name?: string;
  role: string;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface Case {
  id: string;
  title: string;
  description?: string;
  status: 'open' | 'closed' | 'pending' | 'archived';
  priority?: 'low' | 'medium' | 'high' | 'critical';
  caseNumber?: string;
  assignedUserId?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface Evidence {
  id: string;
  title: string;
  description?: string;
  evidenceType: string;
  caseId?: string;
  filePath?: string;
  metadata?: any;
  isVerified: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface LegalDocument {
  id: string;
  title: string;
  documentType: string;
  content?: string;
  citation?: string;
  jurisdiction?: string;
  datePublished?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface Criminal {
  id: string;
  firstName: string;
  lastName: string;
  aliasNames?: string[];
  dateOfBirth?: Date;
  address?: string;
  phoneNumber?: string;
  criminalHistory?: any;
  createdAt: Date;
  updatedAt: Date;
}

export interface PersonOfInterest {
  id: string;
  firstName: string;
  lastName: string;
  alias?: string;
  relationship?: string;
  caseId?: string;
  notes?: string;
  createdAt: Date;
  updatedAt: Date;
}

export type EntityName = 
  | 'users'
  | 'cases' 
  | 'evidence'
  | 'legalDocuments'
  | 'criminals'
  | 'personsOfInterest'
  | 'reports'
  | 'ragMessages'
  | 'ragSessions';

export type EntityData = 
  | User 
  | Case 
  | Evidence 
  | LegalDocument 
  | Criminal 
  | PersonOfInterest;

class CrudApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'CrudApiError';
  }
}

export class CrudClient {
  private baseUrl = '/api/v1/crud';

  private async request<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    params?: URLSearchParams | null,
    body?: any
  ): Promise<CrudResponse<T>> {
    const url = params ? `${this.baseUrl}?${params.toString()}` : this.baseUrl;
    
    const options: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    try {
      const response = await fetch(url, options);
      const data = await response.json();

      if (!response.ok) {
        throw new CrudApiError(
          data.message || 'Request failed',
          response.status,
          data.details
        );
      }

      return data;
    } catch (error) {
      if (error instanceof CrudApiError) {
        throw error;
      }
      throw new CrudApiError(
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
    }
  }

  // Generic CRUD operations
  async create<T extends EntityData>(
    entity: EntityName,
    data: Partial<T>
  ): Promise<CrudResponse<T>> {
    return this.request<T>('POST', null, {
      action: 'create',
      entity,
      data
    });
  }

  async read<T extends EntityData>(
    entity: EntityName,
    id: string
  ): Promise<CrudResponse<T>> {
    const params = new URLSearchParams({
      action: 'read',
      entity,
      id
    });
    return this.request<T>('GET', params);
  }

  async update<T extends EntityData>(
    entity: EntityName,
    id: string,
    data: Partial<T>
  ): Promise<CrudResponse<T>> {
    return this.request<T>('POST', null, {
      action: 'update',
      entity,
      id,
      data
    });
  }

  async delete<T extends EntityData>(
    entity: EntityName,
    id: string
  ): Promise<CrudResponse<T>> {
    return this.request<T>('POST', null, {
      action: 'delete',
      entity,
      id
    });
  }

  async list<T extends EntityData>(
    entity: EntityName,
    options: PaginationOptions = {}
  ): Promise<CrudResponse<T[]>> {
    const params = new URLSearchParams({
      action: 'list',
      entity,
      ...(options.page && { page: options.page.toString() }),
      ...(options.limit && { limit: options.limit.toString() }),
      ...(options.sortBy && { sortBy: options.sortBy }),
      ...(options.sortOrder && { sortOrder: options.sortOrder })
    });
    return this.request<T[]>('GET', params);
  }

  async search<T extends EntityData>(
    entity: EntityName,
    options: SearchOptions
  ): Promise<CrudResponse<T[]>> {
    const params = new URLSearchParams({
      action: options.vector ? 'vector_search' : 'search',
      entity,
      search: options.query,
      ...(options.page && { page: options.page.toString() }),
      ...(options.limit && { limit: options.limit.toString() }),
      ...(options.similarity_threshold && { 
        similarity_threshold: options.similarity_threshold.toString() 
      })
    });
    return this.request<T[]>('GET', params);
  }

  // Entity-specific convenience methods
  async createCase(data: Partial<Case>): Promise<CrudResponse<Case>> {
    return this.create<Case>('cases', data);
  }

  async getCases(options: PaginationOptions = {}): Promise<CrudResponse<Case[]>> {
    return this.list<Case>('cases', options);
  }

  async updateCase(id: string, data: Partial<Case>): Promise<CrudResponse<Case>> {
    return this.update<Case>('cases', id, data);
  }

  async deleteCase(id: string): Promise<CrudResponse<Case>> {
    return this.delete<Case>('cases', id);
  }

  async searchCases(query: string, options: Partial<SearchOptions> = {}): Promise<CrudResponse<Case[]>> {
    return this.search<Case>('cases', { query, ...options });
  }

  async createEvidence(data: Partial<Evidence>): Promise<CrudResponse<Evidence>> {
    return this.create<Evidence>('evidence', data);
  }

  async getEvidence(options: PaginationOptions = {}): Promise<CrudResponse<Evidence[]>> {
    return this.list<Evidence>('evidence', options);
  }

  async updateEvidence(id: string, data: Partial<Evidence>): Promise<CrudResponse<Evidence>> {
    return this.update<Evidence>('evidence', id, data);
  }

  async deleteEvidence(id: string): Promise<CrudResponse<Evidence>> {
    return this.delete<Evidence>('evidence', id);
  }

  async searchEvidence(query: string, options: Partial<SearchOptions> = {}): Promise<CrudResponse<Evidence[]>> {
    return this.search<Evidence>('evidence', { query, ...options });
  }

  async createCriminal(data: Partial<Criminal>): Promise<CrudResponse<Criminal>> {
    return this.create<Criminal>('criminals', data);
  }

  async getCriminals(options: PaginationOptions = {}): Promise<CrudResponse<Criminal[]>> {
    return this.list<Criminal>('criminals', options);
  }

  async updateCriminal(id: string, data: Partial<Criminal>): Promise<CrudResponse<Criminal>> {
    return this.update<Criminal>('criminals', id, data);
  }

  async deleteCriminal(id: string): Promise<CrudResponse<Criminal>> {
    return this.delete<Criminal>('criminals', id);
  }

  async createLegalDocument(data: Partial<LegalDocument>): Promise<CrudResponse<LegalDocument>> {
    return this.create<LegalDocument>('legalDocuments', data);
  }

  async getLegalDocuments(options: PaginationOptions = {}): Promise<CrudResponse<LegalDocument[]>> {
    return this.list<LegalDocument>('legalDocuments', options);
  }

  async searchLegalDocuments(query: string, options: Partial<SearchOptions> = {}): Promise<CrudResponse<LegalDocument[]>> {
    return this.search<LegalDocument>('legalDocuments', { query, ...options });
  }

  async createPersonOfInterest(data: Partial<PersonOfInterest>): Promise<CrudResponse<PersonOfInterest>> {
    return this.create<PersonOfInterest>('personsOfInterest', data);
  }

  async getPersonsOfInterest(options: PaginationOptions = {}): Promise<CrudResponse<PersonOfInterest[]>> {
    return this.list<PersonOfInterest>('personsOfInterest', options);
  }

  // Vector search methods
  async vectorSearchCases(query: string, similarity_threshold = 0.7, limit = 20): Promise<CrudResponse<Case[]>> {
    return this.search<Case>('cases', {
      query,
      vector: true,
      similarity_threshold,
      limit
    });
  }

  async vectorSearchEvidence(query: string, similarity_threshold = 0.7, limit = 20): Promise<CrudResponse<Evidence[]>> {
    return this.search<Evidence>('evidence', {
      query,
      vector: true,
      similarity_threshold,
      limit
    });
  }

  async vectorSearchLegalDocuments(query: string, similarity_threshold = 0.7, limit = 20): Promise<CrudResponse<LegalDocument[]>> {
    return this.search<LegalDocument>('legalDocuments', {
      query,
      vector: true,
      similarity_threshold,
      limit
    });
  }

  // Health check
  async healthCheck(): Promise<CrudResponse<{ status: string; timestamp: string }>> {
    try {
      const params = new URLSearchParams({ action: 'health' });
      return this.request('GET', params);
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Health check failed'
      };
    }
  }
}

// Export singleton instance
export const crudClient = new CrudClient();

// Export error class
export { CrudApiError };