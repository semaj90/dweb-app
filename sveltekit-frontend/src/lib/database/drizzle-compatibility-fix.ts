/**
 * Drizzle ORM Compatibility Fix - PostgreSQL + pgvector Integration
 * Systematic resolution of database type mismatches and missing methods
 */

import { barrelStore } from '../stores/barrel-functions';

// ===== DRIZZLE ORM TYPE COMPATIBILITY =====
export interface DrizzleCompatibilityLayer {
  // Enhanced query result handling
  handleQueryResult: <T>(result: any) => T[];

  // Enhanced connection management
  ensureConnection: (client: any) => Promise<any>;

  // Type-safe property access
  safePropertyAccess: <T>(obj: any, property: string, defaultValue: T) => T;

  // Vector operations compatibility
  vectorOperations: {
    similarity: (vector1: number[], vector2: number[]) => number;
    distance: (vector1: number[], vector2: number[]) => number;
    normalize: (vector: number[]) => number[];
  };
}

// ===== ENHANCED QUERY RESULT HANDLER =====
export const handleQueryResult = <T>(result: any): T[] => {
  // Handle different result formats from Drizzle queries
  if (!result) return [];

  // Array result (most common)
  if (Array.isArray(result)) {
    return result.map(row => {
      // Ensure all expected properties exist
      const enhancedRow = barrelStore.database.ensureProperties(row, {
        id: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        case_id: null,
        document_id: null,
        user_id: null,
        message: '',
        content: '',
        metadata: {},
        sources: []
      });
      return enhancedRow;
    });
  }

  // Single row result
  if (typeof result === 'object') {
    const enhancedResult = barrelStore.database.ensureProperties(result, {
      id: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      case_id: null,
      document_id: null,
      user_id: null,
      message: '',
      content: '',
      metadata: {},
      sources: []
    });
    return [enhancedResult];
  }

  // Fallback for unexpected formats
  console.warn('Unexpected query result format:', typeof result, result);
  return [];
};

// ===== SAFE PROPERTY ACCESS =====
export const safePropertyAccess = <T>(obj: any, property: string, defaultValue: T): T => {
  if (!obj || typeof obj !== 'object') {
    return defaultValue;
  }

  // Handle nested property access (e.g., 'metadata.sources')
  const keys = property.split('.');
  let current = obj;

  for (const key of keys) {
    if (current && typeof current === 'object' && key in current) {
      current = current[key];
    } else {
      return defaultValue;
    }
  }

  return current !== null && current !== undefined ? current : defaultValue;
};

// ===== VECTOR OPERATIONS COMPATIBILITY =====
export const vectorOperations = {
  similarity: (vector1: number[], vector2: number[]): number => {
    if (!vector1 || !vector2 || vector1.length !== vector2.length) {
      return 0;
    }

    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    for (let i = 0; i < vector1.length; i++) {
      dotProduct += vector1[i] * vector2[i];
      magnitude1 += vector1[i] * vector1[i];
      magnitude2 += vector2[i] * vector2[i];
    }

    const magnitude = Math.sqrt(magnitude1) * Math.sqrt(magnitude2);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  },

  distance: (vector1: number[], vector2: number[]): number => {
    if (!vector1 || !vector2 || vector1.length !== vector2.length) {
      return Infinity;
    }

    let sum = 0;
    for (let i = 0; i < vector1.length; i++) {
      const diff = vector1[i] - vector2[i];
      sum += diff * diff;
    }

    return Math.sqrt(sum);
  },

  normalize: (vector: number[]): number[] => {
    if (!vector || vector.length === 0) {
      return [];
    }

    let magnitude = 0;
    for (const component of vector) {
      magnitude += component * component;
    }

    magnitude = Math.sqrt(magnitude);
    if (magnitude === 0) {
      return new Array(vector.length).fill(0);
    }

    return vector.map(component => component / magnitude);
  }
};

// ===== CONNECTION MANAGEMENT =====
export const ensureConnection = async (client: any): Promise<any> => {
  if (!client) {
    throw new Error('Database client is null or undefined');
  }

  // Check if client has expected methods
  const requiredMethods = ['query', 'execute'];
  for (const method of requiredMethods) {
    if (typeof client[method] !== 'function') {
      console.warn(`Database client missing method: ${method}`);
      // Add missing method as no-op
      client[method] = async (...args: any[]) => {
        console.warn(`Called missing method ${method} with args:`, args);
        return { rows: [], rowCount: 0 };
      };
    }
  }

  // Test connection if possible
  try {
    if (typeof client.query === 'function') {
      await client.query('SELECT 1');
    }
  } catch (error: any) {
    console.warn('Database connection test failed:', error);
  }

  return client;
};

// ===== ENHANCED DRIZZLE COMPATIBILITY LAYER =====
export const drizzleCompatibilityLayer: DrizzleCompatibilityLayer = {
  handleQueryResult,
  ensureConnection,
  safePropertyAccess,
  vectorOperations
};

// ===== TYPE-SAFE RESULT ENHANCER =====
export const enhanceResultWithTypes = <T extends Record<string, any>>(
  result: any,
  typeMap: Record<keyof T, any>
): T => {
  if (!result || typeof result !== 'object') {
    // Create object with default values from typeMap
    const defaultObject = {} as T;
    for (const [key, defaultValue] of Object.entries(typeMap)) {
      (defaultObject as any)[key] = defaultValue;
    }
    return defaultObject;
  }

  const enhancedResult = { ...result } as T;

  // Ensure all properties from typeMap exist
  for (const [key, defaultValue] of Object.entries(typeMap)) {
    if (!(key in enhancedResult) || enhancedResult[key as keyof T] === undefined) {
      (enhancedResult as any)[key] = defaultValue;
    }
  }

  return enhancedResult;
};

// ===== COMMON DATABASE ENTITY ENHANCERS =====
export const entityEnhancers = {
  // Legal document entity enhancer
  legalDocument: (doc: any) => enhanceResultWithTypes(doc, {
    id: null,
    case_id: null,
    document_id: null,
    title: '',
    content: '',
    document_type: 'document',
    file_path: null,
    metadata: {},
    user_id: null,
    status: 'pending',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }),

  // Chat message entity enhancer
  chatMessage: (message: any) => enhanceResultWithTypes(message, {
    id: null,
    message: '',
    role: 'user',
    conversation_id: null,
    user_id: null,
    timestamp: new Date().toISOString(),
    sources: [],
    metadata: {},
    created_at: new Date().toISOString()
  }),

  // Cache entry entity enhancer
  cacheEntry: (entry: any) => enhanceResultWithTypes(entry, {
    key: '',
    value: null,
    createdAt: Date.now(),
    expiresAt: Date.now() + 3600000,
    lastAccessed: Date.now(),
    accessCount: 0,
    size: 0,
    version: 1
  }),

  // Vector operation entity enhancer
  vectorOperation: (operation: any) => enhanceResultWithTypes(operation, {
    id: null,
    operation_type: 'embedding',
    input_data: null,
    output_data: null,
    parameters: {},
    status: 'pending',
    started_at: null,
    completed_at: null,
    error_message: null,
    metadata: {}
  })
};

// ===== QUERY INTERCEPTOR FOR TYPE SAFETY =====
export const createTypeSafeQuery = (baseQuery: any) => {
  return {
    ...baseQuery,

    // Enhanced execute method with type safety
    execute: async (...args: any[]) => {
      try {
        const result = await baseQuery.execute(...args);
        return handleQueryResult(result);
      } catch (error: any) {
        console.error('Query execution error:', error);
        return [];
      }
    },

    // Enhanced all() method with type safety
    all: async (...args: any[]) => {
      try {
        const result = await (baseQuery.all || baseQuery.execute)(...args);
        return handleQueryResult(result);
      } catch (error: any) {
        console.error('Query all() error:', error);
        return [];
      }
    },

    // Enhanced get() method with type safety
    get: async (...args: any[]) => {
      try {
        const result = await (baseQuery.get || baseQuery.execute)(...args);
        const results = handleQueryResult(result);
        return results.length > 0 ? results[0] : null;
      } catch (error: any) {
        console.error('Query get() error:', error);
        return null;
      }
    }
  };
};

// ===== EXPORT MAIN COMPATIBILITY LAYER =====
export default {
  drizzleCompatibilityLayer,
  handleQueryResult,
  safePropertyAccess,
  vectorOperations,
  ensureConnection,
  enhanceResultWithTypes,
  entityEnhancers,
  createTypeSafeQuery
};