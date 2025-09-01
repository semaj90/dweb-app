/**
 * Standardized API Response Utilities
 * Provides consistent response formats across all API routes
 */

import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  requestId?: string;
  timestamp: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}

/**
 * Create a successful API response
 */
export function apiSuccess<T>(
  data: T,
  message?: string,
  requestId?: string
) {
  const response: ApiResponse<T> = {
    success: true,
    data,
    timestamp: new Date().toISOString()
  };

  if (message) response.message = message;
  if (requestId) response.requestId = requestId;

  return json(response);
}

/**
 * Create an error API response
 */
export function apiError(
  message: string,
  status: number = 400,
  code?: string,
  details?: any,
  requestId?: string
) {
  const response: ApiResponse = {
    success: false,
    error: message,
    timestamp: new Date().toISOString()
  };

  if (requestId) response.requestId = requestId;
  if (code || details) {
    response.data = { code, details };
  }

  return json(response, { status });
}

/**
 * Validate request body against required fields
 */
export function validateRequest(
  body: any,
  requiredFields: string[]
): string | null {
  if (!body || typeof body !== 'object') {
    return 'Invalid request body';
  }

  for (const field of requiredFields) {
    if (!(field in body) || body[field] === null || body[field] === undefined) {
      return `Missing required field: ${field}`;
    }
  }

  return null;
}

/**
 * Get request ID from locals (set by hooks.server.ts)
 */
export function getRequestId(event: RequestEvent): string {
  return (event.locals as any).requestId || `req_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

/**
 * Create standardized cache headers
 */
export function getCacheHeaders(maxAge: number = 0) {
  if (maxAge === 0) {
    return {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0'
    };
  }

  return {
    'Cache-Control': `public, max-age=${maxAge}`
  };
}

/**
 * Standardized API handler wrapper with error catching
 */
export function withErrorHandling<T extends RequestEvent>(
  handler: (event: T) => Promise<Response>
) {
  return async (event: T): Promise<Response> => {
    const requestId = getRequestId(event);
    
    try {
      return await handler(event);
    } catch (error: any) {
      console.error(`API Error [${requestId}]:`, error);
      
      return apiError(
        error.message || 'Internal server error',
        500,
        error.code || 'INTERNAL_ERROR',
        process.env.NODE_ENV === 'development' ? error.stack : undefined,
        requestId
      );
    }
  };
}