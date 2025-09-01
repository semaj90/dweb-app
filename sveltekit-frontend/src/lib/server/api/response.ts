// Standardized API Response Handlers for SvelteKit 2
// Production-ready response patterns with comprehensive error handling

import { json, type RequestEvent } from '@sveltejs/kit';
import { z } from 'zod';
import type { ApiResponse, ApiError } from '../../types/api.js';
import type { APIResponse as UnifiedAPIResponse, FormSubmissionResult } from '$lib/types';
import path from "path";
import { URL } from "url";

// Standard response interface
export interface StandardApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  meta?: {
    timestamp: string;
    requestId: string;
    processingTime: number;
    version: string;
    pagination?: {
      page: number;
      limit: number;
      total: number;
      hasNext: boolean;
      hasPrev: boolean;
    };
  };
}

// Enhanced error class for API errors
export class ApiErrorClass extends Error {
  public readonly code: string;
  public readonly statusCode: number;
  public readonly details?: Record<string, any>;
  public readonly timestamp: Date;

  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    statusCode: number = 500,
    details?: Record<string, any>
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    this.timestamp = new Date();
  }
}

// Success response builder
export function apiSuccess<T>(
  data: T,
  requestId: string = generateRequestId(),
  processingTime: number = 0,
  pagination?: StandardApiResponse<T>['meta']['pagination']
): Response {
  const response: StandardApiResponse<T> = {
    success: true,
    data,
    meta: {
      timestamp: new Date().toISOString(),
      requestId,
      processingTime,
      version: '2.0',
      ...(pagination && { pagination })
    }
  };

  return json(response);
}

// Error response builder
export function apiError(
  error: ApiErrorClass | Error | string,
  requestId: string = generateRequestId(),
  processingTime: number = 0
): Response {
  let apiErrorData: ApiError;
  let statusCode = 500;

  if (error instanceof ApiErrorClass) {
    apiErrorData = {
      code: error.code,
      message: error.message,
      details: error.details,
      timestamp: error.timestamp
    };
    statusCode = error.statusCode;
  } else if (error instanceof Error) {
    apiErrorData = {
      code: 'INTERNAL_ERROR',
      message: error.message,
      timestamp: new Date()
    };
  } else {
    apiErrorData = {
      code: 'UNKNOWN_ERROR',
      message: typeof error === 'string' ? error : 'Unknown error occurred',
      timestamp: new Date()
    };
  }

  const response: StandardApiResponse<never> = {
    success: false,
    error: apiErrorData,
    meta: {
      timestamp: new Date().toISOString(),
      requestId,
      processingTime,
      version: '2.0'
    }
  };

  return json(response, { status: statusCode });
}

// Validation error response
export function validationError(
  validationResult: z.ZodError,
  requestId: string = generateRequestId(),
  processingTime: number = 0
): Response {
  const details = validationResult.errors.reduce((acc, err) => {
    const path = err.path.join('.');
    acc[path] = err.message;
    return acc;
  }, {} as Record<string, string>);

  const apiErr = new ApiErrorClass(
    'Validation failed',
    'VALIDATION_ERROR',
    400,
    { fields: details }
  );

  return apiError(apiErr, requestId, processingTime);
}

// --- Unified Builders (Lightweight wrappers aligned with new shared types) ---
export function buildSuccessResponse<T>(
  data: T,
  metadata: { processingTimeMs: number; requestId: string }
): UnifiedAPIResponse<T> {
  return {
    success: true,
    data,
    metadata: { ...metadata, timestamp: new Date().toISOString() }
  };
}

export function buildErrorResponse(
  code: string,
  message: string,
  metadata: { processingTimeMs: number; requestId: string }
): UnifiedAPIResponse<never> {
  return {
    success: false,
    error: { code, message },
    metadata: { ...metadata, timestamp: new Date().toISOString() }
  } as UnifiedAPIResponse<never>;
}

export function buildFormSubmissionResult<T>(
  result: Omit<FormSubmissionResult<T>, 'metadata'>,
  metadata: { processingTimeMs: number; requestId: string }
): FormSubmissionResult<T> {
  return {
    ...result,
    metadata: { ...metadata, timestamp: new Date().toISOString() }
  };
}

// Request ID generator
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// API wrapper function for consistent error handling
export async function withApiHandler<T>(
  handler: (event: RequestEvent) => Promise<T>,
  event: RequestEvent
): Promise<Response> {
  const startTime = Date.now();
  const requestId = generateRequestId();

  // Add request ID to locals for tracking
  (event.locals as any).requestId = requestId;

  try {
    const result = await handler(event);
    const processingTime = Date.now() - startTime;

    // If the handler returned a Response, return it as-is
    if (result instanceof Response) {
      return result;
    }

    // Otherwise wrap the result in a standard success response
    return apiSuccess(result, requestId, processingTime);
  } catch (error: any) {
    const processingTime = Date.now() - startTime;

    // Log error for monitoring
    console.error(`API Error [${requestId}]:`, {
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      url: event.url.pathname,
      method: event.request.method,
      processingTime
    });

    return apiError(error as Error, requestId, processingTime);
  }
}

// Common API errors
export const CommonErrors = {
  NotFound: (resource: string) =>
    new ApiErrorClass(`${resource} not found`, 'NOT_FOUND', 404),

  Unauthorized: (message = 'Unauthorized access') =>
    new ApiErrorClass(message, 'UNAUTHORIZED', 401),

  Forbidden: (message = 'Access forbidden') =>
    new ApiErrorClass(message, 'FORBIDDEN', 403),

  BadRequest: (message: string, details?: Record<string, any>) =>
    new ApiErrorClass(message, 'BAD_REQUEST', 400, details),

  InternalError: (message = 'Internal server error') =>
    new ApiErrorClass(message, 'INTERNAL_ERROR', 500),

  ServiceUnavailable: (service: string) =>
    new ApiErrorClass(`${service} is currently unavailable`, 'SERVICE_UNAVAILABLE', 503),

  RateLimited: (message = 'Rate limit exceeded') =>
    new ApiErrorClass(message, 'RATE_LIMITED', 429),

  DatabaseError: (operation: string, details?: any) =>
    new ApiErrorClass(
      `Database operation failed: ${operation}`,
      'DATABASE_ERROR',
      500,
      details
    ),

  ValidationFailed: (field: string, reason: string) =>
    new ApiErrorClass(
      `Validation failed for field '${field}': ${reason}`,
      'VALIDATION_ERROR',
      400,
      { field, reason }
    )
} as const;

// Type-safe request body parser with validation
export async function parseRequestBody<T>(
  request: Request,
  schema: z.ZodSchema<T>
): Promise<T> {
  try {
    const body = await request.json();
    return schema.parse(body);
  } catch (error: any) {
    if (error instanceof z.ZodError) {
      throw error; // Let the caller handle validation errors
    }
    throw new ApiErrorClass('Invalid JSON in request body', 'INVALID_JSON', 400);
  }
}

// Pagination helper
export function createPagination(page: number, limit: number, total: number) {
  const offset = (page - 1) * limit;
  const hasNext = offset + limit < total;
  const hasPrev = page > 1;

  return {
    page,
    limit,
    total,
    hasNext,
    hasPrev,
    offset
  };
}
