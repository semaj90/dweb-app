// ---
// TYPE GUARD UTILITIES
//
// This file provides type guard functions for safely handling union types and 
// unknown values, addressing TypeScript errors related to type discrimination.
// ---

import type {
  AITask,
  APIResponse,
  WorkerStatus,
  WorkerMessage,
  WorkerMessageType,
  AITaskType,
  ServiceStatus,
  LegalCase,
  Evidence,
  User,
  Notification,
  NotificationType
} from '$lib/types';

// --- Core Type Guards ---

export function isAPIResponse<T>(value: unknown): value is APIResponse<T> {
  return (
    typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    typeof (value as any).success === 'boolean'
  );
}

export function isServiceStatus(value: unknown): value is ServiceStatus {
  return typeof value === 'string' && 
    ['operational', 'degraded', 'offline', 'unknown'].includes(value);
}

// --- AI & Worker Type Guards ---

export function isAITaskType(value: unknown): value is AITaskType {
  return typeof value === 'string' && 
    ['generate', 'analyze', 'embed', 'search', 'embedding', 'analysis', 'classification', 'summarization'].includes(value);
}

export function isWorkerMessageType(value: unknown): value is WorkerMessageType {
  return typeof value === 'string' && 
    ['error', 'status', 'result', 'task', 'TASK_STARTED', 'TASK_COMPLETED', 'TASK_ERROR', 'TASK_CANCELLED', 'STATUS_UPDATE'].includes(value);
}

export function isAITask(value: unknown): value is AITask {
  return (
    typeof value === 'object' &&
    value !== null &&
    'taskId' in value &&
    'type' in value &&
    'providerId' in value &&
    'model' in value &&
    'prompt' in value &&
    'timestamp' in value &&
    'priority' in value &&
    typeof (value as any).taskId === 'string' &&
    isAITaskType((value as any).type) &&
    typeof (value as any).providerId === 'string' &&
    typeof (value as any).model === 'string' &&
    typeof (value as any).prompt === 'string' &&
    typeof (value as any).timestamp === 'number' &&
    ['low', 'medium', 'high'].includes((value as any).priority)
  );
}

export function isWorkerStatus(value: unknown): value is WorkerStatus {
  return (
    typeof value === 'object' &&
    value !== null &&
    'status' in value &&
    'activeRequests' in value &&
    'queueLength' in value &&
    'providers' in value &&
    'maxConcurrent' in value &&
    'uptime' in value &&
    'totalProcessed' in value &&
    'errors' in value &&
    'performance' in value &&
    'lastActivity' in value &&
    ['idle', 'processing', 'error'].includes((value as any).status) &&
    typeof (value as any).activeRequests === 'number' &&
    typeof (value as any).queueLength === 'number' &&
    Array.isArray((value as any).providers)
  );
}

export function isWorkerMessage(value: unknown): value is WorkerMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    isWorkerMessageType((value as any).type)
  );
}

// --- Legal Domain Type Guards ---

export function isLegalCase(value: unknown): value is LegalCase {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'title' in value &&
    'description' in value &&
    'priority' in value &&
    'status' in value &&
    'createdAt' in value &&
    'updatedAt' in value &&
    'userId' in value &&
    'metadata' in value &&
    typeof (value as any).id === 'string' &&
    typeof (value as any).title === 'string' &&
    typeof (value as any).description === 'string' &&
    ['low', 'medium', 'high', 'critical'].includes((value as any).priority) &&
    ['open', 'in_progress', 'closed', 'archived'].includes((value as any).status)
  );
}

export function isEvidence(value: unknown): value is Evidence {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'caseId' in value &&
    'title' in value &&
    'description' in value &&
    'type' in value &&
    'uploadedAt' in value &&
    'metadata' in value &&
    typeof (value as any).id === 'string' &&
    typeof (value as any).caseId === 'string' &&
    typeof (value as any).title === 'string' &&
    typeof (value as any).description === 'string' &&
    ['document', 'image', 'video', 'audio', 'other'].includes((value as any).type)
  );
}

// --- Authentication Type Guards ---

export function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'email' in value &&
    'role' in value &&
    'permissions' in value &&
    'createdAt' in value &&
    typeof (value as any).id === 'string' &&
    typeof (value as any).email === 'string' &&
    ['user', 'admin', 'attorney', 'paralegal'].includes((value as any).role) &&
    Array.isArray((value as any).permissions)
  );
}

// --- UI Type Guards ---

export function isNotificationType(value: unknown): value is NotificationType {
  return typeof value === 'string' && 
    ['info', 'success', 'warning', 'error'].includes(value);
}

export function isNotification(value: unknown): value is Notification {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'type' in value &&
    'title' in value &&
    'message' in value &&
    'timestamp' in value &&
    typeof (value as any).id === 'string' &&
    isNotificationType((value as any).type) &&
    typeof (value as any).title === 'string' &&
    typeof (value as any).message === 'string' &&
    typeof (value as any).timestamp === 'number'
  );
}

// --- Utility Type Guards ---

export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

export function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isArray(value: unknown): value is unknown[] {
  return Array.isArray(value);
}

export function isRecord(value: unknown): value is Record<string, any> {
  return isObject(value);
}

// --- Enhanced Discrimination Helpers ---

export function discriminateWorkerMessage(message: WorkerMessage): {
  isAITask: boolean;
  isWorkerStatus: boolean;
  isAPIResponse: boolean;
  aiTask?: AITask;
  workerStatus?: WorkerStatus;
  apiResponse?: APIResponse<unknown>;
} {
  const result = {
    isAITask: false,
    isWorkerStatus: false,
    isAPIResponse: false,
  } as any;

  if (message.payload) {
    if (isAITask(message.payload)) {
      result.isAITask = true;
      result.aiTask = message.payload;
    } else if (isWorkerStatus(message.payload)) {
      result.isWorkerStatus = true;
      result.workerStatus = message.payload;
    } else if (isAPIResponse(message.payload)) {
      result.isAPIResponse = true;
      result.apiResponse = message.payload;
    }
  }

  return result;
}

// --- Safe Property Access ---

export function safeGet<T>(obj: unknown, path: string, defaultValue: T): T {
  try {
    const keys = path.split('.');
    let current: any = obj;
    
    for (const key of keys) {
      if (current == null || typeof current !== 'object') {
        return defaultValue;
      }
      current = current[key];
    }
    
    return current !== undefined ? current : defaultValue;
  } catch {
    return defaultValue;
  }
}

export function hasProperty<K extends string>(
  obj: unknown,
  prop: K
): obj is Record<K, unknown> {
  return typeof obj === 'object' && obj !== null && prop in obj;
}

// --- Type Assertion Helpers ---

export function assertIsAITask(value: unknown): asserts value is AITask {
  if (!isAITask(value)) {
    throw new Error('Value is not a valid AITask');
  }
}

export function assertIsWorkerStatus(value: unknown): asserts value is WorkerStatus {
  if (!isWorkerStatus(value)) {
    throw new Error('Value is not a valid WorkerStatus');
  }
}

export function assertIsAPIResponse<T>(value: unknown): asserts value is APIResponse<T> {
  if (!isAPIResponse(value)) {
    throw new Error('Value is not a valid APIResponse');
  }
}