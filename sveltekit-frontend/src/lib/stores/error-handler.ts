import type { User } from "$lib/types/user";

import { writable } from "svelte/store";

export interface ErrorDetails {
  code?: string;
  message: string;
  details?: string;
  timestamp: Date;
  context?: Record<string, any>;
  stack?: string;
  retry?: () => Promise<void>;
}
export interface UserFriendlyError {
  title: string;
  message: string;
  suggestion?: string;
  canRetry?: boolean;
  showDetails?: boolean;
  severity: "info" | "warning" | "error" | "critical";
}
class ErrorHandler {
  private errorStore = writable<UserFriendlyError | null>(null);
  private errorHistory = writable<ErrorDetails[]>([]);

  // Subscribe to errors
  subscribe = this.errorStore.subscribe;
  subscribeHistory = this.errorHistory.subscribe;

  // Clear current error
  clear() {
    this.errorStore.set(null);
  }
  // Handle different types of errors
  handle(
    error: unknown,
    context?: Record<string, any>,
    retryFn?: () => Promise<void>,
  ) {
    const errorDetails = this.parseError(error, context);

    // Add to history
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]); // Keep last 50 errors

    // Create user-friendly error
    const userError = this.createUserFriendlyError(errorDetails, retryFn);
    this.errorStore.set(userError);

    // Log for debugging
    this.logError(errorDetails);

    return userError;
  }
  // Handle API errors specifically
  handleApiError(
    response: Response,
    context?: Record<string, any>,
    retryFn?: () => Promise<void>,
  ) {
    const errorDetails: ErrorDetails = {
      code: `HTTP_${response.status}`,
      message: response.statusText || "API request failed",
      details: `${response.status} ${response.statusText}`,
      timestamp: new Date(),
      context: {
        url: response.url,
        status: response.status,
        ...context,
      },
    };

    const userError = this.createUserFriendlyError(errorDetails, retryFn);
    this.errorStore.set(userError);
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]);

    return userError;
  }
  // Handle network errors
  handleNetworkError(
    error: unknown,
    context?: Record<string, any>,
    retryFn?: () => Promise<void>,
  ) {
    const errorDetails: ErrorDetails = {
      code: "NETWORK_ERROR",
      message: "Network connection failed",
      details: error instanceof Error ? error.message : "Unknown network error",
      timestamp: new Date(),
      context: {
        type: "network",
        ...context,
      },
    };

    const userError: UserFriendlyError = {
      title: "Connection Problem",
      message:
        "Unable to connect to the server. Please check your internet connection.",
      suggestion: "Try refreshing the page or check your network connection.",
      canRetry: !!retryFn,
      severity: "error",
    };

    this.errorStore.set(userError);
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]);

    return userError;
  }
  // Handle validation errors
  handleValidationError(
    errors: Record<string, string[]> | string[],
    context?: Record<string, any>,
  ) {
    const errorDetails: ErrorDetails = {
      code: "VALIDATION_ERROR",
      message: "Validation failed",
      details: Array.isArray(errors)
        ? errors.join(", ")
        : Object.entries(errors)
            .map(([field, msgs]) => `${field}: ${msgs.join(", ")}`)
            .join("; "),
      timestamp: new Date(),
      context: {
        type: "validation",
        errors,
        ...context,
      },
    };

    const userError: UserFriendlyError = {
      title: "Validation Error",
      message: "Please check the form and correct any errors.",
      suggestion: errorDetails.details,
      canRetry: false,
      severity: "warning",
    };

    this.errorStore.set(userError);
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]);

    return userError;
  }
  // Handle authentication errors
  handleAuthError(context?: Record<string, any>) {
    const errorDetails: ErrorDetails = {
      code: "AUTH_ERROR",
      message: "Authentication required",
      details: "User session has expired or is invalid",
      timestamp: new Date(),
      context: {
        type: "authentication",
        ...context,
      },
    };

    const userError: UserFriendlyError = {
      title: "Authentication Required",
      message: "Your session has expired. Please log in again.",
      suggestion: "Click to redirect to login page.",
      canRetry: false,
      severity: "warning",
    };

    this.errorStore.set(userError);
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]);

    return userError;
  }
  // Handle file upload errors
  handleFileUploadError(
    error: unknown,
    filename?: string,
    context?: Record<string, any>,
    retryFn?: () => Promise<void>,
  ) {
    const errorDetails = this.parseError(error, {
      type: "file_upload",
      filename,
      ...context,
    });

    let userError: UserFriendlyError;

    if (errorDetails.message.includes("too large")) {
      userError = {
        title: "File Too Large",
        message: `The file "${filename}" is too large to upload.`,
        suggestion: "Please compress the file or choose a smaller file.",
        canRetry: false,
        severity: "warning",
      };
    } else if (errorDetails.message.includes("not allowed")) {
      userError = {
        title: "File Type Not Allowed",
        message: `The file type of "${filename}" is not supported.`,
        suggestion:
          "Please choose a supported file format (images, documents, videos, etc.).",
        canRetry: false,
        severity: "warning",
      };
    } else {
      userError = {
        title: "Upload Failed",
        message: `Failed to upload "${filename}".`,
        suggestion:
          "Please try again or contact support if the problem persists.",
        canRetry: !!retryFn,
        severity: "error",
      };
    }
    this.errorStore.set(userError);
    this.errorHistory.update((history) => [
      errorDetails,
      ...history.slice(0, 49),
    ]);

    return userError;
  }
  private parseError(
    error: unknown,
    context?: Record<string, any>,
  ): ErrorDetails {
    let message = "An unknown error occurred";
    let details = "";
    let stack = "";

    if (error instanceof Error) {
      message = error.message;
      details = error.toString();
      stack = error.stack || "";
    } else if (typeof error === "string") {
      message = error;
      details = error;
    } else if (error && typeof error === "object") {
      message =
        (error as any).message || (error as any).error || "Object error";
      details = JSON.stringify(error);
    }
    return {
      message,
      details,
      stack,
      timestamp: new Date(),
      context,
    };
  }
  private createUserFriendlyError(
    errorDetails: ErrorDetails,
    retryFn?: () => Promise<void>,
  ): UserFriendlyError {
    const { message, details, context } = errorDetails;

    // Network/connection errors
    if (
      message.includes("fetch") ||
      message.includes("network") ||
      message.includes("connection")
    ) {
      return {
        title: "Connection Problem",
        message: "Unable to connect to the server.",
        suggestion: "Please check your internet connection and try again.",
        canRetry: !!retryFn,
        severity: "error",
      };
    }
    // Authentication errors
    if (
      message.includes("authentication") ||
      message.includes("unauthorized") ||
      message.includes("401")
    ) {
      return {
        title: "Authentication Required",
        message: "You need to log in to continue.",
        suggestion: "Please log in and try again.",
        canRetry: false,
        severity: "warning",
      };
    }
    // Permission errors
    if (
      message.includes("permission") ||
      message.includes("forbidden") ||
      message.includes("403")
    ) {
      return {
        title: "Permission Denied",
        message: "You don't have permission to perform this action.",
        suggestion: "Contact an administrator if you believe this is an error.",
        canRetry: false,
        severity: "warning",
      };
    }
    // Validation errors
    if (
      message.includes("validation") ||
      message.includes("invalid") ||
      message.includes("required")
    ) {
      return {
        title: "Validation Error",
        message: "Some information is missing or incorrect.",
        suggestion: details || "Please check your input and try again.",
        canRetry: false,
        severity: "warning",
      };
    }
    // File upload errors
    if (context?.type === "file_upload") {
      if (message.includes("too large")) {
        return {
          title: "File Too Large",
          message: "The selected file is too large.",
          suggestion: "Please choose a smaller file or compress it.",
          canRetry: false,
          severity: "warning",
        };
      } else if (message.includes("not allowed")) {
        return {
          title: "File Type Not Allowed",
          message: "This file type is not supported.",
          suggestion: "Please choose a supported file format.",
          canRetry: false,
          severity: "warning",
        };
      }
    }
    // Server errors
    if (
      message.includes("500") ||
      message.includes("server error") ||
      message.includes("internal")
    ) {
      return {
        title: "Server Error",
        message: "Something went wrong on our end.",
        suggestion:
          "Please try again in a few moments. If the problem persists, contact support.",
        canRetry: !!retryFn,
        severity: "error",
      };
    }
    // Generic error
    return {
      title: "Something Went Wrong",
      message: message || "An unexpected error occurred.",
      suggestion: "Please try again. If the problem persists, contact support.",
      canRetry: !!retryFn,
      severity: "error",
      showDetails: true,
    };
  }
  private logError(errorDetails: ErrorDetails) {
    console.group("🚨 Error Handler");
    console.error("Message:", errorDetails.message);
    if (errorDetails.code) console.error("Code:", errorDetails.code);
    if (errorDetails.details) console.error("Details:", errorDetails.details);
    if (errorDetails.context) console.error("Context:", errorDetails.context);
    if (errorDetails.stack) console.error("Stack:", errorDetails.stack);
    console.groupEnd();
  }
  // Get error statistics for debugging
  getErrorStats() {
    let history: ErrorDetails[] = [];
    this.errorHistory.subscribe((h) => (history = h))();

    const stats = {
      total: history.length,
      byCode: {} as Record<string, number>,
      byType: {} as Record<string, number>,
      recent: history.slice(0, 10),
      last24Hours: history.filter(
        (e) => Date.now() - e.timestamp.getTime() < 24 * 60 * 60 * 1000,
      ).length,
    };

    history.forEach((error) => {
      if (error.code) {
        stats.byCode[error.code] = (stats.byCode[error.code] || 0) + 1;
      }
      if (error.context?.type) {
        stats.byType[error.context.type] =
          (stats.byType[error.context.type] || 0) + 1;
      }
    });

    return stats;
  }
  // Clear error history
  clearHistory() {
    this.errorHistory.set([]);
  }
}
// Create singleton instance
export const errorHandler = new ErrorHandler();

// Convenience functions
export function handleError(
  error: unknown,
  context?: Record<string, any>,
  retryFn?: () => Promise<void>,
) {
  return errorHandler.handle(error, context, retryFn);
}
export function handleApiError(
  response: Response,
  context?: Record<string, any>,
  retryFn?: () => Promise<void>,
) {
  return errorHandler.handleApiError(response, context, retryFn);
}
export function handleNetworkError(
  error: unknown,
  context?: Record<string, any>,
  retryFn?: () => Promise<void>,
) {
  return errorHandler.handleNetworkError(error, context, retryFn);
}
export function handleValidationError(
  errors: Record<string, string[]> | string[],
  context?: Record<string, any>,
) {
  return errorHandler.handleValidationError(errors, context);
}
export function handleAuthError(context?: Record<string, any>) {
  return errorHandler.handleAuthError(context);
}
export function handleFileUploadError(
  error: unknown,
  filename?: string,
  context?: Record<string, any>,
  retryFn?: () => Promise<void>,
) {
  return errorHandler.handleFileUploadError(error, filename, context, retryFn);
}
export function clearError() {
  errorHandler.clear();
}
// Store exports
export const currentError = errorHandler;
export const errorHistory = errorHandler;
