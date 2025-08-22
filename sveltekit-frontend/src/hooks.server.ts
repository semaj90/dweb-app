import crypto from "crypto";
import { sequence } from "@sveltejs/kit/hooks";
import type { Handle, HandleServerError } from "@sveltejs/kit";

import { lucia } from "$lib/server/auth";
import type { Session } from "lucia"; // for stronger typing of session
import { db } from "$lib/server/db";
import { vectorOps } from "$lib/server/db/enhanced-vector-operations.js";
import { productionServiceClient } from "$lib/services/productionServiceClient";
// Use canonical User type (with createdAt/updatedAt) from types/index
import type { User } from "$lib/types/index";

// Augment SvelteKit's Locals so `event.locals` usages in this file are typed.
declare module "@sveltejs/kit" {
  interface Locals {
    apiContext?: APIContext; // use the strongly typed interface below
    /**
     * Authenticated application user (canonical shape from $lib/types)
     */
    user?: User | null;
    /**
     * Active Lucia session (null if unauthenticated)
     */
    session?: Session | null;
    serviceHealth?: any;
  }
}

// --- TYPE DEFINITIONS ---

/**
 * @interface APIContext
 * @description Defines the shape of the context object available in `event.locals.apiContext`.
 * This provides a strongly-typed context for database access, services, and request-specific metadata
 * to be used within server-side routes and logic.
 */
interface APIContext {
  db: typeof db;
  vectorOps: typeof vectorOps;
  productionServices: typeof productionServiceClient;
  userId?: string;
  userRole?: User['role'];
  requestId: string;
  startTime: number;
  features: {
    enhancedRAG: boolean;
    vectorSearch: boolean;
    multiCoreOllama: boolean;
    nvidiaLLama: boolean;
    neo4jIntegration: boolean;
    realTimeAnalytics: boolean;
  };
}

// --- HOOK IMPLEMENTATIONS ---

/**
 * @function requestLogger
 * @description Logs the start and end of each incoming request, including method, URL, status, and duration.
 * This is the first hook in the sequence to run.
 */
const requestLogger: Handle = async ({ event, resolve }) => {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  console.log(`[${new Date().toISOString()}] --> ${event.request.method} ${event.url.pathname}${event.url.search} | ID: ${requestId}`);

  const response = await resolve(event);

  const duration = Date.now() - startTime;
  console.log(`[${new Date().toISOString()}] <-- ${event.request.method} ${event.url.pathname} ${response.status} (${duration}ms) | ID: ${requestId}`);

  return response;
};

/**
 * @function initializeServices
 * @description Initializes and attaches the core application services and feature flags to `event.locals.apiContext`.
 * This makes services like the database and other clients available to all subsequent hooks and server routes.
 */
const initializeServices: Handle = async ({ event, resolve }) => {
  // We can grab the requestId and startTime from the logger hook if we pass them via event.locals,
  // but for simplicity, we'll keep them scoped here for now. A shared UUID is a good enhancement.
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  event.locals.apiContext = {
    db,
    vectorOps,
    productionServices: productionServiceClient,
    requestId,
    startTime,
    features: {
      enhancedRAG: true,
      vectorSearch: true,
      multiCoreOllama: true,
      nvidiaLLama: true,
      neo4jIntegration: true,
      realTimeAnalytics: true
    }
  } as APIContext;

  // Perform a non-blocking service health check during server-side rendering.
  productionServiceClient.checkAllServicesHealth()
    .then(healthStatus => {
      event.locals.serviceHealth = healthStatus;
    })
    .catch(error => {
      console.warn('SSR Service health check failed:', error);
      event.locals.serviceHealth = {}; // Default to empty object on failure
    });

  return resolve(event);
};

/**
 * @function authenticationHandler
 * @description Handles user authentication using Lucia. It validates the session cookie,
 * refreshes it if necessary, and transforms the Lucia user object into the application-specific User type.
 * It also populates `userId` and `userRole` in the apiContext.
 */
const authenticationHandler: Handle = async ({ event, resolve }) => {
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  if (!sessionId) {
    event.locals.user = null;
    event.locals.session = null;
    return resolve(event);
  }

  try {
    const { session, user: luciaUser } = await lucia.validateSession(sessionId);

    // If session is fresh, create and set a new session cookie
    if (session && session.fresh) {
      const sessionCookie = lucia.createSessionCookie(session.id);
      event.cookies.set(sessionCookie.name, sessionCookie.value, {
        path: ".",
        ...sessionCookie.attributes,
      });
    }

    // If session is invalid, create and set a blank session cookie to remove the old one
    if (!session) {
      const sessionCookie = lucia.createBlankSessionCookie();
      event.cookies.set(sessionCookie.name, sessionCookie.value, {
        path: ".",
        ...sessionCookie.attributes,
      });
    }

    // If a user is validated, transform it to the app's User type
    if (luciaUser) {
      // luciaUser is typed loosely; cast to any for property extraction
      const raw: any = luciaUser as any;
      const nowIso = new Date().toISOString();
      const appUser: User = {
        id: raw.id,
        email: raw.email ?? "",
        name: raw.name || raw.email || raw.id,
        // Ensure required props are non-undefined (interface demands strings)
        firstName: raw.firstName ?? "",
        lastName: raw.lastName ?? "",
        role: (raw.role as User['role']) || "user",
        createdAt: (raw.createdAt || raw.created_at || nowIso),
        updatedAt: (raw.updatedAt || nowIso),
        avatarUrl: raw.avatarUrl,
        isActive: Boolean(raw.isActive ?? true),
        emailVerified: Boolean(raw.emailVerified),
        preferences: raw.preferences
      };
      event.locals.user = appUser;

      if (event.locals.apiContext) {
        (event.locals.apiContext as APIContext).userId = appUser.id;
        (event.locals.apiContext as APIContext).userRole = appUser.role;
      }
    } else {
      event.locals.user = null;
    }

    // Store full session object; downstream legacy code expecting string will need updating
    event.locals.session = (session as any) ?? null;
  } catch (error) {
    console.error("Authentication error in server hook:", error);
    event.locals.user = null;
    event.locals.session = null;
  }

  return resolve(event);
};

/**
 * @function customErrorHandler
 * @description A centralized error handler for server-side errors.
 * It logs the error and returns a standardized error shape to the client.
 */
const customErrorHandler: HandleServerError = ({ error, event }) => {
  const errorId = crypto.randomUUID();
  console.error(`[SERVER ERROR] ID: ${errorId}`);
  console.error(`Error processing ${event.request.method} ${event.url.pathname}:`, error);

  return {
    message: 'An unexpected error occurred on the server.',
    code: (error as any)?.code ?? 'INTERNAL_SERVER_ERROR',
    errorId: errorId
  };
};


// --- EXPORTS ---

/**
 * @description The main `handle` export for SvelteKit hooks.
 * It uses `sequence` to compose the hooks in a specific order:
 * 1. `requestLogger` - Logs the request.
 * 2. `initializeServices` - Sets up DB, services, and feature flags.
 * 3. `authenticationHandler` - Manages user session and identity.
 */
export const handle: Handle = sequence(
  requestLogger,
  initializeServices,
  authenticationHandler
);

/**
 * @description The main `handleError` export for SvelteKit.
 * This function is invoked when an error is thrown during request handling.
 */
export const handleError: HandleServerError = customErrorHandler;
