import { lucia } from "$lib/server/auth";
import type { Handle, HandleFetch, HandleServerError } from "@sveltejs/kit";
import { sequence } from '@sveltejs/kit/hooks';
import { db } from '$lib/server/db/index.js';
import { vectorOps } from '$lib/server/db/enhanced-vector-operations.js';
import { productionServiceClient } from '$lib/services/productionServiceClient.js';

// Enhanced API context for SSR
interface APIContext {
  db: typeof db;
  vectorOps: typeof vectorOps;
  productionServices: typeof productionServiceClient;
  userId?: string;
  userRole?: string;
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

// Initialize services hook
const initializeServices: Handle = async ({ event, resolve }) => {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  // Add API context to event.locals
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

  // Check service health for SSR
  try {
    const healthStatus = await productionServiceClient.checkAllServicesHealth();
    event.locals.serviceHealth = healthStatus;
  } catch (error) {
    console.warn('Service health check failed during SSR:', error);
    event.locals.serviceHealth = {};
  }

  return resolve(event);
};

// Original Lucia auth hook
const luciaAuthHook: Handle = async ({ event, resolve }) => {
  const sessionId = event.cookies.get(lucia.sessionCookieName);
  if (!sessionId) {
    event.locals.user = null;
    event.locals.session = null;
    return resolve(event);
  }

  const { session, user } = await lucia.validateSession(sessionId);
  if (session && session.fresh) {
    const sessionCookie = lucia.createSessionCookie(session.id);
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });
  }
  if (!session) {
    const sessionCookie = lucia.createBlankSessionCookie();
    event.cookies.set(sessionCookie.name, sessionCookie.value, {
      path: ".",
      ...sessionCookie.attributes,
    });
  }

  // Set user and session in locals  
  event.locals.user = user;
  event.locals.session = session?.id || null;
  
  return resolve(event);
};
