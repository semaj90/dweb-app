import loki from 'lokijs';
import type { RequestEvent } from '@sveltejs/kit';

/**
 * +layout.server.ts
 *
 * Lightweight server-side caching using LokiJS (recommended for dev / single-process).
 * Caches "startupStatus" from $lib/services/multi-library-startup for a short TTL.
 *
 * Replace with a Redis-backed implementation for production (shared cache).
 */


type CacheDoc = {
  key: string;
  value: any;
  expiresAt?: number;
};

// Create a single DB instance for the server process lifetime
const db = new loki('server-cache.db');
const collectionName = 'layoutCache';
let layoutCache = db.getCollection<CacheDoc>(collectionName);
if (!layoutCache) {
  layoutCache = db.addCollection<CacheDoc>(collectionName, {
    unique: ['key'],
    autoupdate: true
  });
}

const getFromCache = (key: string): any | null => {
  const doc = layoutCache.findOne({ key }) as CacheDoc | null;
  if (!doc) return null;
  if (doc.expiresAt && Date.now() > doc.expiresAt) {
    // expired — remove and treat as miss
    layoutCache.remove(doc);
    return null;
  }
  return doc.value;
};

const setCache = (key: string, value: any, ttlSeconds?: number) => {
  const expiresAt = ttlSeconds ? Date.now() + ttlSeconds * 1000 : undefined;
  const existing = layoutCache.findOne({ key }) as CacheDoc | null;
  if (existing) {
    existing.value = value;
    existing.expiresAt = expiresAt;
    layoutCache.update(existing);
  } else {
    layoutCache.insert({ key, value, expiresAt });
  }
};

// TTL for startup status cache (adjust for your needs)
const STARTUP_TTL_SECONDS = 60 * 5; // 5 minutes

export const load = async (event?: RequestEvent): Promise<any> => {
  const cacheKey = 'layout:startupStatus';

  // 1. Try to return cached startup status
  const cached = getFromCache(cacheKey);
  if (cached) {
    return {
      startupStatus: cached,
      _cacheHit: true
    };
  }

  // 2. Cache miss — attempt to initialize the multi-library startup service
  try {
    // dynamic import keeps this file usable if the module is temporarily problematic
    const mod = await import('$lib/services/multi-library-startup');
    const multiLibraryStartup = mod?.multiLibraryStartup;

    if (multiLibraryStartup?.initialize && typeof multiLibraryStartup.initialize === 'function') {
      const startupStatus = await multiLibraryStartup.initialize();

      // store the result in the in-process cache
      setCache(cacheKey, startupStatus, STARTUP_TTL_SECONDS);

      return {
        startupStatus,
        _cacheHit: false
      };
    } else {
      // If the module doesn't export expected API, return a safe fallback
      return {
        startupStatus: null,
        error: 'multiLibraryStartup not available'
      };
    }
  } catch (err) {
    // Graceful fallback: log and return null startupStatus
    console.error('Error initializing multi-library-startup in layout server load:', err);
    return {
      startupStatus: null,
      error: 'Failed to initialize startup services'
    };
  }
};