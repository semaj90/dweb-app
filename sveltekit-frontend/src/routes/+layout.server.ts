// Temporary simplified layout server to resolve startup issues
// TODO: Re-enable enhanced load after server starts successfully

// import { createEnhancedLayoutLoad } from '$lib/server/ssr/enhanced-load';

// Minimal layout load for testing
export const load = async () => {
  return {
    props: {}
  };
};

// Enhanced SSR layout load with caching and performance optimization
// export const load = createEnhancedLayoutLoad();