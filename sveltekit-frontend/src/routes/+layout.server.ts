import { URL } from "url";
// @ts-nocheck

// Expose session and user to all layouts/pages (SSR)
export const load = async ({ locals, url, request }) => {
  // Enhanced SSR context with hydration support
  const contextData = {
    user: locals.user,
    // SSR hydration context
    hydrationContext: {
      timestamp: new Date().toISOString(),
      route: url.pathname,
      userAgent: request?.headers.get("user-agent") || null,
      // Golden ratio layout settings for consistent SSR/client rendering
      goldenRatio: {
        phi: 1.618,
        containerWidth: 1200,
        mainContentRatio: 0.618,
        sidebarRatio: 0.382,
      },
      // AI system status for client hydration
      aiSystemStatus: {
        localLLMEnabled: true,
        ragEnabled: true,
        vectorSearchEnabled: true,
        streamingEnabled: true,
      },
      // Theme and UI preferences
      uiPreferences: {
        theme: "auto",
        language: "en",
        accessibility: {
          highContrast: false,
          reducedMotion: false,
          screenReader: false,
        },
      },
    },
  };

  return contextData;
};
