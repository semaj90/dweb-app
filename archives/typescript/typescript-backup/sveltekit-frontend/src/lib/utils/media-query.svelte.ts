// Reactive Media Query Store for Svelte 5
// File: src/lib/utils/media-query.svelte.ts

import { browser } from '$app/environment';

/**
 * Creates a reactive media query store using Svelte 5 runes
 * @param query - CSS media query string
 * @returns Reactive boolean indicating if query matches
 */
export function createMediaQuery(query: string) {
  let matches = $state(false);
  let mediaQuery: MediaQueryList | null = null;

  // Initialize if in browser
  if (browser) {
    mediaQuery = window.matchMedia(query);
    matches = mediaQuery.matches;

    // Update matches when media query changes
    const updateMatches = (e: MediaQueryListEvent) => {
      matches = e.matches;
    };

    mediaQuery.addEventListener('change', updateMatches);

    // Cleanup function
    $effect(() => {
      return () => {
        if (mediaQuery) {
          mediaQuery.removeEventListener('change', updateMatches);
        }
      };
    });
  }

  return {
    get matches() {
      return matches;
    }
  };
}

/**
 * Common breakpoint queries
 */
export const breakpoints = {
  sm: '(min-width: 640px)',
  md: '(min-width: 768px)', 
  lg: '(min-width: 1024px)',
  xl: '(min-width: 1280px)',
  '2xl': '(min-width: 1536px)',
  mobile: '(max-width: 767px)',
  tablet: '(min-width: 768px) and (max-width: 1023px)',
  desktop: '(min-width: 1024px)',
  retina: '(-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)',
  landscape: '(orientation: landscape)',
  portrait: '(orientation: portrait)',
  prefersColorSchemeDark: '(prefers-color-scheme: dark)',
  prefersReducedMotion: '(prefers-reduced-motion: reduce)'
} as const;

/**
 * Predefined media query hooks
 */
export function useMediaQuery(query: string) {
  return createMediaQuery(query);
}

export function useBreakpoint(breakpoint: keyof typeof breakpoints) {
  return createMediaQuery(breakpoints[breakpoint]);
}

// Convenience hooks for common breakpoints
export function useIsMobile() {
  return createMediaQuery(breakpoints.mobile);
}

export function useIsTablet() {
  return createMediaQuery(breakpoints.tablet);
}

export function useIsDesktop() {
  return createMediaQuery(breakpoints.desktop);
}

export function useIsDark() {
  return createMediaQuery(breakpoints.prefersColorSchemeDark);
}

export function usePrefersReducedMotion() {
  return createMediaQuery(breakpoints.prefersReducedMotion);
}