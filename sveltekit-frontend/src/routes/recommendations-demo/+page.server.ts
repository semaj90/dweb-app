// Server-side data loading for recommendations demo
import type { PageServerLoad } from './$types.js';
import { URL } from "url";

export const load: PageServerLoad = async ({ fetch, url }) => {
  const personId = url.searchParams.get('personId') || 'demo-user-1';
  
  // Pre-load FOAF data for SSR
  let foafData = null;
  let initialSuggestions = [];
  
  try {
    // Load FOAF recommendations
    const foafResponse = await fetch(`/api/foaf/${personId}?limit=3`);
    if (foafResponse.ok) {
      foafData = await foafResponse.json();
    }
    
    // Load initial suggestions
    const suggestResponse = await fetch('/api/suggest?q=legal&limit=6');
    if (suggestResponse.ok) {
      const suggestData = await suggestResponse.json();
      initialSuggestions = suggestData.suggestions || [];
    }
  } catch (error: any) {
    console.error('SSR data loading error:', error);
  }

  return {
    personId,
    foafData,
    initialSuggestions,
    meta: {
      title: 'Legal AI Recommendations Demo',
      description: 'Friend-of-a-friend professional network and intelligent search suggestions'
    }
  };
};