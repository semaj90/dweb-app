
import type { ServerLoad, Actions } from "@sveltejs/kit";
import { redirect } from "@sveltejs/kit";
import { cases, criminals } from '$lib/server/db/schema-postgres';
import { db } from '$lib/server/db/index';

export const load: ServerLoad = async ({ locals }) => {
  // Session information for dashboard display
  const sessionInfo = {
    userId: locals.session?.user?.id ?? null,
    sessionId: locals.session?.id ?? null,
    email: locals.session?.user?.email ?? null,
    isAuthenticated: !!locals.session?.user
  };
  
  // Return mock data for now (keeping existing for backward compatibility)
  const recentCases = [
    { id: 'case-001', title: 'Sample Legal Case', status: 'active', createdAt: new Date() },
    { id: 'case-002', title: 'Evidence Analysis', status: 'pending', createdAt: new Date() }
  ];

  const recentCriminals = [
    { id: 'poi-001', name: 'Sample POI', status: 'active', createdAt: new Date() }
  ];

  return {
    // Session data for display
    ...sessionInfo,
    
    // Existing dashboard data
    recentCases,
    recentCriminals,
  };
};

export const actions: Actions = {
  logout: async ({ cookies }) => {
    // Clear the auth-session cookie
    cookies.delete('auth-session', { path: '/' });

    // Redirect back to homepage after logout
    throw redirect(303, '/');
  }
};
