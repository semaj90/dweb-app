import type { ServerLoad as PageServerLoad, Actions } from "@sveltejs/kit";
import { redirect } from '@sveltejs/kit';

export const load: PageServerLoad = async ({ locals }) => {
  // Session information for homepage display
  const sessionInfo = {
    userId: locals.session?.user?.id ?? null,
    sessionId: locals.session?.id ?? null,
    email: locals.session?.user?.email ?? null,
    isAuthenticated: !!locals.session?.user
  };

  return {
    ...sessionInfo
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