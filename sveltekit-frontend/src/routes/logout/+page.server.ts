import type { Actions, PageServerLoad } from "./$types";

import { redirect } from "@sveltejs/kit";
import { invalidateSession, deleteSessionTokenCookie } from "$lib/server/session";
export const load: PageServerLoad = async ({ cookies, locals }) => {
  if (!locals.user) throw redirect(302, "/login");
  const sessionId = cookies.get("session_id");
  if (sessionId) {
    await invalidateSession(sessionId);
    deleteSessionTokenCookie({ cookies } as any);
  }
  throw redirect(302, "/login");
};

export const actions: Actions = {};
