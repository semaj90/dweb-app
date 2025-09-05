// @ts-nocheck
// @ts-nocheck
import { redirect } from "@sveltejs/kit";
import { invalidateSession, clearSessionCookie } from "$lib/server/lucia";
import type { Actions, PageServerLoad } from "./$types";

export const load = async ({ cookies, locals }: Parameters<PageServerLoad>[0]) => {
  if (!locals.user) throw redirect(302, "/login");
  const sessionId = cookies.get("session_id");
  if (sessionId) {
    await invalidateSession(sessionId);
    clearSessionCookie(cookies);
  }
  throw redirect(302, "/login");
};

export const actions = {};
;null as any as Actions;