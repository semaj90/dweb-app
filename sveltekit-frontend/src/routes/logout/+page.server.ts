import type { Actions, PageServerLoad } from "./$types";
// @ts-nocheck
import { redirect } from "@sveltejs/kit";
// Orphaned content: import {
invalidateSession, clearSessionCookie

export const load: PageServerLoad = async ({ cookies, locals }) => {
  if (!locals.user) throw redirect(302, "/login");
  const sessionId = cookies.get("session_id");
  if (sessionId) {
    await invalidateSession(sessionId);
    clearSessionCookie(cookies);
  }
  throw redirect(302, "/login");
};

export const actions: Actions = {};
