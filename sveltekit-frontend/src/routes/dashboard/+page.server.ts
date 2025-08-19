// @ts-nocheck
import { cases, criminals } from "$lib/server/db/schema-postgres";
// Orphaned content: import {

import { db } from "$lib/server/db";
// Orphaned content: import type { PageServerLoad

export const load: PageServerLoad = async ({ locals }) => {
  if (!locals.session || !locals.user) {
    throw redirect(302, "/login");
  }
  // Get recent cases
  const recentCases = await db
    .select()
    .from(cases)
    .limit(5)
    .orderBy(cases.createdAt);

  // Get recent criminals/POIs
  const recentCriminals = await db
    .select()
    .from(criminals)
    .limit(6)
    .orderBy(criminals.createdAt);

  return {
    recentCases,
    recentCriminals,
  };
};
