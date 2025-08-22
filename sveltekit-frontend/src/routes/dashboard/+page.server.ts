
import type { PageServerLoad } from "@sveltejs/kit";
import { redirect } from "@sveltejs/kit";
import { cases, criminals } from "$lib/server/db/schema-postgres";
import { db } from "$lib/server/db";

export const load: PageServerLoad = async ({ locals }) => {
  // For demo purposes, skip authentication check
  // if (!locals.session || !locals.user) {
  //   throw redirect(302, "/login");
  // }
  
  // Return mock data for now
  const recentCases = [
    { id: 'case-001', title: 'Sample Legal Case', status: 'active', createdAt: new Date() },
    { id: 'case-002', title: 'Evidence Analysis', status: 'pending', createdAt: new Date() }
  ];

  const recentCriminals = [
    { id: 'poi-001', name: 'Sample POI', status: 'active', createdAt: new Date() }
  ];

  return {
    recentCases,
    recentCriminals,
  };
};
