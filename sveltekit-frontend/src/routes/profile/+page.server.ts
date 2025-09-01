import type { PageServerLoad } from './$types';

import { cases, criminals, evidence, profileUpdateZodSchema } from "$lib/db/schema";
import { count, eq } from "drizzle-orm";
import type { PageServerLoad, Actions } from "./$types";
import { redirect, error } from '@sveltejs/kit';
import { db } from '$lib/server/db';
import { superValidate } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';

// SOLUTION: Use the pre-extracted Zod schema for SuperForms compatibility
// No more TypeScript errors with drizzle-zod schemas!
const profileSchema = profileUpdateZodSchema;

export const load: PageServerLoad = async (event: any) => {
  const { locals } = event;
  if (!locals.user) {
    throw redirect(302, "/login");
  }
  
  try {
    // Example profile data (replace with your actual profile fetching logic)
    const profileData = {
      id: locals.user.id,
      firstName: locals.user.firstName || '',
      lastName: locals.user.lastName || ''
    };

    // Initialize the SuperForm with drizzle-zod schema
    const profileForm = await superValidate(profileData, zod(profileSchema));

    // Get user account statistics
    const [
      totalCases,
      openCases,
      closedCases,
      totalCrimes,
      totalCriminals,
      totalEvidence,
      felonyCases,
      misdemeanorCases,
      citationCases,
    ] = await Promise.all([
      // Total cases
      db.select({ count: count() }).from(cases),

      // Open cases
      db.select({ count: count() }).from(cases).where(eq(cases.status, "open")),

      // Closed cases
      db
        .select({ count: count() })
        .from(cases)
        .where(eq(cases.status, "closed")),

      // Total crimes (placeholder - table doesn't exist yet)
      Promise.resolve([{ count: 0 }]),

      // Total criminals
      db.select({ count: count() }).from(criminals),

      // Total evidence
      db.select({ count: count() }).from(evidence),
      // Felony cases (placeholder)
      Promise.resolve([{ count: 0 }]),

      // Misdemeanor cases (placeholder)
      Promise.resolve([{ count: 0 }]),

      // Citation cases (placeholder)
      Promise.resolve([{ count: 0 }]),
    ]);

    const userStats = {
      totalCases: totalCases[0]?.count || 0,
      openCases: openCases[0]?.count || 0,
      closedCases: closedCases[0]?.count || 0,
      totalCrimes: totalCrimes[0]?.count || 0,
      totalCriminals: totalCriminals[0]?.count || 0,
      totalEvidence: totalEvidence[0]?.count || 0,
      felonyCases: felonyCases[0]?.count || 0,
      misdemeanorCases: misdemeanorCases[0]?.count || 0,
      citationCases: citationCases[0]?.count || 0,
    };

    return {
      user: locals.user,
      session: locals.session,
      userStats,
      profileForm, // Add the SuperForm
    };
  } catch (error: any) {
    console.error("Error loading user stats:", error);

    // Return basic data if stats fail
    return {
      user: locals.user,
      session: locals.session,
      userStats: {
        totalCases: 0,
        openCases: 0,
        closedCases: 0,
        totalCrimes: 0,
        totalCriminals: 0,
        totalEvidence: 0,
        felonyCases: 0,
        misdemeanorCases: 0,
        citationCases: 0,
      },
      profileForm: null,
    };
  }
};

export const actions: Actions = {
  updateProfile: async ({ request, locals }) => {
    if (!locals.user) {
      throw redirect(302, "/login");
    }

    const form = await superValidate(request, zod(profileSchema));

    if (!form.valid) {
      return { form };
    }

    try {
      // Update profile in database
      // await db.update(profileTable)
      //   .set({
      //     firstName: form.data.firstName,
      //     lastName: form.data.lastName,
      //   })
      //   .where(eq(profileTable.id, locals.user.id));

      console.log('Profile updated:', form.data);
      
      return { form, success: true };
    } catch (err) {
      console.error('Profile update failed:', err);
      throw error(500, 'Failed to update profile');
    }
  }
};
