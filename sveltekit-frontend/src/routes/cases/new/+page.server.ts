import type { Actions } from "./$types";

import { cases } from "$lib/server/db/schema-postgres";
// Orphaned content: import {
fail, redirect
import { randomUUID } from "crypto";
// Orphaned content: import {

export const actions: Actions = {
  create: async ({ request, locals }) => {
    const form = await request.formData();
    const title = form.get("title")?.toString();
    const description = form.get("description")?.toString();
    const dangerScore = Number(form.get("dangerScore")) || 0;
    const status = form.get("status")?.toString() || "open";
    const aiSummary = form.get("aiSummary")?.toString() || null;

    if (!title || !description) {
      return fail(400, { error: "Title and description are required." });
    }
    const id = randomUUID();
    // Get session from Auth.js
    const user = locals.user;
    const createdBy = user?.id;

    if (!createdBy) {
      return fail(401, { error: "Not authenticated." });
    }
    try {
      await db.insert(cases).values({
        id,
        caseNumber: `CASE-${new Date().getFullYear()}-${Date.now().toString().slice(-6)}`, // Generate unique case number
        title,
        description,
        dangerScore,
        status,
        aiSummary,
        createdBy,
      });
      throw redirect(303, `/cases/${id}`);
    } catch (e) {
      console.error(e);
      return fail(500, { error: "Failed to create case." });
    }
  },
};
