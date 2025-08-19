import { URL } from "url";
// @ts-nocheck
import { json } from "@sveltejs/kit";
// Orphaned content: import {

import { personsOfInterest } from "$lib/server/db/schema-postgres";
// Orphaned content: import {

export async function GET({ url }) {
  try {
    const caseId = url.searchParams.get('caseId');
    let query = db.select().from(personsOfInterest).limit(500);
    if (caseId) {
      query = query.where(eq(personsOfInterest.caseId, caseId));
    }
    const results = await query;
    return json(results);
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

export async function POST({ request }) {
  try {
    const body = await request.json();
    const [created] = await db
      .insert(personsOfInterest)
      .values({
        caseId: body.caseId,
        name: body.name,
        role: body.role || 'suspect',
        riskLevel: body.riskLevel || 'unknown',
        status: body.status || 'active',
        description: body.description || null,
        tags: body.tags || [],
        createdBy: body.userId || null,
      })
      .returning();
    return json(created, { status: 201 });
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

export async function PUT({ request }) {
  try {
    const body = await request.json();
    if (!body.id) return json({ error: 'id required' }, { status: 400 });
    const [updated] = await db
      .update(personsOfInterest)
      .set({
        name: body.name,
        role: body.role,
        riskLevel: body.riskLevel,
        status: body.status,
        description: body.description,
        tags: body.tags,
        updatedAt: new Date(),
      })
      .where(eq(personsOfInterest.id, body.id))
      .returning();
    return json(updated);
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

export async function DELETE({ url }) {
  try {
    const id = url.searchParams.get('id');
    if (!id) return json({ error: 'id required' }, { status: 400 });
    await db.delete(personsOfInterest).where(eq(personsOfInterest.id, id));
    return json({ success: true });
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

