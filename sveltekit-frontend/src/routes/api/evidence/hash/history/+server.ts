
import { evidence, hashVerifications, users } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


export const POST: RequestHandler = async ({ request, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const {
    evidenceId,
    verifiedHash,
    method = "manual",
    notes,
  } = await request.json();

  if (!evidenceId || !verifiedHash) {
    return json(
      { error: "evidenceId and verifiedHash required" },
      { status: 400 },
    );
  }
  // Validate hash format
  if (!/^[a-f0-9]{64}$/i.test(verifiedHash)) {
    return json({ error: "Invalid hash format" }, { status: 400 });
  }
  try {
    // Get the evidence item
    const evidenceItem = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (evidenceItem.length === 0) {
      return json({ error: "Evidence not found" }, { status: 404 });
    }
    const item = evidenceItem[0];
    const storedHash = item.hash;
    const result = storedHash?.toLowerCase() === verifiedHash.toLowerCase();

    // Insert verification record using Drizzle
    const verificationResult = await db
      .insert(hashVerifications)
      .values({
        evidenceId,
        verifiedHash: verifiedHash.toLowerCase(),
        storedHash,
        result,
        verificationMethod: method,
        verifiedBy: userId,
        notes,
      })
      .returning();

    return json({
      success: true,
      result,
      evidenceId,
      fileName: item.fileName,
      verifiedHash: verifiedHash.toLowerCase(),
      storedHash,
      verificationId: verificationResult[0]?.id,
      message: result
        ? "Hash verification successful - file integrity confirmed"
        : "Hash verification failed - file may have been modified",
      verifiedAt: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error("Error recording hash verification:", error);
    return json(
      {
        error: "Failed to record verification",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const GET: RequestHandler = async ({ url, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    const limit = parseInt(url.searchParams.get("limit") || "20");
    const offset = parseInt(url.searchParams.get("offset") || "0");
    const evidenceId = url.searchParams.get("evidenceId");

    let query = db
      .select({
        id: hashVerifications.id,
        evidenceId: hashVerifications.evidenceId,
        verifiedHash: hashVerifications.verifiedHash,
        storedHash: hashVerifications.storedHash,
        result: hashVerifications.result,
        verificationMethod: hashVerifications.verificationMethod,
        notes: hashVerifications.notes,
        verifiedAt: hashVerifications.verifiedAt,
        verifierName: users.name,
        evidenceTitle: evidence.title,
        evidenceFileName: evidence.fileName,
      })
      .from(hashVerifications)
      .leftJoin(users, eq(hashVerifications.verifiedBy, users.id))
      .leftJoin(evidence, eq(hashVerifications.evidenceId, evidence.id))
      .orderBy(desc(hashVerifications.verifiedAt))
      .limit(limit)
      .offset(offset);

    if (evidenceId) {
      query = query.where(eq(hashVerifications.evidenceId, evidenceId)) as any;
    }
    const verifications = await query;

    return json(verifications);
  } catch (error: any) {
    console.error("Error fetching hash verification history:", error);
    return json(
      { error: "Failed to fetch verification history" },
      { status: 500 },
    );
  }
};
