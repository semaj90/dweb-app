import { json } from "@sveltejs/kit";
import { cases, evidence, users } from "$lib/server/db/schema-postgres";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


export const GET: RequestHandler = async ({ url, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const hash = url.searchParams.get("hash");
  if (!hash) {
    return json({ error: "Hash parameter required" }, { status: 400 });
  }
  // Validate hash format (SHA256 should be 64 hex characters)
  if (!/^[a-f0-9]{64}$/i.test(hash)) {
    return json(
      {
        error:
          "Invalid hash format. Expected 64-character hexadecimal string (SHA256)",
      },
      { status: 400 },
    );
  }
  try {
    // Search for evidence with matching hash
    const evidenceResults = await db
      .select({
        id: evidence.id,
        caseId: evidence.caseId,
        title: evidence.title,
        description: evidence.description,
        fileType: evidence.fileType,
        fileName: evidence.fileName,
        fileSize: evidence.fileSize,
        hash: evidence.hash,
        fileUrl: evidence.fileUrl,
        uploadedAt: evidence.uploadedAt,
        uploadedBy: evidence.uploadedBy,
        caseName: cases.name,
        caseNumber: cases.caseNumber,
        uploaderName: users.name,
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id))
      .leftJoin(users, eq(evidence.uploadedBy, users.id))
      .where(eq(evidence.hash, hash.toLowerCase()));

    if (evidenceResults.length === 0) {
      return json({
        found: false,
        message: "No evidence found with the specified hash",
        hash,
      });
    }
    return json({
      found: true,
      message: `Found ${evidenceResults.length} evidence item(s) matching the hash`,
      hash,
      evidence: evidenceResults,
    });
  } catch (error: any) {
    console.error("Error searching evidence by hash:", error);
    return json(
      {
        error: "Failed to search evidence by hash",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const { hash, evidenceId } = await request.json();

  if (!hash || !evidenceId) {
    return json({ error: "Hash and evidenceId required" }, { status: 400 });
  }
  // Validate hash format
  if (!/^[a-f0-9]{64}$/i.test(hash)) {
    return json(
      {
        error:
          "Invalid hash format. Expected 64-character hexadecimal string (SHA256)",
      },
      { status: 400 },
    );
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
    const providedHash = hash.toLowerCase();
    const storedHash = item.hash?.toLowerCase();

    // Compare hashes
    const isMatch = storedHash === providedHash;

    return json({
      evidenceId,
      fileName: item.fileName,
      providedHash,
      storedHash,
      isMatch,
      message: isMatch
        ? "Hash verification successful - file integrity confirmed"
        : "Hash verification failed - file may have been modified or corrupted",
      uploadedAt: item.uploadedAt,
      fileSize: item.fileSize,
    });
  } catch (error: any) {
    console.error("Error verifying evidence hash:", error);
    return json(
      {
        error: "Failed to verify evidence hash",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
