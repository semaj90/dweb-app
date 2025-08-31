
import { cases, evidence } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const { hashes, evidenceIds } = await request.json();

  if (!hashes && !evidenceIds) {
    return json(
      { error: "Either hashes or evidenceIds array required" },
      { status: 400 },
    );
  }
  try {
    let results: any[] = [];

    if (hashes && Array.isArray(hashes)) {
      // Bulk hash search
      const hashResults = await db
        .select({
          id: evidence.id,
          title: evidence.title,
          fileName: evidence.fileName,
          hash: evidence.hash,
          fileSize: evidence.fileSize,
          uploadedAt: evidence.uploadedAt,
          caseName: cases.name,
          caseNumber: cases.caseNumber,
        })
        .from(evidence)
        .leftJoin(cases, eq(evidence.caseId, cases.id))
        .where(inArray(evidence.hash, hashes));

      results = hashes.map((hash) => {
        const found = hashResults.filter((item) => item.hash === hash);
        return {
          hash,
          found: found.length > 0,
          evidence: found,
        };
      });
    }
    if (evidenceIds && Array.isArray(evidenceIds)) {
      // Bulk hash verification for specific evidence
      const evidenceItems = await db
        .select()
        .from(evidence)
        .where(inArray(evidence.id, evidenceIds));

      const verificationResults = evidenceItems.map((item) => ({
        evidenceId: item.id,
        fileName: item.fileName,
        storedHash: item.hash,
        hasHash: !!item.hash,
        uploadedAt: item.uploadedAt,
      }));

      results = verificationResults;
    }
    // Calculate summary statistics
    const stats = {
      totalProcessed: results.length,
      verified: results.filter((r) => r.found || r.hasHash).length,
      missing: results.filter((r) => !r.found && !r.hasHash).length,
      processedAt: new Date().toISOString(),
    };

    return json({
      success: true,
      results,
      stats,
      message: `Processed ${results.length} item(s) for bulk hash operations`,
    });
  } catch (error: any) {
    console.error("Bulk hash operation failed:", error);
    return json(
      {
        error: "Bulk operation failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};

// GET endpoint for bulk status checking
export const GET: RequestHandler = async ({ url, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    // Get recent hash verification statistics
    const recentEvidence = await db
      .select({
        id: evidence.id,
        fileName: evidence.fileName,
        hash: evidence.hash,
        uploadedAt: evidence.uploadedAt,
      })
      .from(evidence)
      .orderBy(evidence.uploadedAt)
      .limit(100);

    const stats = {
      totalEvidence: recentEvidence.length,
      withHashes: recentEvidence.filter((e: any) => e.hash).length,
      withoutHashes: recentEvidence.filter((e: any) => !e.hash).length,
      hashCoverage:
        recentEvidence.length > 0
          ? (
              (recentEvidence.filter((e: any) => e.hash).length /
                recentEvidence.length) *
              100
            ).toFixed(1)
          : "0",
      lastUpdated: new Date().toISOString(),
    };

    return json({
      stats,
      recentEvidence: recentEvidence.slice(0, 10), // Return 10 most recent
    });
  } catch (error: any) {
    console.error("Failed to get bulk hash status:", error);
    return json(
      {
        error: "Failed to get hash status",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    );
  }
};
