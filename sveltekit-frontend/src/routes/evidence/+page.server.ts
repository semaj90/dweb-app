import { evidence } from "$lib/server/db/schema-postgres";
import { error, fail } from "@sveltejs/kit";
import { eq, and } from "drizzle-orm";
import { superValidate } from "sveltekit-superforms";
import { zod } from "sveltekit-superforms/adapters";
import { z } from "zod";
import { db } from "$lib/server/db/index";
import { createRedisClient } from "$lib/server/redis";
import type { PageServerLoad } from "./$types";

// Schema for validating evidence form data
const evidenceSchema = z.object({
  id: z.string().optional(),
  caseId: z.string().uuid(),
  title: z.string().min(1, "Title is required"),
  description: z.string().optional(),
  type: z.enum(["document", "image", "video", "audio", "other"]),
  url: z.string().url("Must be a valid URL").optional(),
  fileSize: z.number().optional(),
  mimeType: z.string().optional(),
  tags: z.array(z.string()).default([]),
  metadata: z.record(z.unknown()).optional(),
});

export const load: PageServerLoad = async ({ url, locals }) => {
  const user = locals.user;
  if (!user) {
    throw error(401, "Authentication required");
  }
  const caseId = url.searchParams.get("caseId");
  const limit = parseInt(url.searchParams.get("limit") || "50");
  const offset = parseInt(url.searchParams.get("offset") || "0");

  try {
    // Create cache key
    const cacheKey = `evidence:${caseId || "all"}:${limit}:${offset}:${user.id}`;

    // Try to get data from Redis cache
    let evidenceData = null;
    try {
      const redis = await createRedisClient();
      const cached = await redis.get(cacheKey);
      if (cached) {
        evidenceData = JSON.parse(cached);
        await redis.quit();
      }
    } catch (redisError) {
      console.warn("Redis cache unavailable:", redisError);
    }
    // If not in cache, fetch from database
    if (!evidenceData) {
      const conditions = [];

      if (caseId) {
        conditions.push(eq(evidence.caseId, caseId));
      }
      
      const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

      evidenceData = await db
        .select()
        .from(evidence)
        .where(whereClause)
        .limit(limit)
        .offset(offset);

      // Cache the result for 5 minutes
      try {
        const redis = await createRedisClient();
        await redis.setex(cacheKey, 300, JSON.stringify(evidenceData));
        await redis.quit();
      } catch (redisError) {
        console.warn("Failed to cache evidence data:", redisError);
      }
    }
    return {
      evidence: evidenceData || [],
      caseId,
      totalCount: evidenceData?.length || 0,
      hasMore: evidenceData?.length === limit,
    };
  } catch (err) {
    console.error("Failed to load evidence:", err);
    throw error(500, "Failed to load evidence data");
  }
};

export const actions = {
  // CREATE and UPDATE evidence
  save: async ({ request }) => {
    const form = await superValidate(request, zod(evidenceSchema));

    if (!form.valid) {
      return fail(400, { form });
    }
    const { id, ...formData } = form.data;
    
    // Transform form data to match database schema
    const evidenceData = {
      caseId: formData.caseId as string,
      title: formData.title as string,
      description: formData.description as string,
      evidenceType: formData.type as string,
      tags: formData.tags as string[],
      fileUrl: formData.url as string,
      fileSize: formData.fileSize as number,
      mimeType: formData.mimeType as string,
    };

    try {
      if (id) {
        // UPDATE existing evidence
        await db
          .update(evidence)
          .set(evidenceData)
          .where(eq(evidence.id, id as string));
      } else {
        // CREATE new evidence
        await db.insert(evidence).values(evidenceData);
      }
      return { form };
    } catch (error) {
      console.error("Error saving evidence:", error);
      return fail(500, {
        form,
        error: "Failed to save evidence",
      });
    }
  },

  // DELETE evidence
  delete: async ({ request }) => {
    const formData = await request.formData();
    const id = formData.get("id") as string;

    if (!id) {
      return fail(400, { error: "Missing evidence ID" });
    }
    try {
      await db.delete(evidence).where(eq(evidence.id, id));
      return { success: true };
    } catch (error) {
      console.error("Error deleting evidence:", error);
      return fail(500, { error: "Failed to delete evidence" });
    }
  },
};
