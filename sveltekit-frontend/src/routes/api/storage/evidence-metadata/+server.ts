// @ts-nocheck
import { json } from "@sveltejs/kit";

import { evidence } from "$lib/server/db/schema-postgres";

export async function POST({ request }) {
  try {
    const body = await request.json();
    const {
      caseId,
      fileName,
      fileType,
      size,
      minioBucket = 'legal-documents',
      minioObjectKey,
      hash,
      uploadedBy,
    } = body;

    if (!caseId || !fileName || !minioObjectKey) {
      return json({ error: 'caseId, fileName, minioObjectKey required' }, { status: 400 });
    }

    const [record] = await db
      .insert(evidence)
      .values({
        caseId,
        title: fileName,
        fileName,
        fileType,
        sizeBytes: size,
        storagePath: `${minioBucket}/${minioObjectKey}`,
        hash,
        status: 'uploaded',
        uploadedBy,
        metadata: { bucket: minioBucket, objectKey: minioObjectKey },
      })
      .returning();

    return json(record, { status: 201 });
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

export async function PUT({ request }) {
  try {
    const body = await request.json();
    if (!body.id) return json({ error: 'id required' }, { status: 400 });
    const [updated] = await db
      .update(evidence)
      .set({
        description: body.description,
        tags: body.tags,
        updatedAt: new Date(),
      })
      .where(eq(evidence.id, body.id))
      .returning();
    return json(updated);
  } catch (e: any) {
    return json({ error: e.message }, { status: 500 });
  }
}

