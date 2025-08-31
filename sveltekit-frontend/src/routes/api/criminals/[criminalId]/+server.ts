import { json } from "@sveltejs/kit";
import { criminals } from "$lib/server/db/schema-postgres";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const criminalId = params.criminalId;
    if (!criminalId) {
      return json({ error: "Criminal ID is required" }, { status: 400 });
    }
    const criminalResult = await db
      .select()
      .from(criminals)
      .where(eq(criminals.id, criminalId))
      .limit(1);

    if (!criminalResult.length) {
      return json({ error: "Criminal record not found" }, { status: 404 });
    }
    return json(criminalResult[0]);
  } catch (error: any) {
    console.error("Error fetching criminal record:", error);
    return json({ error: "Failed to fetch criminal record" }, { status: 500 });
  }
};

export const PUT: RequestHandler = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const criminalId = params.criminalId;
    if (!criminalId) {
      return json({ error: "Criminal ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if criminal exists
    const existingCriminal = await db
      .select()
      .from(criminals)
      .where(eq(criminals.id, criminalId))
      .limit(1);

    if (!existingCriminal.length) {
      return json({ error: "Criminal record not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Map frontend fields to schema fields - only update provided fields
    if (data.firstName !== undefined)
      updateData.firstName = data.firstName.trim();
    if (data.lastName !== undefined) updateData.lastName = data.lastName.trim();
    if (data.middleName !== undefined)
      updateData.middleName = data.middleName?.trim() || null;
    if (data.aliases !== undefined) updateData.aliases = data.aliases;
    if (data.dateOfBirth !== undefined) {
      updateData.dateOfBirth = data.dateOfBirth
        ? new Date(data.dateOfBirth)
        : null;
    }
    if (data.placeOfBirth !== undefined)
      updateData.placeOfBirth = data.placeOfBirth?.trim() || null;
    if (data.address !== undefined)
      updateData.address = data.address?.trim() || null;
    if (data.phone !== undefined) updateData.phone = data.phone?.trim() || null;
    if (data.email !== undefined) updateData.email = data.email?.trim() || null;
    if (data.socialSecurityNumber !== undefined)
      updateData.socialSecurityNumber =
        data.socialSecurityNumber?.trim() || null;
    if (data.driversLicense !== undefined)
      updateData.driversLicense = data.driversLicense?.trim() || null;
    if (data.height !== undefined)
      updateData.height = data.height ? Number(data.height) : null;
    if (data.weight !== undefined)
      updateData.weight = data.weight ? Number(data.weight) : null;
    if (data.eyeColor !== undefined)
      updateData.eyeColor = data.eyeColor?.trim() || null;
    if (data.hairColor !== undefined)
      updateData.hairColor = data.hairColor?.trim() || null;
    if (data.distinguishingMarks !== undefined)
      updateData.distinguishingMarks = data.distinguishingMarks?.trim() || null;
    if (data.photoUrl !== undefined)
      updateData.photoUrl = data.photoUrl?.trim() || null;
    if (data.fingerprints !== undefined)
      updateData.fingerprints = data.fingerprints;
    if (data.threatLevel !== undefined)
      updateData.threatLevel = data.threatLevel;
    if (data.status !== undefined) updateData.status = data.status;
    if (data.notes !== undefined) updateData.notes = data.notes?.trim() || null;
    if (data.aiSummary !== undefined)
      updateData.aiSummary = data.aiSummary?.trim() || null;
    if (data.aiTags !== undefined) updateData.aiTags = data.aiTags;

    const [updatedCriminal] = await db
      .update(criminals)
      .set(updateData)
      .where(eq(criminals.id, criminalId))
      .returning();

    return json(updatedCriminal);
  } catch (error: any) {
    console.error("Error updating criminal record:", error);
    return json({ error: "Failed to update criminal record" }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ params, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const criminalId = params.criminalId;
    if (!criminalId) {
      return json({ error: "Criminal ID is required" }, { status: 400 });
    }
    // Check if criminal exists
    const existingCriminal = await db
      .select()
      .from(criminals)
      .where(eq(criminals.id, criminalId))
      .limit(1);

    if (!existingCriminal.length) {
      return json({ error: "Criminal record not found" }, { status: 404 });
    }
    // Delete the criminal record (cascade will handle related records)
    const [deletedCriminal] = await db
      .delete(criminals)
      .where(eq(criminals.id, criminalId))
      .returning();

    return json({ success: true, deletedCriminal });
  } catch (error: any) {
    console.error("Error deleting criminal record:", error);
    return json({ error: "Failed to delete criminal record" }, { status: 500 });
  }
};

// PATCH endpoint for partial updates (like status changes)
export const PATCH: RequestHandler = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return json({ error: "Not authenticated" }, { status: 401 });
    }
    if (!db) {
      return json({ error: "Database not available" }, { status: 500 });
    }
    const criminalId = params.criminalId;
    if (!criminalId) {
      return json({ error: "Criminal ID is required" }, { status: 400 });
    }
    const data = await request.json();

    // Check if criminal exists
    const existingCriminal = await db
      .select()
      .from(criminals)
      .where(eq(criminals.id, criminalId))
      .limit(1);

    if (!existingCriminal.length) {
      return json({ error: "Criminal record not found" }, { status: 404 });
    }
    const updateData: Record<string, any> = {
      updatedAt: new Date(),
    };

    // Handle specific patch operations
    if (data.operation === "updateThreatLevel") {
      updateData.threatLevel = data.threatLevel;
    } else if (data.operation === "updateStatus") {
      updateData.status = data.status;
    } else if (data.operation === "addAlias") {
      const currentAliases = (existingCriminal[0].aliases as string[]) || [];
      if (!currentAliases.includes(data.alias)) {
        updateData.aliases = [...currentAliases, data.alias];
      }
    } else if (data.operation === "removeAlias") {
      const currentAliases = (existingCriminal[0].aliases as string[]) || [];
      updateData.aliases = currentAliases.filter(
        (alias) => alias !== data.alias,
      );
    } else if (data.operation === "updatePhoto") {
      updateData.photoUrl = data.photoUrl;
    } else if (data.operation === "updateFingerprints") {
      updateData.fingerprints = data.fingerprints;
    } else {
      // Regular field updates
      Object.keys(data).forEach((key) => {
        if (key !== "operation") {
          updateData[key] = data[key];
        }
      });
    }
    const [updatedCriminal] = await db
      .update(criminals)
      .set(updateData)
      .where(eq(criminals.id, criminalId))
      .returning();

    return json(updatedCriminal);
  } catch (error: any) {
    console.error("Error patching criminal record:", error);
    return json({ error: "Failed to update criminal record" }, { status: 500 });
  }
};
