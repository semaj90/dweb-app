
import { personsOfInterest } from "$lib/server/db/schema-postgres";

import { eq } from "drizzle-orm";
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ params }) => {
  try {
    const [poi] = await db
      .select()
      .from(personsOfInterest)
      .where(eq(personsOfInterest.id, params.id));

    if (!poi) {
      return json({ error: "Person of interest not found" }, { status: 404 });
    }
    return json(poi);
  } catch (error: any) {
    console.error("Error fetching POI:", error);
    return json(
      { error: "Failed to fetch person of interest" },
      { status: 500 }
    );
  }
};

export const PUT: RequestHandler = async ({ request, params }) => {
  try {
    const data = await request.json();

    // Remove posX/posY, use position object if needed
    const [poi] = await db
      .update(personsOfInterest)
      .set({
        name: data.name,
        aliases: data.aliases,
        profileData: data.profileData,
        position: data.position || {},
        relationship: data.relationship,
        threatLevel: data.threatLevel,
        status: data.status,
        tags: data.tags,
        updatedAt: new Date(),
      })
      .where(eq(personsOfInterest.id, params.id))
      .returning();

    if (!poi) {
      return json({ error: "Person of interest not found" }, { status: 404 });
    }
    return json(poi);
  } catch (error: any) {
    console.error("Error updating POI:", error);
    return json(
      { error: "Failed to update person of interest" },
      { status: 500 }
    );
  }
};

export const DELETE: RequestHandler = async ({ params }) => {
  try {
    const [poi] = await db
      .delete(personsOfInterest)
      .where(eq(personsOfInterest.id, params.id))
      .returning();

    if (!poi) {
      return json({ error: "Person of interest not found" }, { status: 404 });
    }
    return json({ success: true });
  } catch (error: any) {
    console.error("Error deleting POI:", error);
    return json(
      { error: "Failed to delete person of interest" },
      { status: 500 }
    );
  }
};
