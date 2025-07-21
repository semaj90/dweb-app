import { personsOfInterest } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { eq } from "drizzle-orm";
import { db } from "$lib/server/db/index";
import type { RequestHandler } from "./$types";

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
  } catch (error) {
    console.error("Error fetching POI:", error);
    return json(
      { error: "Failed to fetch person of interest" },
      { status: 500 },
    );
  }
};

export const PUT: RequestHandler = async ({ request, params }) => {
  try {
    const data = await request.json();

    const [poi] = await db
      .update(personsOfInterest)
      .set({
        name: data.name,
        aliases: data.aliases,
        profileData: data.profileData,
        posX: data.posX,
        posY: data.posY,
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
  } catch (error) {
    console.error("Error updating POI:", error);
    return json(
      { error: "Failed to update person of interest" },
      { status: 500 },
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
  } catch (error) {
    console.error("Error deleting POI:", error);
    return json(
      { error: "Failed to delete person of interest" },
      { status: 500 },
    );
  }
};
