
import { personsOfInterest } from "$lib/server/db/schema-postgres";

import { eq } from "drizzle-orm";
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ params }) => {
  try {
    const caseId = params.caseId;

    const pois = await db
      .select()
      .from(personsOfInterest)
      .where(eq(personsOfInterest.caseId, caseId));

    return json(pois);
  } catch (error: any) {
    console.error("Error fetching POIs:", error);
    return json(
      { error: "Failed to fetch persons of interest" },
      { status: 500 }
    );
  }
};

export const POST: RequestHandler = async ({ request, params }) => {
  try {
    const caseId = params.caseId;
    const data = await request.json();

    // Remove posX/posY, use position object if needed
    const [poi] = await db
      .insert(personsOfInterest)
      .values({
        caseId,
        name: data.name || "New Person of Interest",
        aliases: data.aliases || [],
        profileData: data.profileData || {
          who: "",
          what: "",
          why: "",
          how: "",
        },
        position: data.position || {},
        relationship: data.relationship,
        threatLevel: data.threatLevel || "low",
        status: data.status || "active",
        tags: data.tags || [],
        createdBy: data.createdBy || "system", // TODO: Get from session
      })
      .returning();

    return json(poi, { status: 201 });
  } catch (error: any) {
    console.error("Error creating POI:", error);
    return json(
      { error: "Failed to create person of interest" },
      { status: 500 }
    );
  }
};
