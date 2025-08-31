
import { users } from "$lib/server/db/schema-postgres";
import { json } from "@sveltejs/kit";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from 'path';
import { eq } from 'drizzle-orm';
import { db } from "$lib/server/db/index";
import type { RequestHandler } from './$types';


const UPLOAD_DIR = "static/uploads/avatars";
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/svg+xml",
  "image/webp",
];

// Ensure upload directory exists
if (!existsSync(UPLOAD_DIR)) {
  mkdirSync(UPLOAD_DIR, { recursive: true });
}
export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    console.log("Avatar upload started for user:", locals.user.id);

    const formData = await request.formData();
    console.log(
      "FormData received, entries:",
      Array.from(formData.entries()).map(([k, v]) => [k, typeof v]),
    );

    const file = formData.get("avatar") as File;

    if (!file) {
      console.log("No file found in formData");
      return json({ error: "No file provided" }, { status: 400 });
    }
    console.log("File details:", {
      name: file.name,
      type: file.type,
      size: file.size,
    });

    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
      console.log("Invalid file type:", file.type);
      return json(
        {
          error: "Invalid file type. Allowed: JPEG, PNG, GIF, SVG, WebP",
        },
        { status: 400 },
      );
    }
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      console.log("File too large:", file.size);
      return json(
        {
          error: "File too large. Maximum size: 5MB",
        },
        { status: 400 },
      );
    }
    // Generate unique filename
    const timestamp = Date.now();
    const extension = file.name.split(".").pop()?.toLowerCase() || "jpg";
    const filename = `avatar_${locals.user.id}_${timestamp}.${extension}`;
    const filepath = join(UPLOAD_DIR, filename);

    console.log("Saving file to:", filepath);

    // Convert file to buffer and save
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    writeFileSync(filepath, buffer);

    console.log("File saved successfully");

    // Update user's avatar URL in database
    const avatarUrl = `/uploads/avatars/${filename}`;
    await db
      .update(users)
      .set({ avatarUrl })
      .where(eq(users.id, locals.user.id));

    console.log("Database updated successfully");

    return json({
      success: true,
      avatarUrl,
      message: "Avatar uploaded successfully",
    });
  } catch (error: any) {
    console.error("Avatar upload error:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    return json({ error: `Upload failed: ${errorMessage}` }, { status: 500 });
  }
};

export const DELETE: RequestHandler = async ({ locals }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  try {
    // Remove avatar URL from database
    await db
      .update(users)
      .set({ avatarUrl: null })
      .where(eq(users.id, locals.user.id));

    return json({
      success: true,
      message: "Avatar removed successfully",
    });
  } catch (error: any) {
    console.error("Avatar removal error:", error);
    return json({ error: "Removal failed" }, { status: 500 });
  }
};
