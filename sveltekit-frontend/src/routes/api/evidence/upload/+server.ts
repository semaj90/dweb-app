// API endpoint for file uploads
import { error, json } from "@sveltejs/kit";
import { writeFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request, locals }) => {
  // Check authentication
  const session = await locals.auth?.validate();
  if (!session) {
    throw error(401, "Unauthorized");
  }
  const data = await request.formData();
  const file = data.get("file") as File | null;
  const caseId = data.get("caseId") as string | null;

  if (!file) {
    throw error(400, "No file provided");
  }
  if (!caseId) {
    throw error(400, "No case ID provided");
  }
  // Validate file type and size
  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "application/pdf",
    "video/mp4",
    "audio/mpeg",
  ];
  const maxSize = 50 * 1024 * 1024; // 50MB

  if (!allowedTypes.includes(file.type)) {
    throw error(400, "Invalid file type");
  }
  if (file.size > maxSize) {
    throw error(400, "File too large (max 50MB)");
  }
  try {
    // Create upload directory if it doesn't exist
    const uploadDir = join("static", "uploads", "evidence", caseId);
    if (!existsSync(uploadDir)) {
      await mkdir(uploadDir, { recursive: true });
    }
    // Generate unique filename
    const timestamp = Date.now();
    const randomId = crypto.randomUUID().slice(0, 8);
    const extension = file.name.split(".").pop() || "bin";
    const filename = `${timestamp}-${randomId}.${extension}`;
    const filePath = join(uploadDir, filename);

    // Save file
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(filePath, buffer);

    // Return the public URL
    const publicUrl = `/uploads/evidence/${caseId}/${filename}`;

    return json({
      success: true,
      url: publicUrl,
      filename: file.name,
      size: file.size,
      type: file.type,
    });
  } catch (err) {
    console.error("File upload error:", err);
    throw error(500, "Failed to save file");
  }
};
