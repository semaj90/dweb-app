
import { error, json } from "@sveltejs/kit";
import { appendFile, mkdir } from "drizzle-orm";
import { tmpdir } from "os";

const UPLOAD_DIR = join(tmpdir(), "chunked-uploads");

// Receives and appends a single chunk
export async function POST({ request }) {
  try {
    await mkdir(UPLOAD_DIR, { recursive: true });
    const formData = await request.formData();
    const chunk = formData.get("chunk") as File;
    const fileId = formData.get("fileId") as string;

    if (!chunk || !fileId) {
      throw error(400, "Invalid chunk data.");
    }
    const tempFilePath = join(UPLOAD_DIR, fileId);
    await appendFile(tempFilePath, Buffer.from(await chunk.arrayBuffer()));

    return json({ success: true });
  } catch (err) {
    console.error(err);
    throw error(500, "Failed to process chunk.");
  }
}
