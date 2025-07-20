import { error, json } from "@sveltejs/kit";
import { mkdir, rename } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

const UPLOAD_DIR = join(tmpdir(), "chunked-uploads");

// Assembles the final file from chunks
export async function POST({ request }) {
  try {
    const { fileId, filename } = await request.json();

    if (!fileId || !filename) {
      throw error(400, "Invalid finalization data.");
    }
    const tempFilePath = join(UPLOAD_DIR, fileId);
    const finalDirPath = "./uploads";
    const finalFilePath = join(finalDirPath, filename);

    await mkdir(finalDirPath, { recursive: true });
    await rename(tempFilePath, finalFilePath); // Move the assembled file

    return json({ url: `/${finalFilePath}` });
  } catch (err) {
    console.error(err);
    throw error(500, "Failed to finalize upload.");
  }
}
