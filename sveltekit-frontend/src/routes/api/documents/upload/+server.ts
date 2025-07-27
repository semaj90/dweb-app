import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { parseStringPromise } from "xml2js";
import pdf from "pdf-parse";
import { OpenAIEmbeddings } from "@langchain/openai";
import { drizzle } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import { legalDocs } from "$lib/server/db/schema";
import { v4 as uuidv4 } from "uuid";

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});
const db = drizzle(pool);
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

async function processXml(buffer: Buffer): Promise<string> {
  const xmlString = buffer.toString("utf-8");
  const result = await parseStringPromise(xmlString, {
    explicitArray: false,
    ignoreAttrs: true,
  });
  // This is a placeholder. You need to customize this to extract text from your XML structure.
  return JSON.stringify(result);
}

async function processPdf(buffer: Buffer): Promise<string> {
  const data = await pdf(buffer);
  return data.text;
}

async function getAiSummary(
  text: string,
  verbose: boolean,
  thinking: boolean
): Promise<any> {
  // Placeholder for local Gemma model call
  // In a real implementation, you would call your local LLM API here.
  const summary = `This is a ${verbose ? "verbose" : "standard"} summary. Thinking mode was ${thinking ? "on" : "off"}. The document contains: ${text.substring(0, 200)}...`;

  // Simulate a RAG-style synthesis of high-scoring chunks
  if (thinking) {
    // Simulate querying for more context
    await new Promise((resolve) => setTimeout(resolve, 1000)); // Simulate delay
    return {
      summary,
      synthesizedAnalysis:
        "Based on related legal precedents, this document appears to be a standard deed of transfer with no unusual clauses.",
      confidenceScore: 0.95,
    };
  }

  return { summary, confidenceScore: 0.88 };
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const verbose = formData.get("verbose") === "true";
    const thinking = formData.get("thinking") === "true";

    if (!file) {
      return json({ error: "No file uploaded" }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    let textContent = "";

    if (file.type === "application/pdf") {
      textContent = await processPdf(buffer);
    } else if (file.type === "text/xml" || file.name.endsWith(".xml")) {
      textContent = await processXml(buffer);
    } else {
      return json({ error: "Unsupported file type" }, { status: 400 });
    }

    const embedding = await embeddings.embedQuery(textContent);
    const docId = uuidv4();

    await db.insert(legalDocs).values({
      id: docId,
      content: textContent,
      embedding: embedding,
      fileName: file.name,
      createdAt: new Date(),
    });

    const analysis = await getAiSummary(textContent, verbose, thinking);

    return json({
      id: docId,
      fileName: file.name,
      ...analysis,
    });
  } catch (error: any) {
    console.error("Upload error:", error);
    return json(
      { error: error.message || "Failed to process file" },
      { status: 500 }
    );
  }
};
