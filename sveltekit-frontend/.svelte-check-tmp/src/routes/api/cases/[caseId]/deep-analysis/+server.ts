import { json } from "@sveltejs/kit";
import { db } from "$lib/server/db/index";
import { caseActivities, cases, evidence } from "$lib/server/db/index";
import { eq } from "drizzle-orm";
import { QdrantClient } from "@qdrant/js-client-rest";
import type { RequestHandler } from './$types';


// Environment variables fallback
const env = process.env || {};

const QDRANT_URL = env.QDRANT_URL || "http://localhost:6333";
const NLP_SERVICE_URL = env.LLM_SERVICE_URL || "http://localhost:8000";
const OPENAI_API_KEY = env.OPENAI_API_KEY;
const GEMINI_API_KEY = env.GEMINI_API_KEY; // For future use

const qdrantClient = new QdrantClient({ url: QDRANT_URL });

export const POST: RequestHandler = async ({ params, locals, request }) => {
  if (!locals.user) {
    return json({ error: "Not authenticated" }, { status: 401 });
  }
  const { caseId } = params;
  if (!caseId) {
    return json({ error: "Case ID is required" }, { status: 400 });
  }
  const {
    queryText,
    enableMultiLLM,
    complexityLevel = 3,
  } = await request.json();

  if (!queryText) {
    return json({ error: "Query text is required" }, { status: 400 });
  }

  try {
    // --- RAG: RETRIEVAL ---
    const currentCaseResults = await db
      .select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);
    if (!currentCaseResults.length) {
      return json({ error: "Case not found" }, { status: 404 });
    }

    const currentCase = currentCaseResults[0];

    const recentActivities = await db
      .select()
      .from(caseActivities)
      .where(eq(caseActivities.caseId, caseId))
      .orderBy((activities) => activities.createdAt)
      .limit(5);

    // Get recent evidence files
    const recentEvidence = await db
      .select()
      .from(evidence)
      .where(eq(evidence.caseId, caseId))
      .orderBy((evidenceTable) => evidenceTable.uploadedAt)
      .limit(10);

    const embeddingResponse = await fetch(`${NLP_SERVICE_URL}/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: queryText }),
    });

    if (!embeddingResponse.ok) {
      throw new Error("Failed to get embedding from NLP service");
    }
    const queryEmbedding = (await embeddingResponse.json()).embedding;

    // Retrieve from two Qdrant collections in parallel
    const [qdrantFragmentResults, qdrantEvidenceResults] =
      await Promise.allSettled([
        qdrantClient.search("prosecutor_text_fragments", {
          vector: queryEmbedding,
          limit: 3,
          filter: { must: [{ key: "caseId", match: { value: caseId } }] },
          with_payload: true,
        }),
        qdrantClient.search("prosecutor_evidence", {
          vector: queryEmbedding,
          limit: 3,
          filter: { must: [{ key: "caseId", match: { value: caseId } }] },
          with_payload: true,
        }),
      ]);

    const relevantFragments =
      qdrantFragmentResults.status === "fulfilled"
        ? qdrantFragmentResults.value
            .map((hit) => hit.payload?.content)
            .join("\n\n")
        : "";

    const relevantEvidenceSummaries =
      qdrantEvidenceResults.status === "fulfilled"
        ? qdrantEvidenceResults.value
            .map((hit) => hit.payload?.aiSummary)
            .join("\n\n")
        : "";

    // --- RAG: AUGMENTATION ---
    const ragContext = `
            Case Title: ${currentCase.title}
            Case Description: ${currentCase.description}
            Recent Activities: ${recentActivities.map((a) => a.title).join(", ") || "None"}
            Recent Evidence: ${recentEvidence.map((e: any) => e.fileName || e.title).join(", ") || "None"}
            Relevant Case Fragments: ${relevantFragments || "None"}
            Relevant Evidence Summaries: ${relevantEvidenceSummaries || "None"}
        `.trim();

    const basePrompt = `
            Analyze the following query in the context of a legal case. Provide actionable insights and recommendations.
            CONTEXT:
            ---
            ${ragContext}
            ---
            USER QUERY: "${queryText}"
            ANALYSIS:
        `.trim();

    // Define a GBNF grammar to force the local LLM to return a specific JSON structure.
    // This grammar defines an object with a "summary" (string) and "recommendations" (array of strings).
    const jsonGrammar = String.raw`
root   ::= object
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}"
array  ::= "[" ws ( value ("," ws value)* )? "]"
value  ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" (
  [^"\\] |
  "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
)* "\""
number ::= ("-")? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws ::= ([ \t\n]*)
`.trim();

    // --- MULTI-LLM INFERENCE ---
    const promises: Promise<any>[] = [];

    // 1. Local LLM (The Firm's AI)
    promises.push(
      fetch(`${NLP_SERVICE_URL}/generate-with-local-llm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt:
            basePrompt +
            '\n\nReturn your analysis as a valid JSON object with "summary" and "recommendations" keys.',
          max_tokens: 1024,
          grammar: jsonGrammar, // Pass the grammar to constrain the output
        }),
      }).then((res) =>
        res.json().then((data) => ({ source: "firm_ai", data, ok: res.ok }))
      )
    );

    // 2. OpenAI (if enabled and key exists)
    if (enableMultiLLM && OPENAI_API_KEY) {
      promises.push(
        fetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${OPENAI_API_KEY}`,
          },
          body: JSON.stringify({
            model: "gpt-3.5-turbo",
            messages: [{ role: "user", content: basePrompt }],
            max_tokens: 512,
          }),
        }).then((res) =>
          res.json().then((data) => ({ source: "openai", data, ok: res.ok }))
        )
      );
    }
    const settledResults = await Promise.allSettled(promises);

    // --- SYNTHESIS & RESPONSE ---
    const analysisResults: Record<string, any> = {};
    settledResults.forEach((result) => {
      if (result.status === "fulfilled" && result.value.ok) {
        const { source, data } = result.value;
        if (source === "firm_ai") {
          try {
            // The output should be valid JSON because of the grammar
            const parsedResponse = JSON.parse(data.response);
            analysisResults.firm_ai = {
              output: parsedResponse,
              source: "Local LLM (JSON)",
            };
          } catch (e: any) {
            // Fallback if JSON parsing fails despite the grammar (very unlikely)
            console.error(
              "Local LLM output was not valid JSON:",
              data.response
            );
            analysisResults.firm_ai = {
              output: data.response,
              source: "Local LLM (Raw)",
            };
          }
        } else if (source === "openai") {
          analysisResults.openai = {
            output: data.choices[0].message.content,
            source: "OpenAI API",
          };
        }
      } else if (result.status === "fulfilled") {
        // API returned an error
        const { source, data } = result.value;
        analysisResults[source] = { error: data.detail || "API Error" };
      } else {
        // Fetch itself failed
        console.error("Fetch failed:", result.reason);
      }
    });

    return json({ success: true, analysisResults });
  } catch (error: any) {
    console.error("Error in deep analysis endpoint:", error);
    return json({ error: "Failed to perform deep analysis" }, { status: 500 });
  }
};
