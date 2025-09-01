import { json, error } from "@sveltejs/kit";
import { writeFile } from "fs/promises";
import { join } from "path";
import { spawn } from "child_process";
import type { RequestHandler } from './$types.js';
import { URL } from "url";


export interface AnalyzeRequest {
  caseId: string;
  evidenceFile: string;
  evidenceContent?: string;
}

export interface AnalysisResult {
  sessionId: string;
  status: "processing" | "completed" | "failed";
  step: string;
  outputs: {
    evidence_analysis?: unknown;
    persons_extracted?: unknown;
    neo4j_updates?: unknown;
    case_synthesis?: unknown;
  };
  error?: string;
}

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { caseId, evidenceFile, evidenceContent }: AnalyzeRequest = await request.json();
    
    if (!caseId || !evidenceFile) {
      throw error(400, "Missing required fields: caseId and evidenceFile");
    }

    // Generate session ID
    const sessionId = `analysis_${caseId}_${Date.now()}`;
    const tempDir = join(process.cwd(), "temp", sessionId);

    // If evidenceContent is provided, write it to a file
    let actualEvidenceFile = evidenceFile;
    if (evidenceContent) {
      const tempFile = join(tempDir, "evidence.txt");
      await writeFile(tempFile, evidenceContent, "utf-8");
      actualEvidenceFile = tempFile;
    }

    // Start the analysis pipeline
    const analysisPromise = runAnalysisPipeline(
      caseId,
      actualEvidenceFile,
      sessionId,
    );

    // Return immediate response with session ID
    return json({
      sessionId,
      status: "processing",
      message: "Evidence analysis pipeline started",
      pollUrl: `/api/evidence/analyze/${sessionId}`
    });
  } catch (err: any) {
    console.error("Analysis request failed:", err);
    throw error(500, `Analysis failed: ${err}`);
  }
};

async function runAnalysisPipeline(
  caseId: string,
  evidenceFile: string,
  sessionId: string,
): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    // Run the Node.js analysis script
    const scriptFile = join(
      process.cwd(),
      "..",
      "scripts",
      "analyze-evidence.js",
    );
    
    const child = spawn("node", [scriptFile, caseId, evidenceFile], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"]
    });

    let stdout = "";
    let stderr = "";

    child.stdout?.on("data", (data) => {
      stdout += data.toString();
      console.log(`[${sessionId}] ${data}`);
    });

    child.stderr?.on("data", (data) => {
      stderr += data.toString();
      console.error(`[${sessionId}] ${data}`);
    });

    child.on("close", async (code) => {
      if (code === 0) {
        // Success - parse outputs from the analysis script
        try {
          const result = await parseAnalysisOutputs(sessionId, caseId);
          resolve(result);
        } catch (err: any) {
          reject(err);
        }
      } else {
        reject(
          new Error(`Analysis pipeline failed with code ${code}: ${stderr}`),
        );
      }
    });

    child.on("error", (err) => {
      reject(new Error(`Failed to start analysis pipeline: ${err.message}`));
    });
  });
}

async function parseAnalysisOutputs(
  sessionId: string,
  caseId: string,
): Promise<AnalysisResult> {
  const tempDir = join(process.cwd(), "temp", sessionId);
  
  try {
    // Read all output files
    const outputs: any = {};
    const files = [
      "evidence_analysis.json",
      "persons_extracted.json",
      "neo4j_updates.json",
      "case_synthesis.json",
    ];

    for (const file of files) {
      try {
        const filePath = join(tempDir, file);
        const content = await import("fs").then((fs) =>
          fs.promises.readFile(filePath, "utf-8"),
        );
        const key = file.replace(".json", "");
        outputs[key] = JSON.parse(content);
      } catch (err: any) {
        console.warn(`Could not read ${file}:`, err);
      }
    }

    return {
      sessionId,
      status: "completed",
      step: "case_synthesis",
      outputs
    };
  } catch (err: any) {
    return {
      sessionId,
      status: "failed",
      step: "parsing_outputs",
      outputs: {},
      error: err instanceof Error ? err.message : "Unknown parsing error"
    };
  }
}

// GET endpoint to poll analysis status
export const GET: RequestHandler = async ({ url }) => {
  const sessionId = url.pathname.split("/").pop();
  
  if (!sessionId) {
    throw error(400, "Session ID required");
  }

  // Check if analysis is complete by looking for output files
  const tempDir = join(process.cwd(), "temp", sessionId);
  
  try {
    const result = await parseAnalysisOutputs(sessionId, "unknown");
    return json(result);
  } catch (err: any) {
    return json({
      sessionId,
      status: "processing",
      step: "running_analysis",
      outputs: {}
    });
  }
};