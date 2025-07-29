import { json } from "@sveltejs/kit";
import { exec } from "child_process";
import { promisify } from "util";
import { writeFile, readFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import type { RequestHandler } from "./$types";

const execAsync = promisify(exec);

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      caseId,
      evidenceContent,
      evidenceTitle,
      evidenceType = "document",
    } = await request.json();

    if (!caseId || !evidenceContent || !evidenceTitle) {
      return json(
        {
          error:
            "Missing required fields: caseId, evidenceContent, evidenceTitle",
        },
        { status: 400 }
      );
    }

    // Create temp directory if it doesn't exist
    const tempDir = "./temp";
    if (!existsSync(tempDir)) {
      await mkdir(tempDir, { recursive: true });
    }

    // Create evidence file
    const timestamp = Date.now();
    const evidenceFile = `${tempDir}/evidence_${caseId}_${timestamp}.txt`;
    await writeFile(evidenceFile, evidenceContent, "utf8");

    console.log(`Starting multi-agent analysis for case ${caseId}`);
    console.log(`Evidence file: ${evidenceFile}`);

    // Execute the multi-agent analysis script
    const scriptPath =
      process.platform === "win32"
        ? "./scripts/analyze-evidence.bat"
        : "./scripts/analyze-evidence.sh";

    const { stdout, stderr } = await execAsync(
      `${scriptPath} ${caseId} "${evidenceFile}"`
    );

    if (stderr) {
      console.warn("Analysis stderr:", stderr);
    }

    console.log("Analysis stdout:", stdout);

    // Read analysis results
    const analysisDir = `${tempDir}/analysis_${caseId}_${timestamp}`;

    let evidenceAnalysis = {};
    let personsData = {};
    let caseSynthesis = {};
    let neo4jUpdates = {};

    try {
      // Try to read each result file
      if (existsSync(`${analysisDir}/evidence_analysis.json`)) {
        const data = await readFile(
          `${analysisDir}/evidence_analysis.json`,
          "utf8"
        );
        evidenceAnalysis = JSON.parse(data);
      }

      if (existsSync(`${analysisDir}/persons_extracted.json`)) {
        const data = await readFile(
          `${analysisDir}/persons_extracted.json`,
          "utf8"
        );
        personsData = JSON.parse(data);
      }

      if (existsSync(`${analysisDir}/case_synthesis.json`)) {
        const data = await readFile(
          `${analysisDir}/case_synthesis.json`,
          "utf8"
        );
        caseSynthesis = JSON.parse(data);
      }

      if (existsSync(`${analysisDir}/neo4j_updates.json`)) {
        const data = await readFile(
          `${analysisDir}/neo4j_updates.json`,
          "utf8"
        );
        neo4jUpdates = JSON.parse(data);
      }
    } catch (parseError) {
      console.warn("Error parsing analysis results:", parseError);
    }

    // Compile final analysis result
    const analysisResult = {
      id: `analysis_${caseId}_${timestamp}`,
      caseId,
      evidenceAnalysis,
      personsData,
      caseSynthesis,
      neo4jUpdates,
      timestamp: new Date().toISOString(),
      // Remove confidence property (not in schema)
      metadata: {
        evidenceTitle,
        evidenceType,
        analysisDir,
        scriptOutput: stdout,
      },
    };

    return json({
      success: true,
      analysis: analysisResult,
      message: "Multi-agent analysis completed successfully",
    });
  } catch (error: any) {
    console.error("Multi-agent analysis error:", error);

    return json(
      {
        success: false,
        error: error.message,
        message: "Multi-agent analysis failed",
      },
      { status: 500 }
    );
  }
};

// GET endpoint to retrieve analysis results
export const GET: RequestHandler = async ({ url }) => {
  try {
    const caseId = url.searchParams.get("caseId");
    const analysisId = url.searchParams.get("analysisId");

    if (!caseId) {
      return json({ error: "Missing caseId parameter" }, { status: 400 });
    }

    // In production, this would query the database
    // For now, return empty array or mock data
    const analyses = [];

    return json({
      success: true,
      analyses,
      caseId,
    });
  } catch (error: any) {
    console.error("Error retrieving analyses:", error);

    return json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
};
