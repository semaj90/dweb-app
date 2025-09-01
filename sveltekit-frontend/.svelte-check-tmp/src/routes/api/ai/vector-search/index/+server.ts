
import { json } from "@sveltejs/kit";
import { vectorSearchService } from "$lib/services/vector-search";
import { legalDocuments as documents } from "$lib/server/db/schema-postgres";
import type { RequestHandler } from './$types';


// Real-time document indexing endpoint
export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      documentId,
      content,
      filename,
      caseId,
      documentType = "legal-document",
      generateSummary = true,
      extractKeywords = true,
    } = await request.json();

    if (!documentId || !content) {
      return json(
        {
          error: "documentId and content are required",
        },
        { status: 400 }
      );
    }

    // Validate document exists
    const existingDocument = await db
      .select()
      .from(documents)
      .where(eq(documents.id, documentId))
      .limit(1);

    if (existingDocument.length === 0) {
      return json(
        {
          error: "Document not found",
        },
        { status: 404 }
      );
    }

    // Generate AI-powered summary and keywords if requested
    let summary: string | undefined;
    let keywords: string[] | undefined;

    if (generateSummary || extractKeywords) {
      const analysisResult = await generateDocumentAnalysis(
        content,
        filename || "Unknown Document",
        generateSummary,
        extractKeywords
      );

      summary = analysisResult.summary;
      keywords = analysisResult.keywords;
    }

    // Index the document with vector embeddings
    await vectorSearchService.indexDocument(documentId, content, {
      filename,
      caseId,
      documentType,
      summary,
      keywords,
    });

    return json({
      success: true,
      documentId,
      message: "Document successfully indexed for vector search",
      metadata: {
        filename,
        caseId,
        documentType,
        contentLength: content.length,
        summary:
          summary?.substring(0, 100) +
          (summary && summary.length > 100 ? "..." : ""),
        keywordCount: keywords?.length || 0,
        indexedAt: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error("Document indexing error:", error);
    return json(
      {
        error: "Document indexing failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

// Batch indexing endpoint
export const PUT: RequestHandler = async ({ request }) => {
  try {
    const { documentIds, forceReindex = false } = await request.json();

    if (!Array.isArray(documentIds) || documentIds.length === 0) {
      return json(
        {
          error: "documentIds array is required",
        },
        { status: 400 }
      );
    }

    const results = [];
    let successCount = 0;
    let errorCount = 0;

    for (const documentId of documentIds) {
      try {
        // Get document content
        const document = await db
          .select()
          .from(documents)
          .where(eq(documents.id, documentId))
          .limit(1);

        if (document.length === 0) {
          results.push({
            documentId,
            status: "error",
            error: "Document not found",
          });
          errorCount++;
          continue;
        }

        const doc = document[0];

        // Skip if already indexed and not forcing reindex
        if (doc.embedding && !forceReindex) {
          results.push({
            documentId,
            status: "skipped",
            message: "Already indexed (use forceReindex=true to reindex)",
          });
          continue;
        }

        // Generate analysis
        const documentContent = doc.content || doc.fullText || "";
        const documentTitle = doc.title || "Unknown Document";

        const analysis = await generateDocumentAnalysis(
          documentContent,
          documentTitle,
          true,
          true
        );

        // Index the document
        await vectorSearchService.indexDocument(documentId, documentContent, {
          filename: documentTitle,
          caseId: doc.caseId,
          documentType: doc.documentType,
          summary: analysis.summary,
          keywords: analysis.keywords,
        });

        results.push({
          documentId,
          status: "success",
          title: documentTitle,
          summary: analysis.summary?.substring(0, 100) + "...",
        });
        successCount++;
      } catch (error: any) {
        results.push({
          documentId,
          status: "error",
          error: error instanceof Error ? error.message : "Unknown error",
        });
        errorCount++;
      }
    }

    return json({
      success: successCount > 0,
      summary: {
        total: documentIds.length,
        successful: successCount,
        errors: errorCount,
        skipped: documentIds.length - successCount - errorCount,
      },
      results,
      indexedAt: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error("Batch indexing error:", error);
    return json(
      {
        error: "Batch indexing failed",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
};

// Document analysis using local LLM (Ollama)
async function generateDocumentAnalysis(
  content: string,
  filename: string,
  generateSummary: boolean = true,
  extractKeywords: boolean = true
): Promise<{
  summary?: string;
  keywords?: string[];
}> {
  try {
    if (!generateSummary && !extractKeywords) {
      return {};
    }

    const prompt = `
Analyze this legal document: "${filename}"

Content:
${content.substring(0, 3000)}${content.length > 3000 ? "..." : ""}

Please provide:
${generateSummary ? "1. A concise summary (2-3 sentences)" : ""}
${extractKeywords ? "2. Key legal terms and concepts (5-10 keywords)" : ""}

Respond in JSON format:
{
  "summary": "${generateSummary ? "brief summary here" : "null"}",
  "keywords": ${extractKeywords ? '["keyword1", "keyword2", "keyword3"]' : "null"}
}
    `;

    // Call Ollama for analysis
    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "llama3.2",
        prompt,
        stream: false,
        options: {
          temperature: 0.3,
          top_p: 0.9,
        },
      }),
    });

    if (!response.ok) {
      console.warn("Ollama not available, skipping AI analysis");
      return {
        summary: generateSummary
          ? `Legal document: ${filename} (${content.length} characters)`
          : undefined,
        keywords: extractKeywords
          ? ["legal-document", "case-material"]
          : undefined,
      };
    }

    const data = await response.json();

    try {
      // Try to parse JSON response
      const analysisMatch = data.response.match(/\{[\s\S]*\}/);
      if (analysisMatch) {
        const analysis = JSON.parse(analysisMatch[0]);
        return {
          summary: generateSummary ? analysis.summary : undefined,
          keywords: extractKeywords ? analysis.keywords : undefined,
        };
      }
    } catch (parseError) {
      console.warn("Failed to parse AI analysis, using fallback");
    }

    // Fallback analysis
    return {
      summary: generateSummary
        ? `Legal document analysis: ${filename}`
        : undefined,
      keywords: extractKeywords
        ? ["legal-document", "case-evidence"]
        : undefined,
    };
  } catch (error: any) {
    console.error("Document analysis error:", error);
    return {
      summary: generateSummary ? `Document: ${filename}` : undefined,
      keywords: extractKeywords ? ["legal-document"] : undefined,
    };
  }
}
