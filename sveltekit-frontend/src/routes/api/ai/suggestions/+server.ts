import { json } from "@sveltejs/kit";
import type { RequestEvent } from "@sveltejs/kit";

export async function POST({ request }: RequestEvent) {
  try {
    const data = await request.json();

    if (!data.content) {
      return json({ error: "Content is required" }, { status: 400 });
    }
    // Mock AI suggestions for now - integrate with local LLM later
    const suggestions = generateMockSuggestions(data.content, data.reportType);

    return json({
      suggestions,
      model: "mock-ai-v1",
      confidence: 0.8,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error generating AI suggestions:", error);
    return json({ error: "Failed to generate suggestions" }, { status: 500 });
  }
}
function generateMockSuggestions(
  content: string,
  reportType: string = "prosecution_memo",
): string[] {
  const suggestions: string[] = [];

  // Analyze content and generate contextual suggestions
  const contentLower = content.toLowerCase();

  if (reportType === "prosecution_memo") {
    if (
      contentLower.includes("suspect") ||
      contentLower.includes("defendant")
    ) {
      suggestions.push(
        "Consider including the defendant's criminal history and prior convictions to establish a pattern of behavior.",
      );
    }
    if (contentLower.includes("evidence")) {
      suggestions.push(
        "Ensure all evidence is properly authenticated and meets the requirements for admissibility under the Rules of Evidence.",
      );
    }
    if (contentLower.includes("witness")) {
      suggestions.push(
        "Evaluate witness credibility and consider any potential impeachment issues that may arise during trial.",
      );
    }
    if (contentLower.includes("charge") || contentLower.includes("count")) {
      suggestions.push(
        "Review the elements of each charge to ensure sufficient evidence exists to prove guilt beyond a reasonable doubt.",
      );
    }
  } else if (reportType === "case_brief") {
    suggestions.push(
      "Summarize the key legal issues and applicable statutes relevant to this case.",
    );
    suggestions.push(
      "Analyze any potential constitutional issues or procedural defenses.",
    );
  } else if (reportType === "evidence_summary") {
    suggestions.push(
      "Organize evidence chronologically and by relevance to each charge.",
    );
    suggestions.push(
      "Note any chain of custody issues that need to be addressed.",
    );
  }
  // Generic suggestions based on content analysis
  if (content.length < 200) {
    suggestions.push(
      "Consider expanding this section with more detailed analysis and supporting evidence.",
    );
  }
  if (!contentLower.includes("statute") && !contentLower.includes("law")) {
    suggestions.push(
      "Reference applicable statutes and legal precedents to strengthen your argument.",
    );
  }
  if (!contentLower.includes("conclusion") && content.length > 500) {
    suggestions.push(
      "Consider adding a conclusion section to summarize your key findings and recommendations.",
    );
  }
  // Return up to 3 most relevant suggestions
  return suggestions.slice(0, 3);
}
