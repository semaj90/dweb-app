import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request }) => {
  try {
    const {
      content,
      type = "general",
      model = "gemma3-legal",
    } = await request.json();

    if (!content || content.trim().length === 0) {
      return json(
        { error: "No content provided to summarize" },
        { status: 400 },
      );
    }
    // Prepare the prompt based on content type
    let prompt = "";
    switch (type) {
      case "report":
        prompt = `Please provide a concise summary of this case report. Focus on key findings, evidence, and conclusions:

${content}

Summary:`;
        break;

      case "evidence":
        prompt = `Please analyze and summarize this evidence. Highlight important details, potential significance, and any notable patterns:

${content}

Analysis:`;
        break;

      case "poi":
        prompt = `Please provide a comprehensive profile summary for this person of interest. Focus on their background, involvement, motivations, and methods:

${content}

Profile Summary:`;
        break;

      default:
        prompt = `Please provide a clear and concise summary of the following content:

${content}

Summary:`;
    }
    // Call the local Ollama instance
    const startTime = Date.now();

    const response = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.3, // Lower temperature for more focused summaries
          top_p: 0.9,
          top_k: 40,
          max_tokens: 500, // Reasonable length for summaries
        },
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Ollama API error: ${response.status} ${response.statusText}`,
      );
    }
    const data = await response.json();
    const processingTime = Date.now() - startTime;

    if (!data.response) {
      throw new Error("No response from AI model");
    }
    return json({
      summary: data.response.trim(),
      model: model,
      processingTime: processingTime,
      type: type,
    });
  } catch (error) {
    console.error("AI summarization error:", error);

    // Check if it's an Ollama connection error
    if (error instanceof Error && error.message.includes("fetch")) {
      return json(
        {
          error:
            "Unable to connect to local AI service. Please ensure Ollama is running.",
        },
        { status: 503 },
      );
    }
    return json(
      {
        error:
          error instanceof Error ? error.message : "Failed to generate summary",
      },
      { status: 500 },
    );
  }
};
