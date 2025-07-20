import { ollamaChatStream } from "$lib/services/ollamaChatStream";
import { json } from "@sveltejs/kit";

export const POST = async ({ request }) => {
  try {
    const {
      message,
      context,
      conversationId,
      model,
      temperature,
      maxTokens,
      systemPrompt,
    } = await request.json();
    if (!message) {
      return json({ error: "Message is required" }, { status: 400 });
    }
    // Stream chat response from local LLM (Ollama)
    const stream = await ollamaChatStream({
      message,
      context,
      conversationId,
      model: model || "llama3",
      temperature: temperature ?? 0.7,
      maxTokens: maxTokens ?? 1000,
      systemPrompt,
    });
    // For now, collect all chunks and return as a single response (upgrade to streaming later)
    let fullText = "";
    for await (const chunk of stream) {
      fullText += chunk.text || "";
    }
    return json({ response: fullText });
  } catch (error) {
    console.error("Ollama chat error:", error);
    return json({ error: "Failed to process chat" }, { status: 500 });
  }
};
