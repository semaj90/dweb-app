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

    // Mock AI response for now - replace with actual Ollama integration
    const mockResponse = generateMockResponse(message, context);
    
    return json({ 
      response: mockResponse,
      model: model || "gemma3:7b",
      conversationId: conversationId || `conv_${Date.now()}`,
      metadata: {
        provider: "local",
        confidence: 0.85,
        executionTime: Math.random() * 1000 + 500,
        fromCache: false,
      }
    });
  } catch (error) {
    console.error("AI chat error:", error);
    return json({ 
      error: "Failed to process chat",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};

function generateMockResponse(message: string, context?: any[]): string {
  // Simple mock responses for testing
  const responses = [
    `I understand you're asking about: "${message}". Based on the legal context provided, I can help analyze this situation. However, please note that this is a mock response for testing purposes.`,
    `Thank you for your question about "${message}". In legal matters like this, it's important to consider all relevant evidence and applicable statutes. This is a placeholder response while the AI system is being configured.`,
    `Regarding your inquiry: "${message}" - This appears to be a legal matter that requires careful analysis. I'm currently operating in test mode, so please consult with legal professionals for actual advice.`,
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}
