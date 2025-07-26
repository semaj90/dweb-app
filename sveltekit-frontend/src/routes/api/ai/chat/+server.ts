import { json } from "@sveltejs/kit";
import { ollamaService } from "$lib/services/ollama-service";

export const POST = async ({ request }) => {
  const startTime = Date.now();
  
  try {
    const {
      message,
      context,
      conversationId,
      model = "gemma3-legal",
      temperature = 0.1,
      maxTokens = 512,
      systemPrompt
    } = await request.json();
    
    if (!message || message.trim() === "") {
      return json({ error: "Message is required" }, { status: 400 });
    }

    // Use actual Ollama service instead of mock
    try {
      const response = await ollamaService.generate(message, {
        system: systemPrompt || "You are a specialized Legal AI Assistant powered by Gemma 3. You excel at contract analysis, legal research, and providing professional legal guidance.",
        temperature: temperature,
        maxTokens: maxTokens,
        topP: 0.8,
        topK: 20,
        repeatPenalty: 1.05
      });
      
      return json({ 
        response,
        model,
        conversationId: conversationId || `conv_${Date.now()}`,
        metadata: {
          provider: "ollama",
          confidence: 0.9,
          executionTime: Date.now() - startTime,
          fromCache: false,
        }
      });
    } catch (ollamaError) {
      // Fallback to mock response if Ollama fails
      console.warn("Ollama service unavailable, using fallback:", ollamaError);
      
      const mockResponse = generateMockResponse(message, context);
      return json({ 
        response: mockResponse,
        model: model + " (fallback)",
        conversationId: conversationId || `conv_${Date.now()}`,
        metadata: {
          provider: "fallback",
          confidence: 0.5,
          executionTime: Date.now() - startTime,
          fromCache: false,
          fallbackReason: "Ollama service unavailable"
        }
      });
    }
  } catch (error) {
    console.error("AI chat error:", error);
    return json({ 
      error: "Failed to process chat",
      details: error instanceof Error ? error.message : "Unknown error"
    }, { status: 500 });
  }
};

function generateMockResponse(message: string, context?: any[]): string {
  // Enhanced mock responses for legal queries
  const legalResponses = [
    `Regarding your question about "${message}" - This is a complex legal matter that requires careful analysis. As a Legal AI Assistant, I can help break down the key components and considerations involved. However, please note this is operating in fallback mode while connecting to the primary AI model.`,
    
    `Thank you for your legal inquiry: "${message}". In matters like this, it's essential to consider applicable statutes, case precedents, and jurisdiction-specific requirements. I'm currently operating with limited functionality - please ensure the Ollama service is running for full legal analysis capabilities.`,
    
    `Your question about "${message}" touches on important legal principles. While I can provide general guidance, please note that I'm currently in fallback mode. For comprehensive legal analysis, please verify that the Gemma3 Legal AI model is properly loaded and accessible.`,
    
    `I understand you're asking about "${message}". This appears to involve legal concepts that benefit from detailed analysis. Currently operating in emergency fallback mode - please check the Ollama service connection for access to the full Gemma3 Legal AI capabilities.`
  ];
  
  return legalResponses[Math.floor(Math.random() * legalResponses.length)];
}
