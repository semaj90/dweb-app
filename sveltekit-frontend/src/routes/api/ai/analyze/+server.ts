import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { 
      content, 
      caseId, 
      reportId, 
      analysisType = 'comprehensive',
      model = 'gemma3-legal'
    } = await request.json();

    if (!content || content.trim().length === 0) {
      return json(
        { error: 'No content provided for analysis' },
        { status: 400 }
      );
    }

    // Prepare analysis prompt based on type
    let prompt = '';
    switch (analysisType) {
      case 'legal':
        prompt = `Please provide a comprehensive legal analysis of this report. Identify:
1. Key legal issues
2. Relevant laws and regulations
3. Potential violations or compliance issues
4. Legal risks and implications
5. Recommended legal actions

Content to analyze:
${content}

Legal Analysis:`;
        break;

      case 'evidence':
        prompt = `Please analyze this content for evidentiary value. Identify:
1. Types of evidence present
2. Strength and reliability of evidence
3. Gaps in evidence
4. Potential challenges to admissibility
5. Additional evidence needed

Content to analyze:
${content}

Evidence Analysis:`;
        break;

      case 'investigation':
        prompt = `Please provide an investigative analysis of this content. Focus on:
1. Key findings and patterns
2. Investigative leads to pursue
3. Potential witnesses or sources
4. Timeline and sequence of events
5. Recommended next steps

Content to analyze:
${content}

Investigation Analysis:`;
        break;

      default: // comprehensive
        prompt = `Please provide a comprehensive analysis of this legal report. Include:
1. Key points and findings
2. Legal implications
3. Evidentiary considerations
4. Investigative insights
5. Risk assessment
6. Actionable recommendations

Content to analyze:
${content}

Comprehensive Analysis:`;
    }

    // Call the local Ollama instance
    const startTime = Date.now();

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.4, // Balanced temperature for analytical content
          top_p: 0.9,
          top_k: 40,
          max_tokens: 1000, // Longer responses for detailed analysis
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const processingTime = Date.now() - startTime;

    if (!data.response) {
      throw new Error('No response from AI model');
    }

    // Parse the analysis response to extract structured information
    const analysisText = data.response.trim();
    const parsedAnalysis = parseAnalysisResponse(analysisText, analysisType);

    return json({
      success: true,
      analysis: parsedAnalysis,
      rawAnalysis: analysisText,
      analysisType,
      model: model,
      processingTime: processingTime,
      metadata: {
        analyzedAt: new Date().toISOString(),
        caseId,
        reportId,
        contentLength: content.length,
        confidence: calculateConfidence(analysisText)
      }
    });

  } catch (error) {
    console.error('AI analysis error:', error);

    // Check if it's an Ollama connection error
    if (error instanceof Error && error.message.includes('fetch')) {
      return json(
        {
          error: 'Unable to connect to local AI service. Please ensure Ollama is running.',
        },
        { status: 503 }
      );
    }

    return json(
      {
        error: error instanceof Error ? error.message : 'Failed to generate analysis',
      },
      { status: 500 }
    );
  }
};

function parseAnalysisResponse(analysisText: string, analysisType: string) {
  // Extract key points, recommendations, and other structured data
  const sections = analysisText.split(/\d+\.\s+|\n\n+/);
  
  const keyPoints: string[] = [];
  const recommendations: string[] = [];
  const risks: string[] = [];
  const findings: string[] = [];

  sections.forEach(section => {
    const lowerSection = section.toLowerCase();
    const trimmedSection = section.trim();
    
    if (!trimmedSection) return;

    // Categorize content based on keywords
    if (lowerSection.includes('recommend') || lowerSection.includes('should') || lowerSection.includes('suggest')) {
      recommendations.push(trimmedSection);
    } else if (lowerSection.includes('risk') || lowerSection.includes('concern') || lowerSection.includes('challenge')) {
      risks.push(trimmedSection);
    } else if (lowerSection.includes('finding') || lowerSection.includes('evidence') || lowerSection.includes('indicates')) {
      findings.push(trimmedSection);
    } else if (trimmedSection.length > 20) {
      keyPoints.push(trimmedSection);
    }
  });

  return {
    keyPoints: keyPoints.slice(0, 5),
    recommendations: recommendations.slice(0, 5),
    risks: risks.slice(0, 3),
    findings: findings.slice(0, 5),
    confidence: calculateConfidence(analysisText)
  };
}

function calculateConfidence(text: string): number {
  // Simple confidence calculation based on text characteristics
  let confidence = 0.7; // Base confidence

  // Check for specific legal terms and structure
  const legalTerms = [
    'statute', 'regulation', 'precedent', 'evidence', 'testimony',
    'liability', 'violation', 'compliance', 'analysis', 'finding'
  ];

  const lowerText = text.toLowerCase();
  const termCount = legalTerms.filter(term => lowerText.includes(term)).length;
  
  // Adjust confidence based on legal terminology usage
  confidence += (termCount / legalTerms.length) * 0.2;

  // Check for structured analysis (numbered points, clear sections)
  if (text.includes('1.') || text.includes('Key') || text.includes('Recommendation')) {
    confidence += 0.1;
  }

  // Ensure confidence is within reasonable bounds
  return Math.min(0.95, Math.max(0.6, confidence));
}