
/**
 * XState Machine for AI Summary Reading and Analysis
 * Handles evidence reports, document summaries, and synthesis workflows
 */

import { createMachine, assign, fromPromise } from "xstate";

export interface AISummaryContext {
  // Document/Evidence data
  documentId: string | null;
  caseId: string | null;
  documentType: "evidence" | "report" | "contract" | "case_law" | "general";

  // Summary data
  originalContent: string;
  summary: string | null;
  keyInsights: string[];
  confidence: number;

  // AI Mix Synthesis data (added for comprehensive integration)
  llmOutput: {
    content: string;
    model: string;
    confidence: number;
    tokens: number;
    processingTime: number;
  } | null;
  ragOutput: {
    relevantDocs: any[];
    contextSummary: string;
    searchMetrics: any;
  } | null;
  userActivity: {
    recentQueries: string[];
    preferredTopics: string[];
    recommendations: string[];
  } | null;

  // Synthesis progress tracking
  synthesisPipeline: {
    llmComplete: boolean;
    ragComplete: boolean;
    userActivityComplete: boolean;
    fusejsComplete: boolean;
    finalSynthesisComplete: boolean;
  };

  // Reading state
  currentSection: number;
  sections: SummarySection[];
  readingSpeed: number; // WPM
  estimatedReadTime: number; // minutes

  // Analysis state
  analysisResults: AnalysisResult[];
  synthesisData: SynthesisData | null;

  // UI state
  error: string | null;
  loading: boolean;
  isPlaying: boolean;
  progress: number; // 0-100

  // User preferences
  voiceEnabled: boolean;
  highlightMode: "key_points" | "entities" | "legal_terms" | "none";
  readingMode: "sequential" | "insight_focused" | "summary_only";
}

export interface SummarySection {
  id: string;
  title: string;
  content: string;
  type:
    | "executive_summary"
    | "key_findings"
    | "evidence_analysis"
    | "recommendations"
    | "legal_implications";
  importance: "critical" | "high" | "medium" | "low";
  entities: Entity[];
  wordCount: number;
}

export interface Entity {
  text: string;
  type:
    | "person"
    | "organization"
    | "location"
    | "date"
    | "legal_term"
    | "evidence_id";
  confidence: number;
  context?: string;
}

export interface AnalysisResult {
  type:
    | "relevance"
    | "credibility"
    | "legal_significance"
    | "evidence_strength";
  score: number;
  explanation: string;
  recommendations: string[];
}

export interface SynthesisData {
  mainThemes: string[];
  contradictions: string[];
  supportingEvidence: string[];
  gaps: string[];
  legalImplications: string[];
  nextSteps: string[];
}

export type AISummaryEvent =
  | { type: "LOAD_DOCUMENT"; documentId: string; caseId?: string }
  | {
      type: "GENERATE_SUMMARY";
      content: string;
      documentType: AISummaryContext["documentType"];
    }
  | { type: "START_READING" }
  | { type: "PAUSE_READING" }
  | { type: "RESUME_READING" }
  | { type: "STOP_READING" }
  | { type: "NEXT_SECTION" }
  | { type: "PREVIOUS_SECTION" }
  | { type: "JUMP_TO_SECTION"; sectionIndex: number }
  | { type: "ANALYZE_DOCUMENT" }
  | { type: "SYNTHESIZE_INSIGHTS" }
  | { type: "UPDATE_PROGRESS"; progress: number }
  | { type: "UPDATE_PREFERENCES"; preferences: Partial<AISummaryContext> }
  | { type: "RETRY" }
  | { type: "RESET" };

const initialContext: AISummaryContext = {
  documentId: null,
  caseId: null,
  documentType: "general",
  originalContent: "",
  summary: null,
  keyInsights: [],
  confidence: 0,
  
  // AI Mix Synthesis data
  llmOutput: null,
  ragOutput: null,
  userActivity: null,
  
  // Synthesis progress tracking
  synthesisPipeline: {
    llmComplete: false,
    ragComplete: false,
    userActivityComplete: false,
    fusejsComplete: false,
    finalSynthesisComplete: false,
  },
  
  currentSection: 0,
  sections: [],
  readingSpeed: 200, // Average reading speed
  estimatedReadTime: 0,
  analysisResults: [],
  synthesisData: null,
  error: null,
  loading: false,
  isPlaying: false,
  progress: 0,
  voiceEnabled: false,
  highlightMode: "key_points",
  readingMode: "sequential",
};

export const aiSummaryMachine = createMachine({
  types: {} as {
    context: AISummaryContext;
    events: AISummaryEvent;
  },
  id: "aiSummaryMachine",
  initial: "idle",
  context: initialContext,
  states: {
      idle: {
        on: {
          LOAD_DOCUMENT: {
            target: "loading",
            actions: assign({
              documentId: ({
                event,
              }: {
                event: Extract<AISummaryEvent, { type: "LOAD_DOCUMENT" }>;
              }) => event.documentId,
              caseId: ({
                event,
              }: {
                event: Extract<AISummaryEvent, { type: "LOAD_DOCUMENT" }>;
              }) => event.caseId || null,
              loading: true,
              error: null,
            }),
          },
          GENERATE_SUMMARY: {
            target: "generating",
            actions: assign({
              originalContent: ({
                event,
              }: {
                event: Extract<AISummaryEvent, { type: "GENERATE_SUMMARY" }>;
              }) => event.content,
              documentType: ({
                event,
              }: {
                event: Extract<AISummaryEvent, { type: "GENERATE_SUMMARY" }>;
              }) => event.documentType,
              loading: true,
              error: null,
            }),
          },
          UPDATE_PREFERENCES: {
            actions: assign(
              ({
                context,
                event,
              }: {
                context: AISummaryContext;
                event: Extract<AISummaryEvent, { type: "UPDATE_PREFERENCES" }>;
              }) => ({
                ...context,
                ...event.preferences,
              }),
            ),
          },
        },
      },

      loading: {
        invoke: {
          src: "loadDocument",
          onDone: {
            target: "loaded",
            actions: assign({
              originalContent: ({ event }) => event.output.content,
              documentType: ({ event }) => event.output.type,
              loading: false,
            }),
          },
          onError: {
            target: "error",
            actions: assign({
              error: ({ event }) =>
                (event.error as Error)?.message || "Failed to load document",
              loading: false,
            }),
          },
        },
        on: {
          RESET: "idle",
        },
      },

      generating: {
        invoke: {
          src: "generateSummary",
          onDone: {
            target: "ready",
            actions: assign({
              summary: ({ event }) => event.output.summary,
              sections: ({ event }) => event.output.sections,
              keyInsights: ({ event }) => event.output.insights,
              confidence: ({ event }) => event.output.confidence,
              estimatedReadTime: ({ event, context }) =>
                Math.ceil(event.output.wordCount / context.readingSpeed),
              loading: false,
            }),
          },
          onError: {
            target: "error",
            actions: assign({
              error: ({ event }) =>
                (event.error as Error)?.message || "Failed to generate summary",
              loading: false,
            }),
          },
        },
        on: {
          RESET: "idle",
        },
      },

      loaded: {
        on: {
          GENERATE_SUMMARY: {
            target: "generating",
            actions: assign({
              loading: true,
              error: null,
            }),
          },
          RESET: "idle",
        },
      },

      ready: {
        initial: "paused",
        states: {
          paused: {
            on: {
              START_READING: "reading",
              ANALYZE_DOCUMENT: {
                target: "#aiSummaryMachine.analyzing",
                actions: assign({ loading: true }),
              },
              SYNTHESIZE_INSIGHTS: {
                target: "#aiSummaryMachine.synthesizing",
                actions: assign({ loading: true }),
              },
            },
          },
          reading: {
            initial: "playing",
            entry: assign({ isPlaying: true }),
            exit: assign({ isPlaying: false }),
            states: {
              playing: {
                invoke: {
                  src: "readingProgress",
                  id: "readingProgress",
                },
                on: {
                  PAUSE_READING: "paused_mid_read",
                  UPDATE_PROGRESS: {
                    actions: assign({
                      progress: ({ event }) => event.progress,
                    }),
                  },
                },
              },
              paused_mid_read: {
                on: {
                  RESUME_READING: "playing",
                  STOP_READING: "#aiSummaryMachine.ready.paused",
                },
              },
            },
            on: {
              NEXT_SECTION: {
                actions: assign({
                  currentSection: ({ context }) =>
                    Math.min(
                      context.currentSection + 1,
                      context.sections.length - 1,
                    ),
                  progress: ({ context }) =>
                    ((context.currentSection + 1) / context.sections.length) *
                    100,
                }),
              },
              PREVIOUS_SECTION: {
                actions: assign({
                  currentSection: ({ context }) =>
                    Math.max(context.currentSection - 1, 0),
                  progress: ({ context }) =>
                    (context.currentSection / context.sections.length) * 100,
                }),
              },
              JUMP_TO_SECTION: {
                actions: assign({
                  currentSection: ({ event }) => event.sectionIndex,
                  progress: ({ event, context }) =>
                    (event.sectionIndex / context.sections.length) * 100,
                }),
              },
              STOP_READING: "paused",
            },
          },
        },
        on: {
          UPDATE_PREFERENCES: {
            actions: assign(({ context, event }) => ({
              ...context,
              ...event.preferences,
            })),
          },
          RESET: "#aiSummaryMachine.idle",
        },
      },

      analyzing: {
        invoke: {
          src: "analyzeDocument",
          onDone: {
            target: "ready",
            actions: assign({
              analysisResults: ({ event }) => event.output.results,
              loading: false,
            }),
          },
          onError: {
            target: "error",
            actions: assign({
              error: ({ event }) => (event.error as Error)?.message || "Analysis failed",
              loading: false,
            }),
          },
        },
      },

      synthesizing: {
        invoke: {
          src: "synthesizeInsights",
          onDone: {
            target: "ready",
            actions: assign({
              synthesisData: ({ event }) => event.output.synthesis,
              loading: false,
            }),
          },
          onError: {
            target: "error",
            actions: assign({
              error: ({ event }) => (event.error as Error)?.message || "Synthesis failed",
              loading: false,
            }),
          },
        },
      },

      error: {
        on: {
          RETRY: {
            target: "idle",
            actions: assign({
              error: null,
              loading: false,
            }),
          },
          RESET: "idle",
        },
      },
    },
  },
  {
    actors: {
      loadDocument: fromPromise(async ({ input }: { input: AISummaryContext }) => {
        // Mock implementation - would call actual API
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              content: `Evidence Report #${input.documentId}\n\nThis is a comprehensive analysis of the evidence collected in case ${input.caseId}. The findings indicate significant legal implications that require careful consideration...`,
              type: "evidence" as const,
            });
          }, 1000);
        });
      }),

      generateSummary: fromPromise(async ({ input }: { input: AISummaryContext }) => {
        // Mock implementation - would call RAG/AI service
        return new Promise((resolve) => {
          setTimeout(() => {
            const sections: SummarySection[] = [
              {
                id: "exec-summary",
                title: "Executive Summary",
                content:
                  "This evidence report provides a comprehensive analysis of digital forensics findings in the case. Key evidence points to significant security violations and potential criminal activity.",
                type: "executive_summary",
                importance: "critical",
                entities: [
                  {
                    text: "digital forensics",
                    type: "legal_term",
                    confidence: 0.95,
                  },
                  {
                    text: "security violations",
                    type: "legal_term",
                    confidence: 0.92,
                  },
                ],
                wordCount: 120,
              },
              {
                id: "key-findings",
                title: "Key Findings",
                content:
                  "Analysis of the digital evidence reveals unauthorized access attempts, data exfiltration, and potential insider threats. Timeline analysis shows coordinated activities over a 6-month period.",
                type: "key_findings",
                importance: "critical",
                entities: [
                  {
                    text: "unauthorized access",
                    type: "legal_term",
                    confidence: 0.98,
                  },
                  {
                    text: "data exfiltration",
                    type: "legal_term",
                    confidence: 0.96,
                  },
                  { text: "6-month period", type: "date", confidence: 0.89 },
                ],
                wordCount: 180,
              },
              {
                id: "legal-implications",
                title: "Legal Implications",
                content:
                  "The evidence supports charges under the Computer Fraud and Abuse Act (CFAA) and state data protection laws. Recommended prosecution strategy includes focusing on the financial impact and systematic nature of the violations.",
                type: "legal_implications",
                importance: "high",
                entities: [
                  {
                    text: "Computer Fraud and Abuse Act",
                    type: "legal_term",
                    confidence: 0.99,
                  },
                  {
                    text: "data protection laws",
                    type: "legal_term",
                    confidence: 0.94,
                  },
                ],
                wordCount: 150,
              },
            ];

            resolve({
              summary:
                "Comprehensive evidence analysis revealing systematic security violations with strong legal basis for prosecution under federal cybercrime statutes.",
              sections,
              insights: [
                "Strong digital forensics evidence chain",
                "Clear CFAA violation patterns",
                "Financial impact quantifiable",
                "Insider threat indicators present",
              ],
              confidence: 0.92,
              wordCount: sections.reduce(
                (total, section) => total + section.wordCount,
                0,
              ),
            });
          }, 2000);
        });
      }),

      analyzeDocument: fromPromise(async ({ input }: { input: AISummaryContext }) => {
        // Mock implementation - would call analysis service
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              results: [
                {
                  type: "relevance" as const,
                  score: 0.94,
                  explanation:
                    "Highly relevant to the case with direct evidence of alleged violations",
                  recommendations: [
                    "Prioritize in evidence presentation",
                    "Prepare expert testimony",
                  ],
                },
                {
                  type: "credibility" as const,
                  score: 0.89,
                  explanation:
                    "Strong chain of custody and forensic methodology",
                  recommendations: [
                    "Verify forensic tool calibration",
                    "Document examiner credentials",
                  ],
                },
                {
                  type: "legal_significance" as const,
                  score: 0.96,
                  explanation:
                    "Critical evidence for establishing intent and systematic violations",
                  recommendations: [
                    "Central to prosecution strategy",
                    "Prepare for technical challenges",
                  ],
                },
              ],
            });
          }, 1500);
        });
      }),

      synthesizeInsights: fromPromise(async ({ input }: { input: AISummaryContext }) => {
        // Mock implementation - would call synthesis service
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({
              synthesis: {
                mainThemes: [
                  "Systematic security violations over extended period",
                  "Clear evidence of unauthorized access and data theft",
                  "Strong basis for federal cybercrime prosecution",
                ],
                contradictions: [
                  "Timeline discrepancies require clarification",
                  "Access log gaps during maintenance windows",
                ],
                supportingEvidence: [
                  "Digital forensics findings",
                  "Network traffic analysis",
                  "System log correlations",
                  "Financial impact assessment",
                ],
                gaps: [
                  "Need additional witness testimony on intent",
                  "Require expert analysis of technical methods",
                  "Missing financial records for full impact assessment",
                ],
                legalImplications: [
                  "Strong CFAA violation case",
                  "Potential RICO implications if organized",
                  "Civil remedies available alongside criminal charges",
                ],
                nextSteps: [
                  "Prepare technical expert testimony",
                  "Conduct additional financial analysis",
                  "Coordinate with cybercrime prosecutors",
                  "Prepare jury-friendly technical explanations",
                ],
              },
            });
          }, 2000);
        });
      }),

      readingProgress: fromPromise(async ({ input }: { input: AISummaryContext }) => {
        return new Promise((resolve) => {
          const interval = setInterval(() => {
            // Simulate reading progress - resolve after a timeout
            resolve({ progress: 100 });
          }, 1000);
          
          setTimeout(() => {
            clearInterval(interval);
            resolve({ progress: 100 });
          }, 5000);
        });
      }),
    },
  },
);

export default aiSummaryMachine;
