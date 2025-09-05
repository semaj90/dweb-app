<script lang="ts">
  interface Props {
    caseId: string
    documents: CaseDocument[] ;
    evidenceReports: EvidenceReport[] ;
  }
  let {
    caseId,
    documents = [],
    evidenceReports = []
  } = $props();



  import { useMachine } from '@xstate/svelte';
  import { createMachine, assign } from 'xstate';
  import AISummaryReader from './AISummaryReader.svelte';
  import EvidenceReportSummary from './EvidenceReportSummary.svelte';
  import { 
    FileText, 
    Brain, 
    GitMerge, 
    Scale, 
    Target, 
    CheckCircle, 
    AlertTriangle,
    Clock,
    Users,
    Download,
    Play,
    Pause
  } from 'lucide-svelte';
  import { fly, fade } from 'svelte/transition';

      
  interface CaseDocument {
    id: string
    title: string
    type: 'evidence' | 'report' | 'witness_statement' | 'expert_testimony' | 'legal_brief';
    content: string
    metadata: {
      dateCreated: string
      author: string
      relevanceScore: number
    };
  }

  interface EvidenceReport {
    id: string
    title: string
    type: string
    status: string
    priority: string
    createdAt: string
    updatedAt: string
    analyst: any
    evidence: any
    methodology: any
    findings: any
    legalImplications: any
    attachments: any[];
  }

  interface SynthesisContext {
    caseId: string
    documents: CaseDocument[];
    evidenceReports: EvidenceReport[];
    selectedItems: string[];
    synthesisMode: 'chronological' | 'thematic' | 'evidence_strength' | 'legal_strategy';
    synthesisResult: CaseSynthesis | null;
    progressStage: 'selecting' | 'analyzing' | 'synthesizing' | 'reviewing' | 'complete';
    error: string | null;
    loading: boolean
  }

  interface CaseSynthesis {
    executiveSummary: string
    timeline: TimelineEvent[];
    strengthAssessment: StrengthAssessment
    legalStrategy: LegalStrategy
    riskAnalysis: RiskAnalysis
    recommendations: Recommendation[];
    gaps: string[];
    nextSteps: string[];
  }

  interface TimelineEvent {
    date: string
    event: string
    sources: string[];
    significance: 'critical' | 'high' | 'medium' | 'low';
  }

  interface StrengthAssessment {
    overall: number
    evidenceQuality: number
    legalBasis: number
    witnessCredibility: number
    expertOpinions: number
    areas: {
      name: string
      score: number
      details: string
    }[];
  }

  interface LegalStrategy {
    primaryCharges: string[];
    supportingEvidence: string[];
    potentialDefenses: string[];
    prosecutionApproach: string
    keyArguments: string[];
  }

  interface RiskAnalysis {
    challengePoints: {
      issue: string
      likelihood: number
      impact: number
      mitigation: string
    }[];
    overallRisk: number
  }

  interface Recommendation {
    priority: 'immediate' | 'high' | 'medium' | 'low';
    category: 'evidence' | 'legal' | 'procedural' | 'strategic';
    action: string
    rationale: string
    timeline: string
  }

  const synthesisMachine = createMachine<SynthesisContext>({
    id: 'synthesis',
    initial: 'idle',
    context: {
      caseId,
      documents,
      evidenceReports,
      selectedItems: [],
      synthesisMode: 'thematic',
      synthesisResult: null,
      progressStage: 'selecting',
      error: null,
      loading: false
    },
    states: {
      idle: {
        on: {
          SELECT_ITEMS: {
            actions: assign({
              selectedItems: ({ event }) => event.items,
              progressStage: 'analyzing'
            })
          },
          START_SYNTHESIS: {
            target: 'synthesizing',
            actions: assign({ loading: true, progressStage: 'synthesizing' })
          }
        }
      },
      synthesizing: {
        invoke: {
          src: 'performSynthesis',
          onDone: {
            target: 'complete',
            actions: assign({
              synthesisResult: ({ event }) => event.data,
              loading: false,
              progressStage: 'complete'
            })
          },
          onError: {
            target: 'error',
            actions: assign({
              error: ({ event }) => event.data.message,
              loading: false
            })
          }
        }
      },
      complete: {
        on: {
          RESTART: {
            target: 'idle',
            actions: assign({
              selectedItems: [],
              synthesisResult: null,
              progressStage: 'selecting',
              error: null
            })
          }
        }
      },
      error: {
        on: {
          RETRY: {
            target: 'synthesizing',
            actions: assign({ error: null, loading: true })
          },
          RESTART: {
            target: 'idle',
            actions: assign({
              selectedItems: [],
              synthesisResult: null,
              progressStage: 'selecting',
              error: null
            })
          }
        }
      }
    }
  }, {
    services: {
      performSynthesis: async (context) => {
        // Mock comprehensive synthesis
        return new Promise<CaseSynthesis>((resolve) => {
          setTimeout(() => {
            resolve({
              executiveSummary: "Comprehensive analysis of the case evidence reveals a strong foundation for prosecution with multiple corroborating sources. The digital forensics evidence provides clear proof of unauthorized access, supported by witness testimony and financial records showing systematic fraud over an 18-month period.",
              
              timeline: [
                {
                  date: "2023-01-15",
                  event: "First unauthorized access detected in system logs",
                  sources: ["Digital Forensics Report #001", "Server Log Analysis"],
                  significance: "high"
                },
                {
                  date: "2023-03-22",
                  event: "Large data transfer to external IP address",
                  sources: ["Network Traffic Analysis", "Digital Forensics Report #002"],
                  significance: "critical"
                },
                {
                  date: "2023-06-10",
                  event: "Witness reports suspicious behavior from suspect",
                  sources: ["Witness Statement - J. Smith", "Security Camera Footage"],
                  significance: "medium"
                },
                {
                  date: "2023-08-15",
                  event: "Financial irregularities discovered in company accounts",
                  sources: ["Financial Analysis Report", "Accounting Records"],
                  significance: "critical"
                }
              ],
              
              strengthAssessment: {
                overall: 0.85,
                evidenceQuality: 0.90,
                legalBasis: 0.88,
                witnessCredibility: 0.75,
                expertOpinions: 0.92,
                areas: [
                  {
                    name: "Digital Evidence",
                    score: 0.95,
                    details: "Excellent chain of custody, forensically sound acquisition methods, expert analysis"
                  },
                  {
                    name: "Financial Evidence",
                    score: 0.88,
                    details: "Clear paper trail, professional accounting analysis, quantifiable damages"
                  },
                  {
                    name: "Witness Testimony",
                    score: 0.72,
                    details: "Multiple corroborating witnesses, some credibility concerns to address"
                  }
                ]
              },
              
              legalStrategy: {
                primaryCharges: [
                  "Computer Fraud and Abuse Act (18 U.S.C. § 1030)",
                  "Wire Fraud (18 U.S.C. § 1343)",
                  "Money Laundering (18 U.S.C. § 1956)"
                ],
                supportingEvidence: [
                  "Digital forensics showing unauthorized access",
                  "Financial records demonstrating fraudulent transfers",
                  "Witness testimony establishing intent and knowledge"
                ],
                potentialDefenses: [
                  "Lack of intent to defraud",
                  "Authorization claims",
                  "Technical malfunction defense"
                ],
                prosecutionApproach: "Lead with strong digital evidence, support with financial analysis, use witness testimony to establish intent and pattern of behavior",
                keyArguments: [
                  "Systematic nature of the violations over extended period",
                  "Clear financial motive and quantifiable damages",
                  "Sophisticated methods indicating premeditation",
                  "Multiple independent sources of evidence"
                ]
              },
              
              riskAnalysis: {
                challengePoints: [
                  {
                    issue: "Technical complexity may confuse jury",
                    likelihood: 0.6,
                    impact: 0.7,
                    mitigation: "Prepare clear visual aids and expert testimony in plain language"
                  },
                  {
                    issue: "Defense may challenge digital evidence authenticity",
                    likelihood: 0.8,
                    impact: 0.8,
                    mitigation: "Ensure robust chain of custody documentation and expert certification"
                  },
                  {
                    issue: "Witness credibility concerns",
                    likelihood: 0.4,
                    impact: 0.6,
                    mitigation: "Prepare witnesses thoroughly and focus on corroborating physical evidence"
                  }
                ],
                overallRisk: 0.35
              },
              
              recommendations: [
                {
                  priority: "immediate",
                  category: "evidence",
                  action: "Conduct additional forensic analysis of backup systems",
                  rationale: "May reveal additional evidence of data destruction attempts",
                  timeline: "Within 2 weeks"
                },
                {
                  priority: "high",
                  category: "legal",
                  action: "Prepare technical expert for jury testimony",
                  rationale: "Complex digital evidence requires clear expert explanation",
                  timeline: "Before trial preparation"
                },
                {
                  priority: "medium",
                  category: "strategic",
                  action: "Consider plea negotiations based on cooperation",
                  rationale: "Defendant may provide information about broader criminal network",
                  timeline: "After initial evidence presentation"
                }
              ],
              
              gaps: [
                "Need additional witness interviews to establish motive",
                "Require expert analysis of encryption methods used",
                "Missing financial records from offshore accounts"
              ],
              
              nextSteps: [
                "Schedule expert depositions for digital forensics analysts",
                "Coordinate with financial crimes unit for additional investigation",
                "Prepare comprehensive trial presentation materials",
                "Conduct mock trial with focus group for jury reactions"
              ]
            });
          }, 3000);
        });
      }
    }
  });

  const { state, send } = useMachine(synthesisMachine);

  let selectedDocuments = new Set<string>();
  let selectedReports = new Set<string>();

  let allItems = $derived([])
    ...documents.map(d => ({ id: d.id, type: 'document', title: d.title, data: d })),
    ...evidenceReports.map(r => ({ id: r.id, type: 'report', title: r.title, data: r }))
  ];

  let selectedCount = $derived(selectedDocuments.size + selectedReports.size)

  function toggleSelection(id: string, type: 'document' | 'report') {
    if (type === 'document') {
      if (selectedDocuments.has(id)) {
        selectedDocuments.delete(id);
      } else {
        selectedDocuments.add(id);
      }
      selectedDocuments = new Set(selectedDocuments);
    } else {
      if (selectedReports.has(id)) {
        selectedReports.delete(id);
      } else {
        selectedReports.add(id);
      }
      selectedReports = new Set(selectedReports);
    }
  }

  function startSynthesis() {
    const items = [...selectedDocuments, ...selectedReports];
    send({ type: 'SELECT_ITEMS', items });
    send({ type: 'START_SYNTHESIS' });
  }

  function getScoreColor(score: number): string {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }

  function getPriorityColor(priority: string): string {
    switch (priority) {
      case 'immediate': return 'bg-red-100 text-red-800 border-red-200';
      case 'high': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'low': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  }

  function exportSynthesis() {
    if (!$state.context.synthesisResult) return;
    
    const synthesis = $state.context.synthesisResult;
    const content = `# Case Synthesis Report - ${caseId}

## Executive Summary
${synthesis.executiveSummary}

## Strength Assessment
- Overall: ${Math.round(synthesis.strengthAssessment.overall * 100)}%
- Evidence Quality: ${Math.round(synthesis.strengthAssessment.evidenceQuality * 100)}%
- Legal Basis: ${Math.round(synthesis.strengthAssessment.legalBasis * 100)}%

## Legal Strategy
### Primary Charges
${synthesis.legalStrategy.primaryCharges.map(charge => `- ${charge}`).join('\n')}

### Prosecution Approach
${synthesis.legalStrategy.prosecutionApproach}

## Recommendations
${synthesis.recommendations.map(rec => `### ${rec.priority.toUpperCase()} - ${rec.action}\n${rec.rationale}\n`).join('\n')}

## Next Steps
${synthesis.nextSteps.map(step => `- ${step}`).join('\n')}
`;

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `case-synthesis-${caseId}-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
</script>

<div class="case-synthesis-workflow max-w-7xl mx-auto space-y-6">
  <!-- Header -->
  <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-gray-900 mb-2">Case Synthesis Workflow</h1>
        <p class="text-gray-600">Case ID: {caseId} • Comprehensive analysis and strategic planning</p>
      </div>
      
      <div class="flex items-center gap-4">
        <div class="text-right text-sm text-gray-600">
          <div>{allItems.length} items available</div>
          <div>{selectedCount} items selected</div>
        </div>
        
        {#if $state.context.synthesisResult}
          <button
            onclick={exportSynthesis}
            class="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            <Download class="w-4 h-4" />
            Export
          </button>
        {/if}
      </div>
    </div>

    <!-- Progress Bar -->
    <div class="mt-6">
      <div class="flex items-center justify-between text-sm text-gray-600 mb-2">
        <span>Progress</span>
        <span class="capitalize">{$state.context.progressStage}</span>
      </div>
      <div class="w-full bg-gray-200 rounded-full h-2">
        <div 
          class="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style="width: {
            $state.context.progressStage === 'selecting' ? '25%' :
            $state.context.progressStage === 'analyzing' ? '50%' :
            $state.context.progressStage === 'synthesizing' ? '75%' :
            $state.context.progressStage === 'complete' ? '100%' : '0%'
          }"
        ></div>
      </div>
    </div>
  </div>

  {#if $state.matches('idle') || $state.context.progressStage === 'selecting'}
    <!-- Item Selection -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
      <h2 class="text-lg font-semibold text-gray-900 mb-4">Select Items for Synthesis</h2>
      
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Documents -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">Documents ({documents.length})</h3>
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each documents as doc}
              <label class="flex items-start gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  class="mt-1"
                  checked={selectedDocuments.has(doc.id)}
                  onchange={() => toggleSelection(doc.id, 'document')}
                />
                <div class="flex-1">
                  <div class="font-medium text-gray-900">{doc.title}</div>
                  <div class="text-sm text-gray-600">
                    {doc.type} • {doc.metadata.author} • {new Date(doc.metadata.dateCreated).toLocaleDateString()}
                  </div>
                  <div class="text-xs text-gray-500">
                    Relevance: {Math.round(doc.metadata.relevanceScore * 100)}%
                  </div>
                </div>
              </label>
            {/each}
          </div>
        </div>

        <!-- Evidence Reports -->
        <div>
          <h3 class="font-medium text-gray-900 mb-3">Evidence Reports ({evidenceReports.length})</h3>
          <div class="space-y-3 max-h-96 overflow-y-auto">
            {#each evidenceReports as report}
              <label class="flex items-start gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50">
                <input
                  type="checkbox"
                  class="mt-1"
                  checked={selectedReports.has(report.id)}
                  onchange={() => toggleSelection(report.id, 'report')}
                />
                <div class="flex-1">
                  <div class="font-medium text-gray-900">{report.title}</div>
                  <div class="text-sm text-gray-600">
                    {report.type} • {report.status} • {report.priority} priority
                  </div>
                  <div class="text-xs text-gray-500">
                    Updated: {new Date(report.updatedAt).toLocaleDateString()}
                  </div>
                </div>
              </label>
            {/each}
          </div>
        </div>
      </div>

      <div class="mt-6 flex items-center justify-between">
        <div class="text-sm text-gray-600">
          {selectedCount} items selected for synthesis
        </div>
        <button
          onclick={startSynthesis}
          disabled={selectedCount === 0}
          class="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <GitMerge class="w-4 h-4" />
          Start Synthesis
        </button>
      </div>
    </div>

  {:else if $state.matches('synthesizing')}
    <!-- Loading State -->
    <div class="bg-white border border-gray-200 rounded-lg shadow-sm p-12">
      <div class="text-center">
        <div class="w-16 h-16 mx-auto mb-4 relative">
          <div class="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
          <div class="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
        </div>
        <h3 class="text-lg font-semibold text-gray-900 mb-2">Synthesizing Case Analysis</h3>
        <p class="text-gray-600">Processing {selectedCount} items for comprehensive analysis...</p>
        <div class="mt-4 text-sm text-gray-500">
          This may take a few minutes depending on the complexity of your case.
        </div>
      </div>
    </div>

  {:else if $state.matches('error')}
    <!-- Error State -->
    <div class="bg-red-50 border border-red-200 rounded-lg p-6">
      <div class="flex items-center gap-3">
        <AlertTriangle class="w-6 h-6 text-red-600" />
        <div>
          <h3 class="font-semibold text-red-800">Synthesis Failed</h3>
          <p class="text-red-700">{$state.context.error}</p>
        </div>
      </div>
      <div class="mt-4 flex gap-3">
        <button
          onclick={() => send({ type: 'RETRY' })}
          class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
        >
          Retry Synthesis
        </button>
        <button
          onclick={() => send({ type: 'RESTART' })}
          class="px-4 py-2 border border-red-300 text-red-700 rounded-md hover:bg-red-50 transition-colors"
        >
          Start Over
        </button>
      </div>
    </div>

  {:else if $state.matches('complete') && $state.context.synthesisResult}
    <!-- Synthesis Results -->
    <div class="space-y-6" transitionfly={{ y: 20, duration: 300 }}>
      <!-- Executive Summary -->
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-blue-900 mb-4 flex items-center gap-2">
          <Brain class="w-6 h-6" />
          Executive Summary
        </h2>
        <p class="text-blue-800 leading-relaxed">{$state.context.synthesisResult.executiveSummary}</p>
      </div>

      <!-- Strength Assessment -->
      <div class="bg-white border border-gray-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
          <Target class="w-6 h-6" />
          Strength Assessment
        </h2>
        
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          <div class="text-center">
            <div class="text-2xl font-bold {getScoreColor($state.context.synthesisResult.strengthAssessment.overall)}">
              {Math.round($state.context.synthesisResult.strengthAssessment.overall * 100)}%
            </div>
            <div class="text-sm text-gray-600">Overall</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold {getScoreColor($state.context.synthesisResult.strengthAssessment.evidenceQuality)}">
              {Math.round($state.context.synthesisResult.strengthAssessment.evidenceQuality * 100)}%
            </div>
            <div class="text-sm text-gray-600">Evidence</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold {getScoreColor($state.context.synthesisResult.strengthAssessment.legalBasis)}">
              {Math.round($state.context.synthesisResult.strengthAssessment.legalBasis * 100)}%
            </div>
            <div class="text-sm text-gray-600">Legal Basis</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold {getScoreColor($state.context.synthesisResult.strengthAssessment.witnessCredibility)}">
              {Math.round($state.context.synthesisResult.strengthAssessment.witnessCredibility * 100)}%
            </div>
            <div class="text-sm text-gray-600">Witnesses</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-bold {getScoreColor($state.context.synthesisResult.strengthAssessment.expertOpinions)}">
              {Math.round($state.context.synthesisResult.strengthAssessment.expertOpinions * 100)}%
            </div>
            <div class="text-sm text-gray-600">Experts</div>
          </div>
        </div>

        <div class="space-y-4">
          {#each $state.context.synthesisResult.strengthAssessment.areas as area}
            <div class="border border-gray-200 rounded-lg p-4">
              <div class="flex items-center justify-between mb-2">
                <h4 class="font-medium text-gray-900">{area.name}</h4>
                <span class="text-lg font-semibold {getScoreColor(area.score)}">
                  {Math.round(area.score * 100)}%
                </span>
              </div>
              <p class="text-gray-700 text-sm">{area.details}</p>
            </div>
          {/each}
        </div>
      </div>

      <!-- Legal Strategy -->
      <div class="bg-white border border-gray-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
          <Scale class="w-6 h-6" />
          Legal Strategy
        </h2>
        
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 class="font-medium text-gray-900 mb-3">Primary Charges</h3>
            <ul class="space-y-2">
              {#each $state.context.synthesisResult.legalStrategy.primaryCharges as charge}
                <li class="flex items-start gap-2">
                  <CheckCircle class="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                  <span class="text-gray-700">{charge}</span>
                </li>
              {/each}
            </ul>

            <h3 class="font-medium text-gray-900 mb-3 mt-6">Supporting Evidence</h3>
            <ul class="space-y-2">
              {#each $state.context.synthesisResult.legalStrategy.supportingEvidence as evidence}
                <li class="flex items-start gap-2">
                  <FileText class="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <span class="text-gray-700">{evidence}</span>
                </li>
              {/each}
            </ul>
          </div>

          <div>
            <h3 class="font-medium text-gray-900 mb-3">Prosecution Approach</h3>
            <p class="text-gray-700 mb-6">{$state.context.synthesisResult.legalStrategy.prosecutionApproach}</p>

            <h3 class="font-medium text-gray-900 mb-3">Potential Defenses</h3>
            <ul class="space-y-2">
              {#each $state.context.synthesisResult.legalStrategy.potentialDefenses as defense}
                <li class="flex items-start gap-2">
                  <AlertTriangle class="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                  <span class="text-gray-700">{defense}</span>
                </li>
              {/each}
            </ul>
          </div>
        </div>
      </div>

      <!-- Timeline -->
      <div class="bg-white border border-gray-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
          <Clock class="w-6 h-6" />
          Case Timeline
        </h2>
        
        <div class="space-y-4">
          {#each $state.context.synthesisResult.timeline as event}
            <div class="flex gap-4">
              <div class="flex-shrink-0 w-24 text-sm text-gray-600">
                {new Date(event.date).toLocaleDateString()}
              </div>
              <div class="flex-shrink-0">
                <div class="w-4 h-4 rounded-full mt-1"
                     class:bg-red-500={event.significance === 'critical'}
                     class:bg-orange-500={event.significance === 'high'}
                     class:bg-yellow-500={event.significance === 'medium'}
                     class:bg-gray-500={event.significance === 'low'}
                ></div>
              </div>
              <div class="flex-1">
                <p class="font-medium text-gray-900">{event.event}</p>
                <div class="text-sm text-gray-600 mt-1">
                  Sources: {event.sources.join(', ')}
                </div>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Recommendations -->
      <div class="bg-white border border-gray-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-gray-900 mb-6 flex items-center gap-2">
          <Users class="w-6 h-6" />
          Recommendations
        </h2>
        
        <div class="space-y-4">
          {#each $state.context.synthesisResult.recommendations as rec}
            <div class="border border-gray-200 rounded-lg p-4">
              <div class="flex items-start justify-between mb-2">
                <h4 class="font-medium text-gray-900">{rec.action}</h4>
                <span class="px-2 py-1 border rounded-full text-xs font-medium {getPriorityColor(rec.priority)}">
                  {rec.priority.toUpperCase()}
                </span>
              </div>
              <p class="text-gray-700 text-sm mb-2">{rec.rationale}</p>
              <div class="flex items-center gap-4 text-xs text-gray-600">
                <span>Category: {rec.category}</span>
                <span>Timeline: {rec.timeline}</span>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Next Steps -->
      <div class="bg-green-50 border border-green-200 rounded-lg p-6">
        <h2 class="text-xl font-semibold text-green-900 mb-4">Next Steps</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          {#each $state.context.synthesisResult.nextSteps as step, index}
            <div class="flex items-start gap-3">
              <span class="flex items-center justify-center w-6 h-6 bg-green-600 text-white text-sm rounded-full flex-shrink-0">
                {index + 1}
              </span>
              <span class="text-green-800">{step}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="flex gap-4">
        <button
          onclick={() => send({ type: 'RESTART' })}
          class="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          New Synthesis
        </button>
        <button
          onclick={exportSynthesis}
          class="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          <Download class="w-4 h-4" />
          Export Report
        </button>
      </div>
    </div>
  {/if}
</div>