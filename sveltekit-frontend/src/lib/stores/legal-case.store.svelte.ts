
// Legal Case Store - Svelte 5 Runes Implementation

// Define types locally since they're not available
export interface LegalCase {
  id: string;
  title: string;
  caseNumber: string;
  description?: string;
  status: "active" | "pending" | "closed" | "archived";
  priority: "low" | "medium" | "high" | "critical";
  confidentialityLevel: number;
}

export interface LegalDocument {
  id: string;
  name: string;
  type: string;
}

export interface AIInsights {
  findings?: any[];
  riskAssessment?: {
    score: number;
    level: string;
  };
  complianceChecks?: any[];
}

export interface AuditLogEntry {
  id: string;
  type: string;
  entityType: string;
  entityId: string;
  userId: string;
  timestamp: Date;
  details?: any;
}

export interface User {
  id: string;
  clearanceLevel: number;
  role: string;
}

// Mock audit service
class LegalAuditService {
  async logAction(action: {
    type: string;
    entityType: string;
    entityId: string;
    userId: string;
    details?: any;
  }): Promise<void> {
    console.log('Audit action logged:', action);
  }
}

export function createLegalCaseStore() {
  // State using Svelte 5 runes
  const cases = $state<LegalCase[]>([]);
  const selectedCase = $state<LegalCase | null>(null);
  const aiInsights = $state<Record<string, AIInsights>>({});
  const auditLog = $state<AuditLogEntry[]>([]);
  const currentUser = $state<User | null>(null);
  const loading = $state({
    cases: false,
    analysis: false,
    documents: false,
  });

  // Derived state for filtered cases based on user clearance
  const filteredCases = $derived(
    !currentUser ? [] : cases.filter(
      (legalCase) =>
        legalCase.confidentialityLevel <= currentUser.clearanceLevel,
    )
  );

  // Derived state for case statistics
  const caseStats = $derived({
    total: filteredCases.length,
    active: filteredCases.filter((c) => c.status === "active").length,
    pending: filteredCases.filter((c) => c.status === "pending").length,
    closed: filteredCases.filter((c) => c.status === "closed").length,
    highPriority: filteredCases.filter((c) => c.priority === "high").length,
  });

  // Audit service instance
  const auditService = new LegalAuditService();

  // Actions
  async function loadCases(): Promise<any> {
    loading.cases = true;
    try {
      const response = await fetch("/api/cases");
      const data = await response.json();
      cases.splice(0, cases.length, ...data);

      await auditService.logAction({
        type: "CASES_LOADED",
        entityType: "CASE",
        entityId: "bulk",
        userId: currentUser?.id || "unknown",
        details: { count: data.length },
      });
    } catch (error: any) {
      console.error("Failed to load cases:", error);
      throw error;
    } finally {
      loading.cases = false;
    }
  }

  async function selectCase(legalCase: LegalCase): Promise<any> {
    (selectedCase as any) = legalCase;

    await auditService.logAction({
      type: "CASE_SELECTED",
      entityType: "CASE",
      entityId: legalCase.id,
      userId: currentUser?.id || "unknown",
    });
  }

  async function analyzeCase(caseId: string): Promise<void> {
    loading.analysis = true;

    try {
      // Log analysis request
      await auditService.logAction({
        type: "CASE_ANALYSIS_REQUESTED",
        entityType: "CASE",
        entityId: caseId,
        userId: currentUser?.id || "unknown",
      });

      const response = await fetch(`/api/cases/${caseId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const insights = await response.json();
      aiInsights[caseId] = insights;

      // Log successful analysis
      await auditService.logAction({
        type: "CASE_ANALYSIS_COMPLETED",
        entityType: "CASE",
        entityId: caseId,
        userId: currentUser?.id || "unknown",
        details: {
          insightCount: insights.findings?.length || 0,
          riskScore: insights.riskAssessment?.score,
        },
      });
    } catch (error: any) {
      console.error("Case analysis failed:", error);

      await auditService.logAction({
        type: "CASE_ANALYSIS_FAILED",
        entityType: "CASE",
        entityId: caseId,
        userId: currentUser?.id || "unknown",
        details: { error: error.message },
      });

      throw error;
    } finally {
      loading.analysis = false;
    }
  }

  async function analyzeDocument(documentId: string): Promise<void> {
    loading.analysis = true;

    try {
      await auditService.logAction({
        type: "DOCUMENT_ANALYSIS_REQUESTED",
        entityType: "DOCUMENT",
        entityId: documentId,
        userId: currentUser?.id || "unknown",
      });

      const response = await fetch(`/api/documents/${documentId}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        throw new Error(`Document analysis failed: ${response.statusText}`);
      }

      const insights = await response.json();
      aiInsights[documentId] = insights;

      await auditService.logAction({
        type: "DOCUMENT_ANALYSIS_COMPLETED",
        entityType: "DOCUMENT",
        entityId: documentId,
        userId: currentUser?.id || "unknown",
        details: {
          complianceScore: insights.complianceChecks?.length || 0,
          riskLevel: insights.riskAssessment?.level,
        },
      });
    } catch (error: any) {
      console.error("Document analysis failed:", error);

      await auditService.logAction({
        type: "DOCUMENT_ANALYSIS_FAILED",
        entityType: "DOCUMENT",
        entityId: documentId,
        userId: currentUser?.id || "unknown",
        details: { error: error.message },
      });

      throw error;
    } finally {
      loading.analysis = false;
    }
  }

  async function updateCaseStatus(caseId: string, newStatus: "active" | "pending" | "closed" | "archived"): Promise<any> {
    const caseIndex = cases.findIndex((c) => c.id === caseId);
    if (caseIndex === -1) return;

    const oldStatus = cases[caseIndex].status;
    cases[caseIndex].status = newStatus;

    try {
      await fetch(`/api/cases/${caseId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });

      await auditService.logAction({
        type: "CASE_STATUS_UPDATED",
        entityType: "CASE",
        entityId: caseId,
        userId: currentUser?.id || "unknown",
        details: { oldStatus, newStatus },
      });
    } catch (error: any) {
      // Rollback on failure
      cases[caseIndex].status = oldStatus;
      throw error;
    }
  }

  function setCurrentUser(user: User) {
    (currentUser as any) = user;
  }

  // Search functionality
  function searchCases(query: string) {
    const searchTerm = query.toLowerCase();
    return filteredCases.filter(
      (legalCase) =>
        legalCase.title.toLowerCase().includes(searchTerm) ||
        legalCase.caseNumber.toLowerCase().includes(searchTerm) ||
        legalCase.description?.toLowerCase().includes(searchTerm),
    );
  }

  // Export the store interface
  return {
    // Readonly state
    get cases() {
      return cases;
    },
    get selectedCase() {
      return selectedCase;
    },
    get aiInsights() {
      return aiInsights;
    },
    get auditLog() {
      return auditLog;
    },
    get currentUser() {
      return currentUser;
    },
    get loading() {
      return loading;
    },
    get filteredCases() {
      return filteredCases;
    },
    get caseStats() {
      return caseStats;
    },

    // Actions
    loadCases,
    selectCase,
    analyzeCase,
    analyzeDocument,
    updateCaseStatus,
    setCurrentUser,
    searchCases,
  };
}

// Global store instance
export const legalCaseStore = createLegalCaseStore();
;