
// Fixed MCP Graph Reader with proper Drizzle ORM query patterns
import { db } from "$lib/server/db";
import {
  cases,
  evidence,
  reports,
  users,
  ragSessions,
} from "$lib/server/db/schema-unified";
import { eq, sql, and, or, desc, count, like } from "drizzle-orm";

export interface GraphQuery {
  nodeTypes?: string[];
  userId?: string;
  caseId?: string;
  searchTerm?: string;
  includeAI?: boolean;
  maxDepth?: number;
  limit?: number;
}

export class MCPGraphReader {
  /**
   * Read case nodes with proper Drizzle ORM query patterns
   */
  private static async readCaseNodes(query: GraphQuery): Promise<{
    nodes: any[];
    relations: any[];
  }> {
    // Build conditions array
    const conditions = [];
    if (query.userId) {
      conditions.push(eq(cases.createdBy, query.userId));
    }
    if (query.caseId) {
      conditions.push(eq(cases.id, query.caseId));
    }

    // Build and execute query with proper where clause
    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

    const caseQuery = db
      .select({
        case: cases,
        creator: {
          id: users.id,
          name: users.name,
          email: users.email,
        },
      })
      .from(cases)
      .leftJoin(users, eq(cases.createdBy, users.id));

    const caseData = whereClause
      ? await caseQuery.where(whereClause).execute()
      : await caseQuery.execute();

    const nodes = caseData.map((item) => ({
      id: item.case.id,
      type: "case",
      label: item.case.title,
      data: {
        title: item.case.title,
        description: item.case.description,
        status: item.case.status,
        priority: item.case.priority,
        caseType: item.case.category,
        creator: item.creator,
      },
      connections: [],
      metadata: {
        createdAt: item.case.createdAt,
        updatedAt: item.case.updatedAt,
        weight:
          item.case.priority === "high"
            ? 10
            : item.case.priority === "medium"
              ? 7
              : 5,
      },
    }));

    const relations = caseData
      .filter((item) => item.creator)
      .map((item) => ({
        from: item.creator!.id,
        to: item.case.id,
        type: "owns" as const,
        weight: 8,
        metadata: { relationship: "case_owner" },
      }));

    return { nodes, relations };
  }

  /**
   * Read evidence nodes with proper query patterns
   */
  private static async readEvidenceNodes(query: GraphQuery): Promise<{
    nodes: any[];
    relations: any[];
  }> {
    // Build conditions
    const conditions = [];
    if (query.userId) {
      conditions.push(eq(evidence.uploadedBy, query.userId));
    }
    if (query.caseId) {
      conditions.push(eq(evidence.caseId, query.caseId));
    }

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

    const evidenceQuery = db
      .select({
        evidence: evidence,
        case: {
          id: cases.id,
          title: cases.title,
        },
        creator: {
          id: users.id,
          name: users.name,
        },
      })
      .from(evidence)
      .leftJoin(cases, eq(evidence.caseId, cases.id))
      .leftJoin(users, eq(evidence.uploadedBy, users.id));

    const evidenceData = whereClause
      ? await evidenceQuery.where(whereClause).execute()
      : await evidenceQuery.execute();

    const nodes = evidenceData.map((item) => ({
      id: item.evidence.id,
      type: "evidence",
      label: item.evidence.title,
      data: {
        title: item.evidence.title,
        description: item.evidence.description,
        evidenceType: item.evidence.evidenceType,
        filePath: item.evidence.fileUrl,
        fileSize: item.evidence.fileSize,
        mimeType: item.evidence.mimeType,
        tags: item.evidence.tags,
        aiTags: item.evidence.aiTags,
        case: item.case,
        creator: item.creator,
      },
      connections: [],
      metadata: {
        createdAt: item.evidence.uploadedAt,
        weight:
          item.evidence.evidenceType === "critical"
            ? 10
            : item.evidence.evidenceType === "digital"
              ? 8
              : 6,
      },
    }));

    const relations = [
      ...evidenceData
        .filter((item) => item.case)
        .map((item) => ({
          from: item.evidence.id,
          to: item.case!.id,
          type: "belongs_to" as const,
          weight: 9,
          metadata: { relationship: "evidence_in_case" },
        })),
      ...evidenceData
        .filter((item) => item.creator)
        .map((item) => ({
          from: item.creator!.id,
          to: item.evidence.id,
          type: "owns" as const,
          weight: 7,
          metadata: { relationship: "evidence_owner" },
        })),
    ];

    return { nodes, relations };
  }

  /**
   * Read report nodes with proper query patterns
   */
  private static async readReportNodes(query: GraphQuery): Promise<{
    nodes: any[];
    relations: any[];
  }> {
    // Build conditions
    const conditions = [];
    if (query.userId) {
      conditions.push(eq(reports.createdBy, query.userId));
    }
    if (query.caseId) {
      conditions.push(eq(reports.caseId, query.caseId));
    }

    const whereClause = conditions.length > 0 ? and(...conditions) : undefined;

    const reportQuery = db
      .select({
        report: reports,
        case: {
          id: cases.id,
          title: cases.title,
        },
        creator: {
          id: users.id,
          name: users.name,
        },
      })
      .from(reports)
      .leftJoin(cases, eq(reports.caseId, cases.id))
      .leftJoin(users, eq(reports.createdBy, users.id));

    const reportData = whereClause
      ? await reportQuery.where(whereClause).execute()
      : await reportQuery.execute();

    const nodes = reportData.map((item) => ({
      id: item.report.id,
      type: "report",
      label: item.report.title,
      data: {
        title: item.report.title,
        content: item.report.content,
        reportType: item.report.reportType,
        status: item.report.status,
        aiAnalysis: item.report.metadata,
        case: item.case,
        creator: item.creator,
      },
      connections: [],
      metadata: {
        createdAt: item.report.createdAt,
        updatedAt: item.report.updatedAt,
        weight:
          item.report.reportType === "person_of_interest"
            ? 9
            : item.report.reportType === "case_analysis"
              ? 8
              : 6,
      },
    }));

    const relations = [
      ...reportData
        .filter((item) => item.case)
        .map((item) => ({
          from: item.report.id,
          to: item.case!.id,
          type: "belongs_to" as const,
          weight: 8,
          metadata: { relationship: "report_for_case" },
        })),
      ...reportData
        .filter((item) => item.creator)
        .map((item) => ({
          from: item.creator!.id,
          to: item.report.id,
          type: "generated_from" as const,
          weight: 7,
          metadata: { relationship: "report_generator" },
        })),
    ];

    return { nodes, relations };
  }

  /**
   * Main read graph method
   */
  static async readGraph(query: GraphQuery): Promise<{
    nodes: any[];
    relations: any[];
    metadata: { totalNodes: number; queryTime: number; mcpSource: string };
  }> {
    const startTime = Date.now();
    const nodes: any[] = [];
    const relations: any[] = [];

    try {
      if (!query.nodeTypes || query.nodeTypes.includes("case")) {
        const caseNodes = await this.readCaseNodes(query);
        nodes.push(...caseNodes.nodes);
        relations.push(...caseNodes.relations);
      }

      if (!query.nodeTypes || query.nodeTypes.includes("evidence")) {
        const evidenceNodes = await this.readEvidenceNodes(query);
        nodes.push(...evidenceNodes.nodes);
        relations.push(...evidenceNodes.relations);
      }

      if (!query.nodeTypes || query.nodeTypes.includes("report")) {
        const reportNodes = await this.readReportNodes(query);
        nodes.push(...reportNodes.nodes);
        relations.push(...reportNodes.relations);
      }

      return {
        nodes,
        relations,
        metadata: {
          totalNodes: nodes.length,
          queryTime: Date.now() - startTime,
          mcpSource: "drizzle-postgres-graph-reader",
        },
      };
    } catch (error: any) {
      console.error("Graph reading error:", error);
      throw new Error(
        `Failed to read graph: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }
}

export default MCPGraphReader;
