
// Production query builder fix for Drizzle ORM type safety
import { eq, and, or, like, desc, sql } from "drizzle-orm";
import { db } from "$lib/server/db";
import { cases, evidence, reports } from "$lib/server/db/schema-postgres";

// Fixed query builder implementation
export class DrizzleQueryBuilder {
  static async getCases(query: any) {
    let caseQuery = db
      .select({
        case: cases,
      })
      .from(cases);

    if (query.userId) {
      caseQuery = caseQuery.where(eq(cases.createdBy, query.userId)) as any;
    }

    if (query.caseId) {
      caseQuery = caseQuery.where(eq(cases.id, query.caseId)) as any;
    }

    return await caseQuery.execute();
  }

  static async getEvidence(query: any) {
    let evidenceQuery = db
      .select({
        evidence: evidence,
      })
      .from(evidence);

    if (query.userId) {
      evidenceQuery = evidenceQuery.where(
        eq(evidence.uploadedBy, query.userId)
      ) as any;
    }

    if (query.caseId) {
      evidenceQuery = evidenceQuery.where(
        eq(evidence.caseId, query.caseId)
      ) as any;
    }

    return await evidenceQuery.execute();
  }

  static async getReports(query: any) {
    let reportQuery = db
      .select({
        report: reports,
      })
      .from(reports);

    if (query.userId) {
      reportQuery = reportQuery.where(
        eq(reports.createdBy, query.userId)
      ) as any;
    }

    if (query.caseId) {
      reportQuery = reportQuery.where(eq(reports.caseId, query.caseId)) as any;
    }

    return await reportQuery.execute();
  }
}
