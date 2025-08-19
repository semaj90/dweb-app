import { legalAnalysisSessions } from "./schema-postgres";
// @ts-nocheck

export type InsertLegalAnalysisSession =
  typeof legalAnalysisSessions.$inferInsert;
