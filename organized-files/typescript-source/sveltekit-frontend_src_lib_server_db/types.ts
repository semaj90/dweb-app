// @ts-nocheck
import { legalAnalysisSessions } from "./schema-postgres";

export type InsertLegalAnalysisSession =
  typeof legalAnalysisSessions.$inferInsert;
