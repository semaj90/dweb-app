// @ts-nocheck
// Production database query utilities with type safety
import { eq, and, or, like, desc, asc, sql, SQL, count } from "drizzle-orm";
import type { PgColumn, PgTable } from "drizzle-orm/pg-core";

export interface QueryFilters {
  search?: string;
  status?: string;
  priority?: string;
  caseId?: string;
  evidenceType?: string;
  activityType?: string;
  assignedTo?: string;
  threatLevel?: string;
  userId?: string;
  limit?: number;
  offset?: number;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
}

export interface PaginationParams {
  page: number;
  limit: number;
  offset: number;
}

export class QueryBuilder {
  static buildFilters(table: any, filters: QueryFilters): SQL[] {
    const conditions: SQL[] = [];

    // Search filters
    if (filters.search) {
      const searchConditions: SQL[] = [];

      if (table.title)
        searchConditions.push(like(table.title, `%${filters.search}%`));
      if (table.description)
        searchConditions.push(like(table.description, `%${filters.search}%`));
      if (table.name)
        searchConditions.push(like(table.name, `%${filters.search}%`));
      if (table.firstName)
        searchConditions.push(like(table.firstName, `%${filters.search}%`));
      if (table.lastName)
        searchConditions.push(like(table.lastName, `%${filters.search}%`));
      if (table.socialSecurityNumber)
        searchConditions.push(
          like(table.socialSecurityNumber, `%${filters.search}%`)
        );
      if (table.driversLicense)
        searchConditions.push(
          like(table.driversLicense, `%${filters.search}%`)
        );

      if (searchConditions.length > 0) {
        conditions.push(or(...searchConditions));
      }
    }

    // Status filters
    if (filters.status && table.status) {
      conditions.push(eq(table.status, filters.status));
    }

    // Priority filters
    if (filters.priority && table.priority) {
      conditions.push(eq(table.priority, filters.priority));
    }

    // Case ID filters
    if (filters.caseId && table.caseId) {
      conditions.push(eq(table.caseId, filters.caseId));
    }

    // Evidence type filters
    if (filters.evidenceType && table.evidenceType) {
      conditions.push(eq(table.evidenceType, filters.evidenceType));
    }

    // Activity type filters
    if (filters.activityType && table.activityType) {
      conditions.push(eq(table.activityType, filters.activityType));
    }

    // Assignment filters
    if (filters.assignedTo && table.assignedTo) {
      conditions.push(eq(table.assignedTo, filters.assignedTo));
    }

    // Threat level filters
    if (filters.threatLevel && table.threatLevel) {
      conditions.push(eq(table.threatLevel, filters.threatLevel));
    }

    // User ID filters
    if (filters.userId && table.userId) {
      conditions.push(eq(table.userId, filters.userId));
    }

    return conditions;
  }

  static applyFilters(conditions: SQL[]): SQL | undefined {
    return conditions.length > 0 ? and(...conditions) : undefined;
  }

  static applySorting(
    table: any,
    sortBy: string,
    order: "asc" | "desc" = "desc"
  ): SQL {
    const column = table[sortBy];
    if (!column) {
      // Default to updatedAt or createdAt
      const defaultColumn = table.updatedAt || table.createdAt || table.id;
      return order === "asc" ? asc(defaultColumn) : desc(defaultColumn);
    }
    return order === "asc" ? asc(column) : desc(column);
  }

  static getPaginationParams(
    page?: string | null,
    limit?: string | null
  ): PaginationParams {
    const pageNum = Math.max(1, parseInt(page || "1"));
    const limitNum = Math.min(100, Math.max(1, parseInt(limit || "20")));
    const offset = (pageNum - 1) * limitNum;

    return { page: pageNum, limit: limitNum, offset };
  }

  static async executeQuery<T>(
    baseQuery: any,
    filters: QueryFilters,
    table: any
  ): Promise<{ data: T[]; total: number; pagination: PaginationParams }> {
    // Build filter conditions
    const conditions = this.buildFilters(table, filters);
    const whereClause = this.applyFilters(conditions);

    // Apply filters to query
    let query = baseQuery;
    if (whereClause) {
      query = query.where(whereClause);
    }

    // Apply sorting
    const sortBy = filters.sortBy || "updatedAt";
    const sortOrder = filters.sortOrder || "desc";
    query = query.orderBy(this.applySorting(table, sortBy, sortOrder));

    // Get pagination params
    const pagination = this.getPaginationParams(
      filters.limit?.toString(),
      filters.offset
        ? (Math.floor(filters.offset / (filters.limit || 20)) + 1).toString()
        : "1"
    );

    // Apply pagination
    query = query.limit(pagination.limit).offset(pagination.offset);

    // Execute main query
    const data = await query.execute();

    // Get total count
    let countQuery = baseQuery.select({ count: count() });
    if (whereClause) {
      countQuery = countQuery.where(whereClause);
    }
    const [{ count: total }] = await countQuery.execute();

    return { data, total, pagination };
  }
}

// Export helper functions
export const {
  buildFilters,
  applyFilters,
  applySorting,
  getPaginationParams,
  executeQuery,
} = QueryBuilder;

export default QueryBuilder;
